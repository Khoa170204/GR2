"""Select a balanced, disjoint set of real labels for gradient scoring."""
import numpy as np
import pandas as pd


def select_meta(frames, synthetic, args):
    def key(text):
        return ' '.join(text.casefold().split())

    pool = frames['train_pool'][['sentence', 'label']].copy()
    pool['key'] = pool.sentence.map(key)
    blocked = (frames['seed'], frames['validation'], frames['test'], synthetic)
    forbidden = {key(text) for frame in blocked for text in frame.sentence}
    pool = pool[~pool.key.isin(forbidden)]
    conflicts = pool.groupby('key').label.nunique()
    pool = pool[~pool.key.isin(conflicts[conflicts > 1].index)].drop_duplicates('key')
    encoder = None
    if args.meta_method == 'clustered':
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        from sklearn.cluster import KMeans
    parts = []
    seed = args.meta_seed
    for label in range(3):
        rows = pool[pool.label == label].reset_index(drop=True)
        count = args.meta_per_class
        if len(rows) < count:
            raise ValueError(f'Class {label} has too few eligible meta examples.')
        embeddings = None
        if encoder is not None:
            embeddings = encoder.encode(rows.sentence.tolist(), normalize_embeddings=True,
                                        batch_size=args.score_batch_size)
        if args.meta_method == 'random':
            indices = np.random.default_rng(np.random.SeedSequence([seed, label])).choice(
                len(rows), count, replace=False)
        else:
            clustering = KMeans(n_clusters=count, n_init=10, random_state=seed).fit(embeddings)
            indices = []
            for cluster, center in enumerate(clustering.cluster_centers_):
                members = np.flatnonzero(clustering.labels_ == cluster)
                if not len(members):
                    raise ValueError(f'Empty meta cluster in class {label}.')
                nearest = np.square(embeddings[members] - center).sum(axis=1).argmin()
                indices.append(members[nearest])
        parts.append(rows.iloc[indices])
    return pd.concat(parts, ignore_index=True).drop(columns="key")


def check_data_contract(frames, synthetic):
    """Fail on label/column errors and cross-set overlap; do not modify data."""
    sets = {name: frames[name] for name in ['seed', 'meta', 'validation', 'test']}
    sets['synthetic'] = synthetic
    keys = {}
    for name, frame in sets.items():
        if not {'sentence', 'label'}.issubset(frame.columns) or frame.empty:
            raise ValueError(f'Invalid/empty {name} frame.')
        if frame.label.isna().any() or not set(frame.label).issubset({0, 1, 2}):
            raise ValueError(f'Invalid labels in {name}.')
        if not frame.sentence.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
            raise ValueError(f'Invalid texts in {name}.')
        keys[name] = set(frame.sentence.map(lambda s: ' '.join(s.casefold().split())))
    for i, left in enumerate(keys):
        for right in list(keys)[i + 1:]:
            if keys[left] & keys[right]:
                raise ValueError(f'Data overlap after case/whitespace normalization: {left}/{right}')


def frozen_meta(args, dataset, frames, synthetic, hashes):
    """Publish once during preparation. Training only reads the frozen file."""
    from tempfile import TemporaryDirectory
    import os
    from pathlib import Path
    from run_io import frame_sha256, read_json, write_json

    path = args.meta_dir / f'{dataset}.json'
    identity = dict(dataset=dataset, method=args.meta_method, seed=args.meta_seed,
                    per_class=args.meta_per_class, split_seed=args.split_seed,
                    inputs={**hashes, 'train_pool': frame_sha256(frames['train_pool'])})
    if not path.exists():
        if not args.prepare_meta:
            raise FileNotFoundError(f'Missing {path}. Run --prepare-meta for {dataset} first.')
        selected = select_meta(frames, synthetic, args)
        check_data_contract({**frames, 'meta': selected}, synthetic)
        payload = dict(identity=identity, sha256=frame_sha256(selected),
                       rows=selected[['sentence', 'label']].to_dict('records'))
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temp:
            ready = Path(temp) / 'meta.json'
            write_json(ready, payload)
            try:
                os.link(ready, path)  # Atomic publish; a concurrent preparer may finish first.
            except FileExistsError:
                pass

    payload = read_json(path)
    if payload['identity'] != identity:
        raise ValueError(f'{path}: data or selection settings differ. Use the matching --meta-dir.')
    meta = pd.DataFrame(payload['rows'])
    expected = {label: args.meta_per_class for label in range(3)}
    if meta.label.value_counts().to_dict() != expected or frame_sha256(meta) != payload['sha256']:
        raise ValueError(f'{path}: invalid reference counts or checksum.')
    check_data_contract({**frames, 'meta': meta}, synthetic)
    print(f'[META] {path}: {len(meta)} rows, SHA256={payload["sha256"]}', flush=True)
    return meta
