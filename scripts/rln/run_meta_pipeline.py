"""Run E5 (Ren), E6 (offline selection), and E7 (selection + Ren) from one M0."""
import argparse
import copy
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from data_utils import prepare_data, prepare_synthetic, frame_sha256, read_json, write_json
from gradient_weighting import (
    IndexedWeights, check_data_contract, head_features, head_gradients, install_ren,
)
from run_ablation_ram import fit_in_ram
from run_experiments import finish, record
from scoring import release_gpu
from train_seed import build_seed_trainer
from training import build_stage_model, build_stage_trainer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=['phrasebank', 'twitter', 'finsent'],
                   default=['phrasebank', 'twitter', 'finsent'])
    p.add_argument('--experiments', nargs='+', choices=['E5', 'E6', 'E7'],
                   default=['E5', 'E6', 'E7'])
    p.add_argument('--train-seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=24266)
    p.add_argument('--meta-seed', type=int, default=24266)
    p.add_argument('--meta-method', choices=['clustered', 'coverage', 'random'], default='clustered')
    p.add_argument('--meta-sweep', action='store_true', help='E5: compare cluster, coverage and random.')
    p.add_argument('--random-meta-seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--less-balanced-fractions', nargs='+', type=Fraction)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--meta-per-class', type=int, default=15)
    p.add_argument('--less-keep-fraction', type=float, default=1/3)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--ren-inner-lr', type=float, default=1e-4)
    p.add_argument('--num-epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--freeze-layers', type=int, choices=range(7), default=4)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--score-batch-size', type=int, default=32)
    p.add_argument('--outputs-dir', type=Path, default=Path('outputs'))
    p.add_argument('--output-root', type=Path, default=Path('output/e5_e7'))
    p.add_argument('--m0-root', type=Path, default=Path('output'))
    p.add_argument('--rebuild-missing-m0', action='store_true')
    p.add_argument('--phrasebank-file', type=Path, default=Path(
        'data/financial_phrasebank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'))
    p.add_argument('--twitter-train-file', type=Path)
    p.add_argument('--twitter-validation-file', type=Path)
    p.add_argument('--finsent-data-dir', type=Path, default=Path('data/finsent/prepared'))
    p.add_argument('--reference-root', type=Path,
                   help='Optional E1–E4 result root with input_sha256.json for each dataset.')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if not 0 < args.less_keep_fraction <= 1:
        p.error('--less-keep-fraction must be in (0, 1].')
    if min(args.meta_per_class, args.score_batch_size, args.num_epochs, args.patience) < 1:
        p.error('Counts and epochs must be positive.')
    if any(not math.isfinite(x) or x <= 0 for x in (args.learning_rate, args.ren_inner_lr)):
        p.error('Learning rates must be finite and positive.')
    if bool(args.twitter_train_file) != bool(args.twitter_validation_file):
        p.error('Supply both local Twitter files or neither.')
    if min([args.meta_seed, *args.random_meta_seeds]) < 0:
        p.error('Meta seeds must be nonnegative.')
    if args.meta_sweep and set(args.experiments) != {'E5'}:
        p.error('--meta-sweep requires --experiments E5.')
    if args.less_balanced_fractions and (set(args.experiments) != {'E6'} or
            any(not 0 < f <= 1 for f in args.less_balanced_fractions)):
        p.error('--less-balanced-fractions requires E6 and fractions in (0, 1].')
    return args


def coverage_indices(embeddings, count):
    center = embeddings.mean(axis=0)
    chosen = [int(np.square(embeddings - center).sum(axis=1).argmin())]
    nearest = np.square(embeddings - embeddings[chosen[0]]).sum(axis=1)
    while len(chosen) < count:
        nearest[chosen] = -np.inf
        index = int(nearest.argmax())
        chosen.append(index)
        nearest = np.minimum(nearest, np.square(embeddings - embeddings[index]).sum(axis=1))
    return chosen


def select_meta_sets(frames, synthetic, args, specs):
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
    if any(method != 'random' for method, _ in specs):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if any(method == 'clustered' for method, _ in specs):
        from sklearn.cluster import KMeans
    parts = {spec: [] for spec in specs}
    for label in range(3):
        rows = pool[pool.label == label].reset_index(drop=True)
        count = args.meta_per_class
        if len(rows) < count:
            raise ValueError(f'Class {label} has too few eligible meta examples.')
        embeddings = None
        if encoder is not None:
            embeddings = encoder.encode(rows.sentence.tolist(), normalize_embeddings=True,
                                        batch_size=args.score_batch_size)
        for method, seed in specs:
            if method == 'random':
                indices = np.random.default_rng(np.random.SeedSequence([seed, label])).choice(
                    len(rows), count, replace=False)
            elif method == 'coverage':
                indices = coverage_indices(embeddings, count)
            else:
                clustering = KMeans(n_clusters=count, n_init=10, random_state=seed).fit(embeddings)
                indices = []
                for cluster, center in enumerate(clustering.cluster_centers_):
                    members = np.flatnonzero(clustering.labels_ == cluster)
                    if not len(members):
                        raise ValueError(f'Empty meta cluster in class {label}.')
                    nearest = np.square(embeddings[members] - center).sum(axis=1).argmin()
                    indices.append(members[nearest])
            parts[(method, seed)].append(rows.iloc[indices])
    return {spec: pd.concat(rows, ignore_index=True).drop(columns='key')
            for spec, rows in parts.items()}


def balanced_selection(ranking, fraction):
    counts = ranking.label.value_counts()
    if len(counts) != 3:
        raise ValueError('Balanced selection requires synthetic candidates in all three labels.')
    # Common quota based on the smallest available label. Round half up.
    budget = int(int(counts.min()) * fraction + Fraction(1, 2))
    if budget < 1:
        raise ValueError('Fraction is too small to retain one example per label.')
    return ranking.groupby('label', sort=False).head(budget).sort_values(
        'synthetic_index').reset_index(drop=True)


def rank_synthetic(model, tokenizer, synthetic, meta, args, config):
    """Single M0, classifier gradients, mean cosine to balanced references."""
    vectors = []
    for frame in (synthetic, meta):
        hidden, logits = head_features(model, tokenizer, frame, args.score_batch_size,
                                      config['max_length'])
        grad = head_gradients(hidden, logits, torch.tensor(frame.label.tolist()),
                              config['label_smoothing'])
        vectors.append(torch.nn.functional.normalize(grad, dim=1))
    scores = (vectors[0] @ vectors[1].mean(dim=0)).numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite synthetic ranking scores.')
    count = math.ceil(len(synthetic) * args.less_keep_fraction)
    ranking = synthetic.assign(less_score=scores).sort_values(
        ['less_score', 'synthetic_index'], ascending=[False, True])
    ranking['selected'] = np.arange(len(ranking)) < count
    return ranking


def checkpoint_path(args, dataset):
    return args.m0_root / dataset / 'seed_only' / f'train_seed_{args.train_seed}' / 'best_model'


def get_m0(args, dataset, frames, config, folder, hashes):
    checkpoint = checkpoint_path(args, dataset)
    if checkpoint.is_dir():
        path = checkpoint.parent / 'run_config.json'
        if path.is_file():
            saved = read_json(path)
            for name in ('seed', 'validation', 'test'):
                expected = saved.get('data_sha256', {}).get(name)
                if expected is not None and expected != hashes[name]:
                    raise ValueError(f'M0 {name} hash mismatch: {checkpoint}')
            for name in ('split_seed', 'train_seed'):
                if name in saved and saved[name] != config[name]:
                    raise ValueError(f'M0 {name} mismatch: {checkpoint}')
        model, tokenizer = build_stage_model(args, dataset, checkpoint, config)
        return model.cpu(), tokenizer, {'checkpoint': str(checkpoint.resolve()),
                                       'metadata_present': path.is_file()}
    if not args.rebuild_missing_m0:
        raise FileNotFoundError(f'Missing M0: {checkpoint}')
    warmup = folder / 'm0_in_ram'
    warmup.mkdir()
    trainer, tokenizer, _, _, _ = build_seed_trainer(SimpleNamespace(**config), warmup, frames)
    selection = fit_in_ram(trainer, args.patience)
    model = trainer.model.cpu()
    del trainer
    release_gpu()
    return model, tokenizer, {'checkpoint': None, 'trained_in_ram': True, **selection}


def run_dataset(args, dataset):
    folder = args.output_root / dataset / f'train_seed_{args.train_seed}'
    frames, audit = prepare_data(
        dataset=dataset, seed_path=args.outputs_dir / f'seed_{dataset}_data_clustered.jsonl',
        split_seed=args.split_seed, phrasebank_file=args.phrasebank_file,
        twitter_train_file=args.twitter_train_file, twitter_validation_file=args.twitter_validation_file,
        expected_seeds=105, finsent_data_dir=args.finsent_data_dir,
    )
    synthetic, removed = prepare_synthetic(
        args.outputs_dir / f'synthetic_{dataset}_data_seed_clustered.jsonl', dataset, frames)
    if dataset == 'finsent' and (
            len(synthetic) != 945 or removed or not synthetic.label.value_counts().eq(315).all()):
        raise ValueError('FinSent requires 945 unique synthetic texts, 315 per class.')
    synthetic = synthetic.reset_index(drop=True).assign(synthetic_index=lambda df: df.index)
    hashes = {name: frame_sha256(frames[name]) for name in ('seed', 'validation', 'test')}
    hashes['synthetic'] = frame_sha256(synthetic)
    if args.reference_root:
        reference = read_json(args.reference_root / dataset / folder.name / 'input_sha256.json')
        differing = [name for name, value in hashes.items() if reference.get(name) != value]
        if differing:
            raise ValueError(f'{dataset}: inputs differ from baseline: {differing}')
    set_seed(args.train_seed)
    specs = [(args.meta_method, args.meta_seed)]
    if args.meta_sweep:
        specs = [('clustered', args.meta_seed), ('coverage', args.meta_seed)]
        specs.extend(('random', seed) for seed in dict.fromkeys(args.random_meta_seeds))
    meta_sets = select_meta_sets(frames, synthetic, args, specs)
    frames['meta'] = meta_sets[specs[0]]
    for meta in meta_sets.values():
        check_data_contract({**frames, 'meta': meta}, synthetic)
    release_gpu()
    contract = check_data_contract(frames, synthetic)
    contract.pop('training_rows_E1', None)
    contract.pop('training_rows_E2_to_E7', None)
    folder.mkdir(parents=True)
    for (method, seed), meta in meta_sets.items():
        meta.to_csv(folder / f'meta_{method}_{seed}.csv', index=False)
    frames['meta'].to_csv(folder / 'meta_samples.csv', index=False)
    write_json(folder / 'input_sha256.json', {**hashes, 'meta': frame_sha256(frames['meta'])})
    write_json(folder / 'data_contract.json', contract)
    write_json(folder / 'data_audit.json', audit)
    write_json(folder / 'excluded_synthetic.json', removed)
    print(f'[DATA OK] {dataset}: {contract["counts"]}', flush=True)
    if args.prepare_only:
        return

    config = dict(dataset=dataset, train_seed=args.train_seed, split_seed=args.split_seed,
                  model='distilbert-base-uncased', model_revision='main',
                  freeze_layers=args.freeze_layers, max_length=512, train_batch_size=16,
                  eval_batch_size=32, learning_rate=args.learning_rate,
                  num_epochs=args.num_epochs, early_stopping_patience=args.patience,
                  weight_decay=args.weight_decay, label_smoothing=0.1)
    write_json(folder / 'run_config.json', {
        'training': config,
        'options': {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k != 'less_balanced_fractions'},
            'less_balanced_fractions': [str(f) for f in args.less_balanced_fractions or []],
        },
        'ranking': 'Offline classifier gradient cosine at M0; mean over balanced meta; no Adam moments',
        'ren': 'Classifier-only virtual SGD on synthetic, rectification and batch-mean normalization',
    })
    m0, tokenizer, identity = get_m0(args, dataset, frames, config, folder, hashes)
    write_json(folder / 'm0.json', identity)
    selected = None
    if {'E6', 'E7'}.intersection(args.experiments):
        m0.to('cuda' if torch.cuda.is_available() else 'cpu')
        ranking = rank_synthetic(m0, tokenizer, synthetic, frames['meta'], args, config)
        m0.cpu()
        ranking_export = ranking.drop(columns='selected') if args.less_balanced_fractions else ranking
        ranking_export.to_csv(folder / 'synthetic_ranking.csv', index=False)
        selected = ranking[ranking.selected].sort_values('synthetic_index').reset_index(drop=True)
        release_gpu()

    jobs = []
    for experiment in dict.fromkeys(args.experiments):
        if args.meta_sweep:
            jobs.extend((experiment, f'E5_{method}_{seed}', meta, synthetic,
                         {'meta_method': method, 'meta_seed': seed})
                        for (method, seed), meta in meta_sets.items())
        elif args.less_balanced_fractions:
            for fraction in dict.fromkeys(args.less_balanced_fractions):
                subset = balanced_selection(ranking, fraction)
                jobs.append((experiment, f'E6_balanced_{fraction.numerator}_{fraction.denominator}',
                             frames['meta'], subset,
                             {'keep_fraction': str(fraction), 'synthetic_per_label': len(subset)//3,
                              'meta_method': args.meta_method, 'meta_seed': args.meta_seed}))
        else:
            jobs.append((experiment, experiment, frames['meta'],
                         synthetic if experiment == 'E5' else selected, {}))

    for experiment, variant, meta, stage_syn, details in jobs:
        stage = folder / variant
        stage.mkdir()
        stage_frames = {**frames, 'meta': meta}
        stage_syn.assign(branch_index=np.arange(len(stage_syn))).to_csv(
            stage / 'training_synthetic.csv', index=False)
        clean = frames['seed'][['sentence', 'label']]
        train = pd.concat([clean, stage_syn[['sentence', 'label']]], ignore_index=True).assign(weight=1.)
        set_seed(args.train_seed)
        model = copy.deepcopy(m0)
        trainer = build_stage_trainer(args, stage, config, stage_frames, train, model, tokenizer)
        trainer.label_names = ['labels']
        if experiment in ('E5', 'E7'):
            trainer.train_dataset = IndexedWeights(trainer.train_dataset, len(clean), len(stage_syn))
            install_ren(trainer, tokenizer, meta, stage, args.ren_inner_lr,
                        batch_size=args.score_batch_size, max_length=config['max_length'],
                        smoothing=config['label_smoothing'], pure=True)
        print(f'[TRAIN] {dataset}/{variant}: seed={len(clean)}, syn={len(stage_syn)}', flush=True)
        best = fit_in_ram(trainer, args.patience)
        metrics = finish(trainer, tokenizer, stage_frames, config, stage, dataset, True, best,
                         test=not args.validation_only)
        record(args.output_root / 'comparison.csv', dataset, args.train_seed, experiment,
               {**metrics, 'variant': variant, **details, 'synthetic_count': len(stage_syn),
                'meta_count': len(meta)})
        del trainer, model
        release_gpu()


def main():
    args = parse_args()
    if not args.prepare_only and torch.cuda.device_count() > 1:
        raise ValueError('Run with CUDA_VISIBLE_DEVICES=0 for a single training GPU.')
    datasets = list(dict.fromkeys(args.datasets))
    for dataset in datasets:
        folder = args.output_root / dataset / f'train_seed_{args.train_seed}'
        if folder.exists():
            raise FileExistsError(f'Choose a fresh --output-root: {folder}')
        if not (args.prepare_only or args.rebuild_missing_m0 or checkpoint_path(args, dataset).is_dir()):
            raise FileNotFoundError(f'Missing M0: {checkpoint_path(args, dataset)}')
    for dataset in datasets:
        run_dataset(args, dataset)


if __name__ == '__main__':
    main()
