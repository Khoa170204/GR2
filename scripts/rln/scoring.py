"""Static R/L/N scoring, probabilities, clipped weights and score caching."""

import gc

import numpy as np
import pandas as pd

from data_utils import read_json, write_json, fingerprint, file_sha256, normalize_label


def release_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assigned_label_probabilities(
    model_path,
    frame,
    dataset,
    batch_size,
    max_length,
):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = (
        AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device).eval()
    )
    # FinBERT và student có thể khác thứ tự label.
    # Ghép theo tên nhãn, không giả định index.
    column = {
        normalize_label(name, dataset): int(index)
        for index, name
        in model.config.id2label.items()
    }
    if set(column) != {0, 1, 2}:
        raise ValueError('Scoring model must expose semantic names for all three labels.')
    result = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[
                start:start + batch_size
            ]
            encoded = tokenizer(
                batch["sentence"].tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            probabilities = (
                model(**encoded).logits.float().softmax(-1)
            )
            indices = torch.tensor([column[int(label)] for label in batch['label']], device=device)
            selected = probabilities.gather(1, indices[:, None]).squeeze(1)
            result.extend(selected.cpu().tolist())
    del model, tokenizer
    release_gpu()
    return np.asarray(result, dtype=np.float64)


def make_weights(R, L, N, minimum, maximum):
    Q = R * L * N
    if (
        not np.isfinite(Q).all()
        or Q.mean() <= 1e-12
    ):
        raise ValueError('Invalid or near-zero mean RLN score. Inspect component scores.')
    weights = np.clip(Q / Q.mean(), minimum, maximum)
    return Q, weights


def score_synthetic(
    args,
    dataset,
    checkpoint,
    frames,
    synthetic,
    contract,
):
    folder = (
        args.output_root
        / dataset
        / "rln_scores"
        / f"train_seed_{args.train_seed}"
    )
    settings = {
        "input": contract,
        "reliability_model": (
            args.reliability_model
        ),
        "embedding_model": args.embedding_model,
        "minimum": args.weight_min,
        "maximum": args.weight_max,
        "formula": (
            "Q=R*(1-pM0)*(1-max_cos); "
            "w=clip(Q/meanQ)"
        ),
    }
    signature = fingerprint(settings)
    marker = folder / "complete.json"
    path = folder / "scored_synthetic.jsonl"
    if marker.exists():
        saved = read_json(marker)
        if (
            saved["signature"] != signature
            or file_sha256(path)
            != saved["file_sha256"]
        ):
            raise ValueError('Cached RLN scores do not match this run.')
        print('[SCORES SKIP] Using cached RLN scores', flush=True)
        return (pd.read_json(path, lines=True), signature)
    folder.mkdir(parents=True, exist_ok=True)
    print('[SCORING] R: reliability checker', flush=True)
    R = assigned_label_probabilities(
        args.reliability_model,
        synthetic,
        dataset,
        args.score_batch_size,
        512,
    )
    print('[SCORING] L: initial student M0', flush=True)
    L = 1.0 - assigned_label_probabilities(
        checkpoint,
        synthetic,
        dataset,
        args.score_batch_size,
        512,
    )
    print('[SCORING] N: SBERT similarity to seed', flush=True)
    from sentence_transformers import (
        SentenceTransformer,
    )
    encoder = SentenceTransformer(args.embedding_model)
    seed_embeddings = encoder.encode(
        frames["seed"]["sentence"].tolist(),
        normalize_embeddings=True,
        batch_size=args.score_batch_size,
        show_progress_bar=True,
    )
    embeddings = encoder.encode(
        synthetic["sentence"].tolist(),
        normalize_embeddings=True,
        batch_size=args.score_batch_size,
        show_progress_bar=True,
    )
    similarity = np.clip(embeddings @ seed_embeddings.T, -1.0, 1.0)
    N = 1.0 - similarity.max(axis=1)
    del encoder
    release_gpu()
    Q, weights = make_weights(R, L, N, args.weight_min, args.weight_max)
    scored = synthetic.copy()
    for name, values in {'R': R, 'L': L, 'N': N, 'Q': Q, 'weight': weights}.items():
        scored[name] = values
    scored.to_json(path, orient='records', lines=True, force_ascii=False, double_precision=15)
    write_json(folder / 'score_config.json', settings)
    write_json(
        folder / "score_statistics.json",
        {
            name: {
                "min": float(scored[name].min()),
                "mean": float(scored[name].mean()),
                "max": float(scored[name].max()),
            }
            for name in ('R', 'L', 'N', 'Q', 'weight')
        },
    )
    write_json(marker, {'signature': signature, 'file_sha256': file_sha256(path)})
    # Dùng cùng biểu diễn số khi chạy mới và đọc cache.
    return (pd.read_json(path, lines=True), signature)
