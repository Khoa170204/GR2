"""Per-example predictions, gradients of the classifier, and RLN normalization."""
import gc
import random

import numpy as np
import torch

from data_utils import normalize_label


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


def student_probabilities(model, tokenizer, frame, dataset, batch_size, max_length):
    """No model moves, no gradients; preserve mode and training RNG states."""
    device = next(model.parameters()).device
    columns = {normalize_label(name, dataset): int(index)
               for index, name in model.config.id2label.items()}
    if set(columns) != {0, 1, 2}:
        raise ValueError('Student label mapping is incomplete.')
    was_training = model.training
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == 'cuda' else []
    result = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for start in range(0, len(frame), batch_size):
                batch = frame.iloc[start:start + batch_size]
                encoded = tokenizer(batch.sentence.tolist(), padding=True, truncation=True,
                                    max_length=max_length, return_tensors='pt').to(device)
                probabilities = model(**encoded).logits.float().softmax(-1)
                indices = torch.tensor([columns[int(label)] for label in batch.label], device=device)
                result.extend(probabilities.gather(1, indices[:, None]).squeeze(1).cpu().tolist())
    finally:
        model.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    values = np.asarray(result, dtype=np.float64)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError('Invalid student probabilities.')
    return np.clip(values, 0.0, 1.0)


def head_features(model, tokenizer, frame, batch_size=32, max_length=512):
    """Capture classifier inputs in eval mode; preserve modes, RNG and .grad."""
    if not isinstance(model.classifier, torch.nn.Linear):
        raise TypeError('Expected a DistilBERT linear classifier.')
    device = next(model.parameters()).device
    modes = [(m, m.training) for m in model.modules()]
    devices = [device.index or 0] if device.type == 'cuda' else []
    capture, hidden, logits = [], [], []
    hook = model.classifier.register_forward_pre_hook(
        lambda module, args: capture.append(args[0].detach()))
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for start in range(0, len(frame), batch_size):
                rows = frame.iloc[start:start + batch_size]
                tokens = tokenizer(rows.sentence.tolist(), padding=True, truncation=True,
                                   max_length=max_length, return_tensors='pt').to(device)
                capture.clear()
                out = model(**tokens).logits
                if len(capture) != 1:
                    raise RuntimeError('Expected one final classifier call.')
                hidden.append(capture[0].float().cpu())
                logits.append(out.float().cpu())
    finally:
        hook.remove()
        for module, mode in modes:
            module.training = mode
    return torch.cat(hidden), torch.cat(logits)


def head_gradients(hidden, logits, labels, smoothing=0.1):
    p = logits.float().softmax(-1)
    target = torch.full_like(p, smoothing / p.shape[1])
    target.scatter_add_(1, labels[:, None], torch.full(
        (len(labels), 1), 1 - smoothing, device=p.device))
    residual = p - target
    return torch.cat([(residual[:, :, None] * hidden[:, None, :]).flatten(1),
                      residual], dim=1)
