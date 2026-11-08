"""Exact final-linear-layer CE gradients; epoch-wise class-balanced cosine weights.

This is a Gr2 adaptation inspired by Ren and LESS, not a reproduction of either.
Scoring uses eval mode and analytic gradients of the smoothed CE used in training.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback
from data_utils import LABELS


def split_meta(frames, synthetic, per_class, seed):
    """Reserve meta from original validation, with case/whitespace overlap checks."""
    key = lambda text: ' '.join(text.casefold().split())
    validation = frames['validation'].copy()
    validation['_key'] = validation.sentence.map(key)
    if validation['_key'].duplicated().any():
        raise ValueError('Duplicate validation texts after case/whitespace normalization.')
    forbidden = set()
    for frame in [frames['seed'], frames['test'], synthetic]:
        forbidden.update(frame.sentence.map(key))
    if set(validation['_key']) & forbidden:
        raise ValueError('Validation/meta overlaps seed, synthetic, or test.')
    parts = []
    for label in range(3):
        pool = validation[validation.label == label]
        if len(pool) <= per_class:
            raise ValueError(f'Class {label}: need more than {per_class} validation rows.')
        parts.append(pool.sample(n=per_class, random_state=seed + label))
    meta = pd.concat(parts)
    remaining = validation.drop(meta.index).drop(columns='_key').reset_index(drop=True)
    meta = meta.drop(columns='_key').reset_index(drop=True)
    return {**frames, 'validation': remaining, 'meta': meta}


def linear_ce_features(hidden, logits, labels, smoothing):
    """Concatenate d(loss)/dW and d(loss)/db for each sample, in float32."""
    probabilities = logits.float().softmax(-1)
    target = torch.full_like(probabilities, smoothing / logits.shape[-1])
    target.scatter_add_(1, labels[:, None], torch.full(
        (len(labels), 1), 1.0 - smoothing, device=logits.device))
    residual = probabilities - target
    weight_grad = residual[:, :, None] * hidden.float()[:, None, :]
    return torch.cat([weight_grad.flatten(1), residual], dim=1)


def gradient_features(model, tokenizer, frame, dataset, batch_size, max_length, smoothing):
    if {int(k): v for k, v in model.config.id2label.items()} != LABELS[dataset]:
        raise ValueError('Student label mapping differs from dataset mapping.')
    if not isinstance(model.classifier, torch.nn.Linear):
        raise TypeError('Expected DistilBERT final linear classifier.')
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    captured = []
    handle = model.classifier.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach()))
    result = []
    devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for start in range(0, len(frame), batch_size):
                batch = frame.iloc[start:start + batch_size]
                encoded = tokenizer(batch.sentence.tolist(), padding=True, truncation=True,
                                    max_length=max_length, return_tensors='pt').to(device)
                captured.clear()
                logits = model(**encoded).logits
                if len(captured) != 1:
                    raise RuntimeError('Expected one classifier call per scoring batch.')
                labels = torch.tensor(batch.label.tolist(), device=device, dtype=torch.long)
                gradient = linear_ce_features(captured[0], logits, labels, smoothing)
                if not torch.isfinite(gradient).all():
                    raise ValueError('Non-finite classifier gradient.')
                # Zero gradients have zero similarity, hence neutral weight 1.
                result.append(torch.nn.functional.normalize(gradient, dim=1).cpu())
    finally:
        handle.remove()
        for module, mode in modes:
            module.training = mode
    return torch.cat(result)


def score_to_weights(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected finite nonempty score vector.')
    scores = np.clip(scores, -1.0, 1.0)
    return np.where(scores >= 0, 1.0 + scores, 1.0 + 0.5 * scores)


class GradientWeights(TrainerCallback):
    def __init__(self, train_data, synthetic, meta, tokenizer, dataset, mode, folder,
                 batch_size=32, max_length=512, smoothing=0.1):
        self.train_data, self.synthetic, self.meta = train_data, synthetic, meta
        self.tokenizer, self.dataset, self.mode = tokenizer, dataset, mode
        self.folder = Path(folder)
        self.batch_size, self.max_length, self.smoothing = batch_size, max_length, smoothing
        self.last_epoch = 0
        self.previous = None

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if args.world_size != 1 or args.dataloader_num_workers != 0:
            raise ValueError('Use one process and zero dataloader workers.')
        epoch = int(round(state.epoch or 0)) + 1
        if epoch == self.last_epoch:
            return control
        refresh = self.previous is None or self.mode == 'dynamic'
        started = time.perf_counter()
        if refresh:
            options = (self.dataset, self.batch_size, self.max_length, self.smoothing)
            meta_vectors = gradient_features(model, self.tokenizer, self.meta, *options)
            syn_vectors = gradient_features(model, self.tokenizer, self.synthetic, *options)
            class_vectors = torch.stack([
                meta_vectors[torch.tensor(self.meta.label.to_numpy() == label)].mean(0)
                for label in range(3)])
            # Do not renormalize the class means: this equals average pairwise cosine.
            by_class = syn_vectors @ class_vectors.T
            scores = by_class.mean(1).numpy()
            weights = score_to_weights(scores)
            self.train_data.update(weights)
            destination = self.folder / 'weights'
            destination.mkdir(exist_ok=True)
            table = self.synthetic.copy()
            table['synthetic_index'] = np.arange(len(table))
            for label in range(3):
                table[f'cosine_class_{label}'] = by_class[:, label].numpy()
            table['score'], table['weight'] = scores, weights
            table.to_csv(destination / f'epoch_{epoch:03d}.csv', index=False)
        else:
            weights = self.previous.copy()
        row = {'epoch': epoch, 'refreshed': refresh,
               'score_seconds': time.perf_counter() - started,
               'weight_min': float(weights.min()), 'weight_max': float(weights.max()),
               'weight_mean': float(weights.mean()),
               'mean_abs_change': None if self.previous is None else
                   float(np.abs(weights - self.previous).mean())}
        path = self.folder / 'weight_history.csv'
        pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
        self.previous, self.last_epoch = weights.copy(), epoch
        print(f'[GRADIENT] {self.dataset}/{self.mode} epoch={epoch}: {row}', flush=True)
        return control
