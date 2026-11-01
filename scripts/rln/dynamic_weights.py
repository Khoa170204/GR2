"""Epoch-wise RLN weights. Imported only by run_rln_dynamic.py."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback

from data_utils import normalize_label
from scoring import make_weights


def normalize_q(q, mode):
    """Normalize over ALL synthetic samples; equal scores get equal weight 1."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError('Q must be a nonempty finite nonnegative vector.')
    low, high = float(q.min()), float(q.max())
    if mode == 'legacy':
        _, weights = make_weights(q, np.ones_like(q), np.ones_like(q), 0.5, 2.0)
        equal = False
    else:
        equal = bool(high - low <= 1e-12)
        weights = np.ones_like(q) if equal else (q - low) / (high - low)
        weights = np.clip(weights, 0.0, 1.0)
    return weights, {'q_min': low, 'q_max': high, 'equal_q_fallback': equal}


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


class MutableWeights(torch.utils.data.Dataset):
    """Read current weights at fetch time, not a stale Arrow/Pandas copy."""
    def __init__(self, base, seed_count, synthetic_count):
        self.base = base
        self.seed_count = seed_count
        self.weights = np.ones(len(base), dtype=np.float64)
        if len(base) != seed_count + synthetic_count:
            raise ValueError('Expected seed rows first, then synthetic rows.')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = dict(self.base[int(index)])
        row['weight'] = float(self.weights[int(index)])
        return row

    def __getitems__(self, indices):
        return [self[index] for index in indices]

    def update(self, synthetic_weights):
        values = np.asarray(synthetic_weights, dtype=np.float64)
        if values.shape != self.weights[self.seed_count:].shape or not np.isfinite(values).all():
            raise ValueError('Invalid synthetic weight vector.')
        self.weights[:self.seed_count] = 1.0
        self.weights[self.seed_count:] = values


class EpochWeights(TrainerCallback):
    def __init__(self, train_data, synthetic, scores, tokenizer, dataset, variant,
                 mode, folder, batch_size=32, max_length=512):
        self.train_data, self.synthetic = train_data, synthetic
        self.scores = {key: np.asarray(value, dtype=np.float64).copy()
                       for key, value in scores.items()}
        self.tokenizer, self.dataset, self.variant = tokenizer, dataset, variant
        self.mode, self.folder = mode, Path(folder)
        self.batch_size, self.max_length = batch_size, max_length
        self.last_epoch = 0
        self.previous = None

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if args.dataloader_num_workers != 0 or args.world_size != 1:
            raise ValueError('Use one training process and dataloader_num_workers=0.')
        epoch = int(round(state.epoch or 0)) + 1
        if epoch == self.last_epoch:
            return control
        refresh = self.mode == 'dynamic' and 'L' in self.variant and epoch > 1
        if refresh:
            self.scores['L'] = 1.0 - student_probabilities(
                model, self.tokenizer, self.synthetic, self.dataset,
                self.batch_size, self.max_length)
        q = np.ones(len(self.synthetic), dtype=np.float64)
        for component in self.variant:
            q *= self.scores[component]
        weights, limits = normalize_q(q, self.mode)
        self.train_data.update(weights)
        # Every synthetic sample remains in the scoring pool, even weight-zero rows.
        output = self.folder / 'weights'
        output.mkdir(exist_ok=True)
        if epoch == 1 or refresh:
            pd.DataFrame({'synthetic_index': np.arange(len(q)), **self.scores,
                          'Q': q, 'weight': weights}).to_csv(
                              output / f'epoch_{epoch:03d}.csv', index=False)
        stats = {'epoch': epoch, 'L_recomputed': refresh, **limits,
                 'L_mean': float(self.scores['L'].mean()),
                 'weight_min': float(weights.min()), 'weight_max': float(weights.max()),
                 'weight_mean': float(weights.mean()), 'zero_weights': int((weights == 0).sum()),
                 'synthetic_weight_sum': float(weights.sum()),
                 'seed_weight_sum': self.train_data.seed_count,
                 'mean_abs_weight_change': None if self.previous is None else
                     float(np.abs(weights - self.previous).mean())}
        path = self.folder / 'weight_history.csv'
        pd.DataFrame([stats]).to_csv(path, mode='a', header=not path.exists(), index=False)
        self.previous, self.last_epoch = weights.copy(), epoch
        print(f'[WEIGHTS] {self.dataset}/{self.variant}/{self.mode}: epoch={epoch}, '
              f'L_recomputed={refresh}, range={weights.min():.4f}..{weights.max():.4f}, '
              f'mean={weights.mean():.4f}', flush=True)
        return control
