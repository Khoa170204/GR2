"""Weight every synthetic row, using the validation-best seed model at epoch one."""
import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback

from scoring import (assigned_label_probabilities, head_features, head_gradients,
                     make_weights, release_gpu, student_probabilities)


METHODS = {
    'E1': 'seed only',
    'E2': 'uniform',
    'E3': 'static RLN',
    'E4': 'dynamic RLN',
    'E5': 'static gradient',
    'E6': 'static RLN + static gradient',
    'E7': 'dynamic RLN + static gradient',
    'E8': 'dynamic RLN + dynamic gradient',
}
USE_RLN = {'E3', 'E4', 'E6', 'E7', 'E8'}
USE_GRADIENT = {'E5', 'E6', 'E7', 'E8'}
DYNAMIC_L = {'E4', 'E7', 'E8'}


def gradient_scores(model, tokenizer, synthetic, reference, batch_size, config):
    def unit_gradients(frame):
        h, logits = head_features(model, tokenizer, frame, batch_size, config['max_length'])
        labels = torch.tensor(frame.label.tolist(), dtype=torch.long)
        gradients = head_gradients(h, logits, labels, config['label_smoothing'])
        return torch.nn.functional.normalize(gradients, dim=1)

    # Mean of pairwise cosines. Renormalizing the reference mean changes the score.
    reference_mean = unit_gradients(reference).mean(dim=0)
    scores = (unit_gradients(synthetic) @ reference_mean).numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite gradient scores.')
    return np.clip(scores, -1, 1)


def initial_scores(experiments, model, tokenizer, frames, synthetic, batch_size, config):
    scores = {}
    if USE_RLN.intersection(experiments):
        from sentence_transformers import SentenceTransformer
        scores['R'] = assigned_label_probabilities('ProsusAI/finbert', synthetic,
                        config['dataset'], batch_size, config['max_length'])
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        seed = encoder.encode(frames['seed'].sentence.tolist(), normalize_embeddings=True,
                              batch_size=batch_size)
        syn = encoder.encode(synthetic.sentence.tolist(), normalize_embeddings=True,
                             batch_size=batch_size)
        scores['N'] = 1 - np.clip(syn @ seed.T, -1, 1).max(axis=1)
        del encoder
        release_gpu()
        scores['L'] = 1 - student_probabilities(model, tokenizer, synthetic, config['dataset'],
                                                 batch_size, config['max_length'])
    if USE_GRADIENT.intersection(experiments):
        scores['G'] = gradient_scores(model, tokenizer, synthetic, frames['meta'], batch_size, config)
    return scores


def weight_table(experiment, scores, size, blend="relative", strength=0.5):
    table = pd.DataFrame({'synthetic_index': np.arange(size)})
    base = np.ones(size)
    if experiment in USE_RLN:
        for key in 'RLN':
            table[key] = scores[key]
        table['Q'], base = make_weights(scores['R'], scores['L'], scores['N'], 0.5, 2.0)
    table['base'] = base
    if experiment in USE_GRADIENT:
        table['G'] = scores['G']
        g = scores['G']
        if blend == "relative":
            base = base * (1 + strength * g)
        elif blend == "add":
            base = base + strength * g
        elif blend == "product":
            base = base * g
        elif blend == "gradient_only":
            base = strength * g
        else:
            raise ValueError(f"Unknown blend: {blend}")
    table['weight'] = np.clip(base, 0.5, 2.0)
    return table


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


class BoundedWeights(TrainerCallback):
    def __init__(self, experiment, data, scores, tokenizer, synthetic, meta, config,
                 batch_size, stage):
        self.experiment, self.data = experiment, data
        # Reassignments of dynamic L/G stay local to this branch.
        self.scores = dict(scores)
        self.tokenizer, self.synthetic, self.meta = tokenizer, synthetic, meta
        self.config, self.batch_size = config, batch_size
        self.folder = stage / 'weights'
        self.folder.mkdir()
        self.last_epoch = 0

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch or 0)) + 1
        if epoch == self.last_epoch or (epoch > 1 and self.experiment not in DYNAMIC_L):
            return control
        if epoch > 1:
            self.scores['L'] = 1 - student_probabilities(model, self.tokenizer, self.synthetic,
                self.config['dataset'], self.batch_size, self.config['max_length'])
            if self.experiment == 'E8':
                self.scores['G'] = gradient_scores(model, self.tokenizer, self.synthetic,
                    self.meta, self.batch_size, self.config)
        table = weight_table(
            self.experiment, self.scores, len(self.synthetic),
            blend=self.config.get("blend", "relative"),
            strength=self.config.get("gradient_strength", 0.5),
        )
        self.data.update(table.weight.to_numpy())
        table.to_csv(self.folder / f'epoch_{epoch:03}.csv', index=False)
        self.last_epoch = epoch
        return control
