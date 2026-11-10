"""RLN priors, LESS-inspired gradient matching and Ren meta-reweighting.

Sources:
- Ren et al. (2018), Algorithm 1, https://arxiv.org/abs/1803.09050
- Xia et al. (2024), LESS Sections 3.1/4, https://arxiv.org/abs/2402.04333

Gr2 adaptations: classifier-only surrogate/features, online epoch-wise LESS,
class-balanced mean cosine by default (LESS uses max over subtasks), RLN priors,
and bounded [0.5,2] final weights. Ren gates are batch-mean normalized, detached,
and retain the prior if every rectified meta-gradient is zero. These mechanisms
are adaptations, not full reproductions of the source methods.
Only reserved meta labels guide weighting; checkpoint validation/test do not.
"""
import csv
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from dynamic_weights import MutableWeights, normalize_q, student_probabilities


def reserve_meta(frames, synthetic, per_class=15, seed=24266, source='train_pool'):
    """Default: additional clean labels from unused train pool; keep old val/test.

    source='validation' explicitly reproduces the earlier reduced-val protocol.
    Neither option changes seed/synthetic/test membership.
    """
    key = lambda text: ' '.join(text.casefold().split())
    if source not in ('train_pool', 'validation'):
        raise ValueError('Unknown meta source.')
    pool = frames[source].copy()
    pool['_key'] = pool.sentence.map(key)
    forbidden = set()
    blocked = [frames['seed'], frames['test'], synthetic]
    if source == 'train_pool':
        blocked.append(frames['validation'])
    for frame in blocked:
        forbidden.update(frame.sentence.map(key))
    if source == 'validation':
        if pool['_key'].duplicated().any() or set(pool['_key']) & forbidden:
            raise ValueError('Invalid validation pool: duplicate texts or heldout overlap.')
    else:
        pool = pool[~pool['_key'].isin(forbidden)]
        # Only meta candidates are filtered. The original train pool and all
        # baseline frames are left intact, including their row order.
        conflicts = pool.groupby('_key').label.nunique()
        pool = pool[~pool['_key'].isin(conflicts[conflicts > 1].index)]
        pool = pool.drop_duplicates('_key')
    parts = []
    for label in range(3):
        candidates = pool[pool.label == label]
        needed = per_class + int(source == 'validation')
        if len(candidates) < needed:
            raise ValueError(f'Class {label}: insufficient disjoint {source} meta candidates.')
        parts.append(candidates.sample(n=per_class, random_state=seed + label))
    meta = pd.concat(parts)
    remaining = frames['validation'].copy()
    if source == 'validation':
        remaining = remaining.drop(meta.index).reset_index(drop=True)
    return {**frames, 'validation': remaining,
            'meta': meta.drop(columns='_key').reset_index(drop=True)}


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
    return {'counts': {n: len(f) for n, f in sets.items()},
            'class_counts': {n: {int(k): int(v) for k, v in f.label.value_counts().items()}
                             for n, f in sets.items()},
            'training_rows_E1': len(frames['seed']),
            'training_rows_E2_to_E7': len(frames['seed']) + len(synthetic),
            'cross_set_text_overlap': False}


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


def head_moments(model, optimizer):
    """Read the Adam state without mutation; absent state is zero."""
    while hasattr(optimizer, 'optimizer'):
        optimizer = optimizer.optimizer
    means, variances = [], []
    steps = []
    for p in (model.classifier.weight, model.classifier.bias):
        state = optimizer.state.get(p, {})
        group = next(g for g in optimizer.param_groups if any(q is p for q in g['params']))
        if tuple(group.get('betas', ())) != (0.9, 0.999):
            raise ValueError('LESS implementation expects Adam betas=(0.9, 0.999).')
        means.append(state.get('exp_avg', torch.zeros_like(p)).detach().float().cpu().flatten())
        variances.append(state.get('exp_avg_sq', torch.zeros_like(p)).detach().float().cpu().flatten())
        steps.append(float(state.get('step', 0)))
    return {'mean': torch.cat(means).clone(), 'variance': torch.cat(variances).clone(),
            'step': min(steps)}


def adam_directions(gradients, model, optimizer, snapshot=None):
    """Mirror LESS repo: m_new / sqrt(v_new + 1e-8), no bias correction.

    Snapshot moments are read only. Candidate gradients do NOT update them.
    A warmup snapshot can be supplied at the first epoch of a fresh branch.
    """
    moments = head_moments(model, optimizer) if snapshot is None else snapshot
    m, v = moments['mean'], moments['variance']
    direction = (0.9 * m + 0.1 * gradients) / torch.sqrt(
        0.999 * v + 0.001 * gradients.square() + 1e-8)
    return direction, moments['step']


class WarmupHeadMoments(TrainerCallback):
    """Capture moments at the same validation-best epoch as M0, not at stop."""
    def __init__(self):
        self.best, self.snapshot, self.epoch = -np.inf, None, None

    def on_evaluate(self, args, state, control, metrics=None, model=None,
                    optimizer=None, **kwargs):
        if metrics['eval_f1'] > self.best:
            self.best = metrics['eval_f1']
            self.snapshot = head_moments(model, optimizer)
            self.epoch = state.epoch
        return control


def less_scores(model, optimizer, tokenizer, synthetic, meta, batch_size, max_length,
                smoothing, aggregation, snapshot=None):
    sh, sl = head_features(model, tokenizer, synthetic, batch_size, max_length)
    mh, ml = head_features(model, tokenizer, meta, batch_size, max_length)
    sg = head_gradients(sh, sl, torch.tensor(synthetic.label.tolist()), smoothing)
    mg = head_gradients(mh, ml, torch.tensor(meta.label.tolist()), smoothing)
    direction, step = adam_directions(sg, model, optimizer, snapshot)
    direction = F.normalize(direction, dim=1)
    mg = F.normalize(mg, dim=1)
    per_class = torch.stack([mg[torch.tensor(meta.label.to_numpy() == c)].mean(0)
                             for c in range(3)])
    similarities = direction @ per_class.T
    score = similarities.mean(1) if aggregation == 'mean' else similarities.max(1).values
    return score.clamp(-1, 1).numpy(), step


def less_multiplier(score):
    score = np.asarray(score, dtype=float)
    return np.where(score >= 0, 1 + score, 1 + 0.5 * score)


def ren_head_gate(hidden, labels, meta_hidden, meta_labels, weight, bias,
                  inner_lr, smoothing=0.1):
    """Ren Algorithm 1 on a detached final-head surrogate; eps starts at 0.

    Returns detached, nonnegative gates with batch mean 1. If all gates are
    zero, return ones (retain RLN prior); this differs from Ren's zero update.
    """
    with torch.enable_grad():
        w = weight.detach().float().clone().requires_grad_(True)
        b = bias.detach().float().clone().requires_grad_(True)
        eps = torch.zeros(len(labels), device=w.device, requires_grad=True)
        losses = F.cross_entropy(F.linear(hidden.detach().float(), w, b), labels,
                                 reduction='none', label_smoothing=smoothing)
        inner = (eps * losses).mean()
        dw, db = torch.autograd.grad(inner, (w, b), create_graph=True)
        meta_loss = F.cross_entropy(F.linear(meta_hidden.detach().float(),
                                             w - inner_lr * dw, b - inner_lr * db),
                                    meta_labels, label_smoothing=smoothing)
        sensitivity = -torch.autograd.grad(meta_loss, eps)[0]
        positive = sensitivity.clamp_min(0).detach()
        fallback = bool(positive.sum() <= 1e-20)
        gate = torch.ones_like(positive) if fallback else positive / positive.mean()
    if not torch.isfinite(gate).all():
        raise ValueError('Non-finite Ren gate.')
    return gate, sensitivity.detach(), fallback


class IndexedWeights(MutableWeights):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row['_syn_index'] = int(index) - self.seed_count
        return row


class GradientWeighting(TrainerCallback):
    def __init__(self, data, synthetic, meta, tokenizer, scores, mode, folder,
                 dataset, batch_size=32, max_length=512, smoothing=0.1, aggregation='mean',
                 warmup_moments=None):
        self.data, self.synthetic, self.meta, self.tokenizer = data, synthetic, meta, tokenizer
        self.scores = {k: np.asarray(v).copy() for k, v in scores.items()}
        self.mode, self.folder, self.dataset = mode, Path(folder), dataset
        self.batch_size, self.max_length = batch_size, max_length
        self.smoothing, self.aggregation = smoothing, aggregation
        self.warmup_moments = warmup_moments
        self.epoch, self.previous = 0, None

    def on_epoch_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if args.world_size != 1 or args.dataloader_num_workers != 0:
            raise ValueError('Use one process and dataloader_num_workers=0.')
        epoch = int(round(state.epoch or 0)) + 1
        if epoch == self.epoch:
            return control
        if epoch > 1 and self.mode != 'static':
            self.scores['L'] = 1 - student_probabilities(model, self.tokenizer, self.synthetic,
                self.dataset, self.batch_size, self.max_length)
        q = self.scores['R'] * self.scores['L'] * self.scores['N']
        rln, _ = normalize_q(q, 'legacy')
        score = np.zeros(len(q))
        moment_step = 0.0
        if self.mode in ('less', 'combined'):
            if epoch == 1 and self.warmup_moments is None:
                raise ValueError('LESS first epoch requires moments from validation-best M0.')
            score, moment_step = less_scores(model, optimizer, self.tokenizer,
                self.synthetic, self.meta, self.batch_size, self.max_length,
                self.smoothing, self.aggregation,
                self.warmup_moments if epoch == 1 else None)
        weights = np.clip(rln * less_multiplier(score), 0.5, 2.0)
        self.data.update(weights)
        destination = self.folder / 'weights'
        destination.mkdir(exist_ok=True)
        if epoch == 1 or self.mode != 'static':
            pd.DataFrame({'synthetic_index': np.arange(len(q)), **self.scores,
                          'Q': q, 'rln_weight': rln, 'less_score': score,
                          'prior_weight': weights}).to_csv(
                destination / f'epoch_{epoch:03d}.csv', index=False)
        row = {'epoch': epoch, 'L_recomputed': epoch > 1 and self.mode != 'static',
               'less_recomputed': self.mode in ('less', 'combined'),
               'adam_moment_source': 'M0_best' if epoch == 1 and self.mode in
                     ('less', 'combined') else 'current_branch',
               'adam_moment_step': moment_step, 'weight_min': weights.min(),
               'weight_max': weights.max(), 'weight_mean': weights.mean(),
               'mean_abs_change': None if self.previous is None else
                    float(np.abs(weights - self.previous).mean())}
        path = self.folder / 'weight_history.csv'
        pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
        self.epoch, self.previous = epoch, weights.copy()
        print(f'[WEIGHTING] {self.dataset}/{self.mode}: {row}', flush=True)
        return control


def blend_ren_weights(prior, gate, strength=1.0):
    """Interpolate AFTER clipping: 0 preserves prior, 1 is the original Ren rule."""
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Ren strength must be between 0 and 1.')
    proposal = (prior * gate).clamp(0.5, 2.0)
    if strength == 0:
        return prior.detach().clone()
    if strength == 1:
        return proposal.detach()
    return ((1 - strength) * prior + strength * proposal).detach()


def install_ren(trainer, tokenizer, meta, folder, inner_lr, smoothing=0.1,
                batch_size=32, max_length=512, strength=1.0):
    """Replace training loss only; evaluation always uses ordinary weighted CE."""
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Ren strength must be between 0 and 1.')
    # Exact no-meta path, including the original forward/RNG behavior.
    if strength == 0:
        raise ValueError('For strength=0 the runner must skip install_ren entirely.')
    original_collator = trainer.data_collator

    def collate(features):
        copies = [dict(row) for row in features]
        indices = [row.pop('_syn_index', -1) for row in copies]
        result = original_collator(copies)
        result['_syn_index'] = torch.tensor(indices, dtype=torch.long)
        return result

    trainer.data_collator = collate
    trainer.label_names = ['labels']
    trainer.model_accepts_loss_kwargs = False
    log_path = Path(folder) / 'ren_batch_history.csv'
    examples_path = Path(folder) / 'ren_sample_weights.csv'

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop('labels')
        prior = inputs.pop('weight').float()
        indices = inputs.pop('_syn_index')
        use_ren = model.training and torch.is_grad_enabled() and bool((indices >= 0).any())
        capture = []
        hook = model.classifier.register_forward_pre_hook(
            lambda module, args: capture.append(args[0])) if use_ren else None
        try:
            outputs = model(**inputs)
        finally:
            if hook is not None:
                hook.remove()
        weights = prior.detach().clone()
        if use_ren:
            mask = indices >= 0
            mh, _ = head_features(model, tokenizer, meta, batch_size, max_length)
            mh = mh.to(labels.device)
            my = torch.tensor(meta.label.tolist(), device=labels.device)
            gate, sensitivity, fallback = ren_head_gate(capture[0][mask], labels[mask],
                mh, my, model.classifier.weight, model.classifier.bias, inner_lr, smoothing)
            proposal = (prior[mask] * gate).clamp(0.5, 2.0)
            weights[mask] = blend_ren_weights(prior[mask], gate, strength)
            weights[~mask] = 1.0
            row = {'step': self.state.global_step, 'epoch': self.state.epoch,
                   'ren_strength': strength,
                   'synthetic_count': int(mask.sum()), 'all_zero_fallback': fallback,
                   'positive_gate_fraction': float((sensitivity > 0).float().mean()),
                   'prior_mean': float(prior[mask].mean()),
                   'final_mean': float(weights[mask].mean()),
                   'final_min': float(weights[mask].min()), 'final_max': float(weights[mask].max()),
                   'proposal_floor_fraction': float((proposal <= 0.5000001).float().mean()),
                   'final_floor_fraction': float((weights[mask] <= 0.5000001).float().mean())}
            with log_path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            with examples_path.open('a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['step', 'synthetic_index', 'prior', 'sensitivity', 'gate', 'weight'])
                writer.writerows(zip([self.state.global_step] * int(mask.sum()),
                    indices[mask].detach().cpu().tolist(), prior[mask].detach().cpu().tolist(),
                    sensitivity.cpu().tolist(), gate.cpu().tolist(), weights[mask].cpu().tolist()))
        losses = F.cross_entropy(outputs.logits.float(), labels, reduction='none',
                                 label_smoothing=smoothing)
        loss = (weights * losses).sum() / weights.sum().clamp_min(1e-12)
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = MethodType(compute_loss, trainer)
