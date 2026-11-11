"""E1--E7: RLN, meta-reweighting and Adam-aware gradient matching.

E1 seed; E2 uniform; E3 static RLN; E4 dynamic RLN; E5 RLN+Ren;
E6 RLN+LESS-inspired matching; E7 RLN+matching+Ren.
Every branch starts independently from M0. Final-head-only scoring/meta-step;
existing trainable student parameters still learn through the real weighted loss.
"""
import os

import argparse
import copy
import json
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from data_utils import prepare_data, prepare_synthetic, write_json, frame_sha256, file_sha256
from train_seed import build_seed_trainer
from training import build_stage_trainer
from scoring import assigned_label_probabilities, release_gpu
from dynamic_weights import student_probabilities, MutableWeights, EpochWeights
from run_ablation_ram import fit_in_ram, test_and_save
from gradient_weighting import (reserve_meta, check_data_contract, IndexedWeights, GradientWeighting, install_ren,
                           WarmupHeadMoments)


EXPERIMENTS = {'E1': 'seed', 'E2': 'uniform', 'E3': 'static', 'E4': 'dynamic',
               'E5': 'ren', 'E6': 'less', 'E7': 'combined',
               'E4_META': 'dynamic_direct_meta'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=['phrasebank', 'twitter', 'finsent'],
                   default=['phrasebank', 'twitter'])
    p.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS),
                   default=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7'])
    p.add_argument('--train-seeds', nargs='+', type=int, default=[42])
    p.add_argument('--split-seed', type=int, default=24266)
    p.add_argument('--meta-seed', type=int, default=24266)
    p.add_argument('--meta-per-class', type=int, default=15)
    p.add_argument('--meta-source', choices=['train_pool', 'validation'], default='train_pool',
                   help='Unused train-pool labels by default: preserve original validation.')
    p.add_argument('--reference-root', type=Path,
                   help='Optional old RLN output root; require matching baseline input hashes.')
    p.add_argument('--num-epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--ren-inner-lr', type=float, default=1e-4)
    p.add_argument('--ren-strength', type=float, default=1.0,
                   help='Ren interpolation: 0=prior only, 1=original E5/E7, e.g. 0.25 or 0.5.')
    p.add_argument('--ren-strength-e7', type=float, default=None,
                   help='Optional E7 override, so E5 and E7 can run together at different strengths.')
    p.add_argument('--less-aggregation', choices=['mean', 'max'], default='mean')
    p.add_argument('--score-batch-size', type=int, default=32)
    p.add_argument('--output-root', type=Path, default=Path('output/weighting_e1_e7_v2'))
    p.add_argument('--phrasebank-file', type=Path, default=Path(
        'data/financial_phrasebank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'))
    p.add_argument('--twitter-train-file', type=Path)
    p.add_argument('--twitter-validation-file', type=Path)
    p.add_argument('--finsent-data-dir', type=Path, default=Path('data/finsent/prepared'))
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--validation-only', action='store_true',
                   help='Save validation results only; never call test evaluation.')
    p.add_argument('--strict-determinism', action='store_true',
                   help='Opt in to stricter CUDA kernels; default follows the original runner.')
    p.add_argument("--freeze-layers", type=int, choices=range(7), default=4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    args = p.parse_args()
    if not np.isfinite(args.ren_strength) or not 0 <= args.ren_strength <= 1:
        p.error('--ren-strength must be finite and between 0 and 1.')
    if args.ren_strength_e7 is not None and (
            not np.isfinite(args.ren_strength_e7) or not 0 <= args.ren_strength_e7 <= 1):
        p.error('--ren-strength-e7 must be finite and between 0 and 1.')
    if min(args.meta_per_class, args.num_epochs, args.patience, args.score_batch_size) < 1:
        p.error('Counts, epochs, patience and batch size must be positive.')
    if any(not np.isfinite(x) or x <= 0 for x in [args.learning_rate, args.ren_inner_lr]):
        p.error('Learning rates must be finite and positive.')
    if min(args.train_seeds + [args.split_seed, args.meta_seed]) < 0:
        p.error('Seeds must be nonnegative.')
    if bool(args.twitter_train_file) != bool(args.twitter_validation_file):
        p.error('Supply both local Twitter files.')
    for name in ['datasets', 'train_seeds', 'experiments']:
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    for ds in args.datasets:
        for seed in args.train_seeds:
            if (args.output_root / ds / f'train_seed_{seed}').exists():
                p.error('Run folder exists; use a fresh output-root (no resume without models).')
    return args


def finish(trainer, tokenizer, frames, config, stage, dataset, weighted, selection, test=True):
    validation = trainer.evaluate()
    write_json(stage / 'validation_metrics.json', validation)
    result = {'validation_accuracy': float(validation['eval_accuracy']),
              'validation_macro_f1': float(validation['eval_f1']), **selection}
    if test:
        result.update(test_and_save(trainer, tokenizer, frames, config, stage,
                                    dataset, weighted, selection))
    else:
        write_json(stage / 'training_history.json', trainer.state.log_history)
        write_json(stage / 'selection.json', selection)
    return result


def record(csv_path, dataset, seed, experiment, metrics):
    row = pd.DataFrame([{'dataset': dataset, 'experiment': experiment,
                         'method': EXPERIMENTS[experiment], 'train_seed': seed, **metrics}])
    if csv_path.exists():
        row = pd.concat([pd.read_csv(csv_path), row], ignore_index=True)
    temporary = csv_path.with_suffix('.tmp')
    row.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    print(f'[RESULT] {dataset}/{seed}/{experiment}: {metrics}', flush=True)


def environment():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                    stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'python': platform.python_version(), 'git_commit': commit,
            'versions': {name: importlib.metadata.version(name) for name in
                         ['torch', 'transformers', 'datasets', 'accelerate', 'numpy', 'pandas']},
            'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}


def run_one(args, dataset, seed):
    set_seed(seed)
    folder = args.output_root / dataset / f'train_seed_{seed}'
    folder.mkdir(parents=True)
    config = dict(dataset=dataset, train_seed=seed, split_seed=args.split_seed,
                  learning_rate=args.learning_rate, num_epochs=args.num_epochs,
                  model='distilbert-base-uncased', model_revision='main', freeze_layers=args.freeze_layers,
                  max_length=512, train_batch_size=16, eval_batch_size=32,
                  weight_decay=args.weight_decay, label_smoothing=0.1,
                  early_stopping_patience=args.patience)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(training=config, environment=environment(),
                    source_sha256={p.name: file_sha256(p) for p in
                                   sorted(Path(__file__).parent.glob('*.py'))})
    write_json(folder / 'run_config.json', settings)
    extra_data_args = {'finsent_data_dir': args.finsent_data_dir} if dataset == 'finsent' else {}
    frames, audit = prepare_data(dataset=dataset,
        seed_path=Path('outputs') / f'seed_{dataset}_data_clustered.jsonl',
        split_seed=args.split_seed, phrasebank_file=args.phrasebank_file,
        twitter_train_file=args.twitter_train_file,
        twitter_validation_file=args.twitter_validation_file, expected_seeds=105, **extra_data_args)
    syn, removed = prepare_synthetic(Path('outputs') /
                    f'synthetic_{dataset}_data_seed_clustered.jsonl', dataset, frames)
    if dataset == 'finsent':
        if len(syn) != 945 or removed or not syn.label.value_counts().eq(315).all():
            raise ValueError('FinSent requires 945 unique synthetic texts, 315/class. Inspect generation/excluded rows.')
    old_count = len(frames['validation'])
    original_hashes = {name: frame_sha256(frames[name]) for name in ['seed', 'validation', 'test']}
    frames = reserve_meta(frames, syn, args.meta_per_class, args.meta_seed, args.meta_source)
    contract = check_data_contract(frames, syn)
    contract['training_rows_E4_META'] = len(frames['seed']) + len(frames['meta']) + len(syn)
    current_hashes = {**{name: frame_sha256(frames[name]) for name in
                         ['seed', 'meta', 'validation', 'test']}, 'synthetic': frame_sha256(syn)}
    contract['unchanged_from_original_split'] = {
        n: current_hashes[n] == original_hashes[n] for n in original_hashes}
    if args.meta_source == 'train_pool' and not all(contract['unchanged_from_original_split'].values()):
        raise AssertionError('Default protocol must keep original seed/validation/test frames.')
    if args.reference_root is not None:
        reference_path = args.reference_root / dataset / f'train_seed_{seed}' / 'input_sha256.json'
        reference = json.loads(reference_path.read_text())
        mismatches = [n for n in ['seed', 'validation', 'test', 'synthetic']
                      if reference.get(n) != current_hashes[n]]
        if mismatches:
            raise ValueError(f'Baseline inputs differ from reference: {mismatches}')
        contract['reference_hashes_match'] = True
    write_json(folder / 'data_contract.json', contract)
    snapshot = folder / 'data_snapshot'
    snapshot.mkdir()
    for name in ['seed', 'meta', 'validation', 'test']:
        frames[name].to_json(snapshot / f'{name}.jsonl', orient='records', lines=True,
                             force_ascii=False)
    syn.to_csv(folder / 'synthetic_samples.csv', index=False)
    write_json(folder / 'data_audit_original_split.json', audit)
    write_json(folder / 'excluded_synthetic.json', removed)
    write_json(folder / 'input_sha256.json', current_hashes)
    write_json(folder / 'meta_protocol.json', {
        'original_validation_count': old_count, 'meta_count': len(frames['meta']),
        'selection_validation_count': len(frames['validation']),
        'source': args.meta_source,
        'meta_direct_training': 'E4_META only; unit weight; M0 still seed-only',
        'head_surrogate_only': True,
        'ren_strength': args.ren_strength,
        'ren_strength_E7': args.ren_strength if args.ren_strength_e7 is None else args.ren_strength_e7,
        'ren_weight_rule': '(1-strength)*prior + strength*clip(prior*gate,0.5,2)',
        'less_aggregation': args.less_aggregation,
        'less_candidate_direction': 'LESS repository m_new/sqrt(v_new+1e-8)',
        'less_moments': 'M0-best at epoch 1; current independent branch at later epochs',
        'less_reference': 'normalized ordinary meta gradients',
        'ren': 'eps=0 virtual SGD on synthetic batch, rectified detached meta-gradient',
        'adaptations': 'final classifier only; epoch-wise LESS; RLN prior; bounded final weights',
        'meta_label_budget_warning': f'E5/E6/E7 access {len(frames["meta"])} extra clean labels.'})
    if args.prepare_only:
        print(f'[DATA OK] {dataset}/{seed}: seed={len(frames["seed"])}, '
              f'syn={len(syn)}, meta={len(frames["meta"])}, val={len(frames["validation"])}')
        return
    e1 = folder / 'E1'
    e1.mkdir()
    trainer, tokenizer, _, _, _ = build_seed_trainer(SimpleNamespace(**config), e1, frames)
    warmup = WarmupHeadMoments()
    trainer.add_callback(warmup)
    selection = fit_in_ram(trainer, args.patience)
    trainer.remove_callback(WarmupHeadMoments)
    if warmup.snapshot is None or warmup.epoch != selection['best_epoch']:
        raise RuntimeError('Warmup moments are not aligned with the selected M0.')
    warmup_moments = warmup.snapshot
    write_json(folder / 'warmup_moment_info.json', {'source_epoch': warmup.epoch,
               'optimizer_step': warmup_moments['step'], 'scope': 'classifier only'})
    del warmup
    write_json(folder / 'model_identity.json', {'name': config['model'],
               'resolved_commit': getattr(trainer.model.config, '_commit_hash', None)})
    csv_path = args.output_root / 'comparison.csv'
    record(csv_path, dataset, seed, 'E1', finish(trainer, tokenizer, frames, config,
            e1, dataset, False, selection, not args.validation_only))
    need_scores = any(x not in ('E1', 'E2') for x in args.experiments)
    l0 = 1 - student_probabilities(trainer.model, tokenizer, syn, dataset,
                args.score_batch_size, 512) if need_scores else None
    m0 = trainer.model.cpu()
    del trainer
    release_gpu()
    scores = None
    for experiment in args.experiments:
        if experiment == 'E1':
            continue
        if experiment != 'E2' and scores is None:
            # Match original ordering: E2 first, then teacher/novelty scoring,
            # then reset training RNG before building each weighted branch.
            r = assigned_label_probabilities('ProsusAI/finbert', syn, dataset, args.score_batch_size, 512)
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            a = encoder.encode(frames['seed'].sentence.tolist(), normalize_embeddings=True,
                                batch_size=args.score_batch_size)
            b = encoder.encode(syn.sentence.tolist(), normalize_embeddings=True,
                                batch_size=args.score_batch_size)
            n = 1 - np.clip(b @ a.T, -1, 1).max(axis=1)
            del encoder
            release_gpu()
            scores = {'R': r, 'L': l0, 'N': n}
            syn.assign(**scores).to_csv(folder / 'initial_scores.csv', index=False)
        stage = folder / experiment
        stage.mkdir()
        set_seed(seed)
        model = copy.deepcopy(m0)
        ren_strength = args.ren_strength_e7 if experiment == 'E7' and args.ren_strength_e7 is not None else args.ren_strength
        clean = frames['seed'][['sentence', 'label']].copy()
        if experiment == 'E4_META':
            clean = pd.concat([clean, frames['meta'][['sentence', 'label']]], ignore_index=True)
        train = pd.concat([clean.assign(weight=1.0),
                           syn[['sentence', 'label']].assign(weight=1.0)], ignore_index=True)
        write_json(stage / 'training_contract.json', {
            'clean_count': len(clean), 'synthetic_count': len(syn),
            'total_count': len(train), 'clean_weight': 1.0,
            'meta_direct_training': experiment == 'E4_META',
            'ren_strength': ren_strength if experiment in ('E5', 'E7') else None,
            'initialization': 'same seed-only M0; fresh optimizer'})
        train.to_json(stage / 'train_snapshot.jsonl', orient='records', lines=True, force_ascii=False)
        stage_args = SimpleNamespace(**{**vars(args), 'train_seed': seed})
        trainer = build_stage_trainer(stage_args, stage, config, frames, train, model, tokenizer)
        updater = None
        mode = EXPERIMENTS[experiment]
        if experiment != 'E2':
            data = IndexedWeights(trainer.train_dataset, len(clean), len(syn)) \
                   if mode in ('ren', 'combined') and ren_strength > 0 else None
            if data is None:
                data = MutableWeights(trainer.train_dataset, len(clean), len(syn))
            trainer.train_dataset = data
            if experiment in ('E3', 'E4', 'E4_META'):
                updater = EpochWeights(data, syn, scores, tokenizer, dataset, 'RLN',
                    'legacy' if experiment == 'E3' else 'dynamic_legacy', stage,
                    args.score_batch_size, 512)
            else:
                updater = GradientWeighting(data, syn, frames['meta'], tokenizer, scores, mode,
                    stage, dataset, args.score_batch_size, 512, 0.1, args.less_aggregation,
                    warmup_moments)
            trainer.add_callback(updater)
            if mode in ('ren', 'combined') and ren_strength > 0:
                install_ren(trainer, tokenizer, frames['meta'], stage, args.ren_inner_lr,
                            batch_size=args.score_batch_size, strength=ren_strength)
        print(f'[TRAIN] {dataset}/{seed}/{experiment}: {mode}, independent M0', flush=True)
        selection = fit_in_ram(trainer, args.patience)
        if updater is not None:
            trainer.remove_callback(type(updater))
        record(csv_path, dataset, seed, experiment, finish(trainer, tokenizer, frames,
               config, stage, dataset, True, selection, not args.validation_only))
        del trainer, model, updater
        release_gpu()
    del m0, tokenizer
    release_gpu()


def main():
    args = parse_args()
    if args.strict_determinism:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not args.prepare_only and torch.cuda.device_count() > 1:
        raise ValueError('Expose one GPU: env CUDA_VISIBLE_DEVICES=0 python ...')
    if args.strict_determinism:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.output_root.mkdir(parents=True, exist_ok=True)
    for seed in args.train_seeds:
        for dataset in args.datasets:
            run_one(args, dataset, seed)
    print(f'Finished: {args.output_root}', flush=True)


if __name__ == '__main__':
    main()
