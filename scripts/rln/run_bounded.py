"""Run independent E1-E8 branches from the same seed-only M0 (default: E5-E8)."""
import argparse
import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['phrasebank', 'twitter', 'finsent'],
                        default=['phrasebank', 'twitter', 'finsent'])
    parser.add_argument('--experiments', nargs='+', choices=[f'E{i}' for i in range(1, 9)],
                        default=['E5', 'E6', 'E7', 'E8'])
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=24266)
    parser.add_argument('--prepare-meta', action='store_true', help='Freeze the reference set and exit.')
    parser.add_argument('--meta-dir', type=Path, default=Path('data/rln_meta'))
    parser.add_argument('--meta-seed', type=int, default=24266)
    parser.add_argument('--meta-per-class', type=int, default=15)
    parser.add_argument('--meta-method', choices=['clustered', 'random'], default='clustered')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--score-batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--freeze-layers', type=int, choices=range(7), default=4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--phrasebank-file', type=Path, default=Path(
        'data/financial_phrasebank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'))
    parser.add_argument('--twitter-train-file', type=Path)
    parser.add_argument('--twitter-validation-file', type=Path)
    parser.add_argument('--finsent-data-dir', type=Path, default=Path('data/finsent/prepared'))
    parser.add_argument('--outputs-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--output-root', type=Path, default=Path('output/rln_results'))
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    if bool(args.twitter_train_file) != bool(args.twitter_validation_file):
        parser.error('Provide both local Twitter files.')
    if min(args.num_epochs, args.patience, args.score_batch_size, args.meta_per_class) < 1:
        parser.error('Counts must be positive.')
    if min(args.train_seed, args.split_seed, args.meta_seed) < 0:
        parser.error('Seeds must be nonnegative.')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Invalid learning rate.')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('Invalid weight decay.')
    args.datasets = list(dict.fromkeys(args.datasets))
    args.experiments = list(dict.fromkeys(args.experiments))
    return args


def load_inputs(args, dataset):
    from data_utils import prepare_data, prepare_synthetic
    from run_io import frame_sha256
    frames, audit = prepare_data(dataset=dataset,
        seed_path=args.outputs_dir / f'seed_{dataset}_data_clustered.jsonl',
        split_seed=args.split_seed, phrasebank_file=args.phrasebank_file,
        twitter_train_file=args.twitter_train_file, twitter_validation_file=args.twitter_validation_file,
        finsent_data_dir=args.finsent_data_dir, expected_seeds=105)
    hashes = {name: frame_sha256(frames[name]) for name in ('seed', 'validation', 'test')}
    syn = None
    removed = []
    if args.prepare_meta or set(args.experiments) != {'E1'}:
        syn, removed = prepare_synthetic(
            args.outputs_dir / f'synthetic_{dataset}_data_seed_clustered.jsonl', dataset, frames)
        syn = syn.reset_index(drop=True)
        if dataset == 'finsent' and (len(syn) != 945 or removed or not syn.label.value_counts().eq(315).all()):
            raise ValueError('Expected 945 unique FinSent synthetic rows, 315 per class.')
        hashes['synthetic'] = frame_sha256(syn)
    return frames, audit, syn, removed, hashes


def run_dataset(args, dataset, folder):
    import pandas as pd
    import torch
    from transformers import set_seed
    from bounded_m0 import train_m0
    from bounded_weights import BoundedWeights, METHODS, USE_GRADIENT, initial_scores
    from run_io import frame_sha256, read_json, write_json
    from bounded_weights import MutableWeights
    from reference import frozen_meta
    from evaluation import fit_in_ram, finish
    from scoring import release_gpu
    from training import build_stage_trainer

    set_seed(args.train_seed)
    config = dict(dataset=dataset, train_seed=args.train_seed, split_seed=args.split_seed,
                  learning_rate=args.learning_rate, num_epochs=args.num_epochs,
                  early_stopping_patience=args.patience, model='distilbert-base-uncased',
                  model_revision='main', freeze_layers=args.freeze_layers, max_length=512,
                  train_batch_size=16, eval_batch_size=32, weight_decay=args.weight_decay,
                  label_smoothing=0.1)
    frames, audit, syn, removed, hashes = load_inputs(args, dataset)
    settings = dict(training=config, input_sha256=hashes, validation_only=args.validation_only,
                    meta_method=args.meta_method, meta_seed=args.meta_seed,
                    meta_per_class=args.meta_per_class, score_batch_size=args.score_batch_size,
                    gradient_strength=0.5, weight_bounds=[0.5, 2])
    manifest = folder / 'run_config.json'
    if manifest.exists():
        previous = read_json(manifest)
        # E1 alone needs no synthetic file. Bind its hash when a later branch first uses it.
        before = previous['input_sha256']
        if 'synthetic' in before and syn is None:
            hashes['synthetic'] = before['synthetic']
        elif 'synthetic' not in before and syn is not None:
            before['synthetic'] = hashes['synthetic']
        if previous != settings:
            raise ValueError(f'Configuration or data changed. Use a different --output-root: {folder}')
    write_json(manifest, settings)
    pending = [e for e in args.experiments if not (folder / e / 'result.json').is_file()]
    if not pending:
        print(f'[SKIP] {dataset}: all requested experiments completed.', flush=True)
        return
    write_json(folder / 'data_audit.json', audit)
    if syn is not None:
        write_json(folder / 'excluded_synthetic.json', removed)
        syn.to_csv(folder / 'synthetic_samples.csv', index_label='synthetic_index')
    if USE_GRADIENT.intersection(pending):
        frames['meta'] = frozen_meta(args, dataset, frames, syn, hashes)
        meta_hash = frame_sha256(frames['meta'])
        ref = folder / 'meta_sha256.json'
        if ref.exists() and read_json(ref) != meta_hash:
            raise ValueError('Reference rows changed across branches.')
        write_json(ref, meta_hash)
        frames['meta'].to_csv(folder / 'meta_samples.csv', index=False)

    m0, tokenizer, m0_selection, identity = train_m0(config, frames, folder)
    m0_info = folder / 'm0.json'
    if m0_info.exists() and read_json(m0_info)['fingerprint'] != identity:
        raise ValueError('This output root contains branches initialized from a different M0.')
    write_json(m0_info, {**m0_selection, 'fingerprint': identity})
    m0.to('cuda' if torch.cuda.is_available() else 'cpu')
    scores = initial_scores(pending, m0, tokenizer, frames, syn, args.score_batch_size, config)
    m0.cpu()
    release_gpu()
    seed_rows = frames['seed'].assign(weight=1.0)
    train = seed_rows if syn is None else pd.concat([seed_rows, syn.assign(weight=1.0)], ignore_index=True)

    for experiment in pending:
        print(f'[{dataset}] {experiment}: {METHODS[experiment]}', flush=True)
        set_seed(args.train_seed)
        with TemporaryDirectory(dir=folder) as temp:
            stage = Path(temp) / experiment
            stage.mkdir()
            rows = seed_rows if experiment == 'E1' else train
            trainer = build_stage_trainer(args, stage, config, frames, rows, copy.deepcopy(m0), tokenizer)
            if experiment not in ('E1', 'E2'):
                data = MutableWeights(trainer.train_dataset, len(frames['seed']), len(syn))
                trainer.train_dataset = data
                trainer.add_callback(BoundedWeights(experiment, data, scores, tokenizer, syn,
                    frames.get('meta'), config, args.score_batch_size, stage))
            if experiment == 'E1':
                # M0 already is E1's validation-best model; evaluate without a second fit.
                selection = m0_selection
            else:
                selection = fit_in_ram(trainer, args.patience)
            metrics = finish(trainer, tokenizer, frames, config, stage, dataset, True, selection,
                             test=not args.validation_only)
            write_json(stage / 'result.json', dict(dataset=dataset, train_seed=args.train_seed,
                       experiment=experiment, method=METHODS[experiment], **metrics))
            stage.rename(folder / experiment)
            del trainer
        release_gpu()
        summarize(args.output_root)


def summarize(root):
    import pandas as pd
    from run_io import locked, read_json
    with locked(root / '.comparison.lock'):
        rows = [read_json(p) for p in sorted(root.glob('*/train_seed_*/E*/result.json'))]
        if rows:
            temp = root / 'comparison.tmp'
            pd.DataFrame(rows).to_csv(temp, index=False)
            temp.replace(root / 'comparison.csv')


def main():
    args = parse_args()
    if args.prepare_meta:
        from reference import frozen_meta
        for dataset in args.datasets:
            frames, _, syn, _, hashes = load_inputs(args, dataset)
            frozen_meta(args, dataset, frames, syn, hashes)
        return
    import torch
    from run_io import locked
    if torch.cuda.device_count() > 1:
        raise ValueError('Expose one GPU with CUDA_VISIBLE_DEVICES=0.')
    for dataset in args.datasets:
        folder = args.output_root / dataset / f'train_seed_{args.train_seed}'
        with locked(folder / '.run.lock'):
            run_dataset(args, dataset, folder)
    summarize(args.output_root)


if __name__ == '__main__':
    main()
