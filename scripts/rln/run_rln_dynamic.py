"""Fresh E1 -> independent E2 / legacy / static01 / dynamic01; results only.

Run from GR2 root. Requires the user's seven existing scripts, unchanged.
Uses the same best-in-RAM selection and weighted loss as run_ablation_ram.py.
An interruption cannot resume trained weights; use a new output root to restart.
"""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import set_seed

from data_utils import (prepare_data, prepare_synthetic, write_json, frame_sha256,
                        file_sha256)
from train_seed import build_seed_trainer
from training import build_stage_trainer
from scoring import assigned_label_probabilities, release_gpu
from run_ablation_ram import fit_in_ram, test_and_save
from dynamic_weights import student_probabilities, MutableWeights, EpochWeights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['phrasebank', 'twitter', 'finsent'],
                        default=['phrasebank', 'twitter'])
    parser.add_argument('--variants', nargs='+', choices=['R', 'L', 'N', 'RL', 'RN', 'LN', 'RLN'],
                        default=['RLN'])
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['legacy', 'static', 'dynamic', 'dynamic_legacy'],
        default=['legacy', 'static', 'dynamic'],
    )
    parser.add_argument('--skip-e2', action='store_true')
    parser.add_argument('--output-root', type=Path, default=Path('output/rln_dynamic_01'))
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=24266)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--score-batch-size', type=int, default=32)
    parser.add_argument('--twitter-train-file', type=Path)
    parser.add_argument('--twitter-validation-file', type=Path)
    parser.add_argument('--phrasebank-file', type=Path, default=Path(
        'data/financial_phrasebank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'))
    args = parser.parse_args()
    if (not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            min(args.num_epochs, args.patience, args.score_batch_size) < 1):
        parser.error('Learning rate, epochs, patience and batch size must be positive.')
    if bool(args.twitter_train_file) != bool(args.twitter_validation_file):
        parser.error('Provide both local Twitter files.')
    for key in ['datasets', 'variants', 'modes']:
        setattr(args, key, list(dict.fromkeys(getattr(args, key))))
    for dataset in args.datasets:
        folder = args.output_root / dataset / f'train_seed_{args.train_seed}'
        if folder.exists():
            parser.error(f'{folder} already exists. Use another output root or train seed.')
    return args


def finish(trainer, tokenizer, frames, config, stage, dataset, weighted, selection):
    validation = trainer.evaluate()
    write_json(stage / 'validation_metrics.json', validation)
    metrics = test_and_save(trainer, tokenizer, frames, config, stage, dataset, weighted, selection)
    return {**metrics, 'validation_accuracy': float(validation['eval_accuracy']),
            'validation_macro_f1': float(validation['eval_f1']), **selection}


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / 'comparison.csv'
    for dataset in args.datasets:
        folder = args.output_root / dataset / f'train_seed_{args.train_seed}'
        seed_folder = folder / 'E1'
        seed_folder.mkdir(parents=True)
        config = dict(dataset=dataset, train_seed=args.train_seed, split_seed=args.split_seed,
                      learning_rate=args.learning_rate, num_epochs=args.num_epochs,
                      model='distilbert-base-uncased', model_revision='main', freeze_layers=4,
                      max_length=512, train_batch_size=16, eval_batch_size=32,
                      weight_decay=0.1, label_smoothing=0.1,
                      early_stopping_patience=args.patience)
        settings = {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}
        write_json(folder / 'run_config.json', {**settings, 'training': config,
                   'normalization_01': '(Q-minQ)/(maxQ-minQ), entire synthetic corpus',
                   'normalization_legacy': 'clip(Q/meanQ, 0.5, 2), entire synthetic corpus',
                   'dynamic_L_modes': ['dynamic', 'dynamic_legacy'],
                   'equal_q_policy': 'all weights 1 when range <= 1e-12',
                   'L_refresh': 'before each training epoch; first uses M0',
                   'seed_weight': 1, 'model_saved': False,
                   'source_sha256': {path.name: file_sha256(path)
                                     for path in sorted(Path(__file__).parent.glob('*.py'))}})
        frames, audit = prepare_data(
            dataset=dataset, seed_path=Path('outputs') / f'seed_{dataset}_data_clustered.jsonl',
            split_seed=args.split_seed, phrasebank_file=args.phrasebank_file,
            twitter_train_file=args.twitter_train_file,
            twitter_validation_file=args.twitter_validation_file, expected_seeds=105)
        synthetic, removed = prepare_synthetic(
            Path('outputs') / f'synthetic_{dataset}_data_seed_clustered.jsonl', dataset, frames)
        write_json(folder / 'data_audit.json', audit)
        write_json(folder / 'excluded_synthetic.json', removed)
        write_json(folder / 'input_sha256.json', {
            **{name: frame_sha256(frames[name]) for name in ['seed', 'validation', 'test']},
            'synthetic': frame_sha256(synthetic)})
        synthetic.to_csv(folder / 'synthetic_samples.csv', index=False)

        def record(experiment, metrics):
            row = pd.DataFrame([{'dataset': dataset, 'experiment': experiment,
                                 'train_seed': args.train_seed, **metrics}])
            if csv_path.is_file():
                row = pd.concat([pd.read_csv(csv_path), row], ignore_index=True)
            temporary = csv_path.with_suffix('.tmp')
            row.to_csv(temporary, index=False)
            temporary.replace(csv_path)
            print(f'[RESULT] {dataset}/{experiment}: {metrics}', flush=True)

        print(f'[{dataset}] Train fresh E1/M0', flush=True)
        trainer, tokenizer, _, _, _ = build_seed_trainer(SimpleNamespace(**config), seed_folder, frames)
        selection = fit_in_ram(trainer, config['early_stopping_patience'])
        record('E1', finish(trainer, tokenizer, frames, config, seed_folder, dataset, False, selection))
        # Score M0 on its existing device; only then retain it on CPU for all branches.
        L0 = 1.0 - student_probabilities(trainer.model, tokenizer, synthetic, dataset,
                                       args.score_batch_size, config['max_length'])
        m0 = trainer.model.cpu()
        del trainer
        release_gpu()

        def run_stage(experiment, variant=None, mode=None, scores=None):
            stage = folder / experiment
            stage.mkdir()
            train = pd.concat([frames['seed'].assign(weight=1.0),
                               synthetic[['sentence', 'label']].assign(weight=1.0)], ignore_index=True)
            set_seed(args.train_seed)
            model = copy.deepcopy(m0)
            trainer = build_stage_trainer(args, stage, config, frames, train, model, tokenizer)
            updater = None
            if variant is not None:
                data = MutableWeights(trainer.train_dataset, len(frames['seed']), len(synthetic))
                trainer.train_dataset = data
                updater = EpochWeights(data, synthetic, scores, tokenizer, dataset, variant,
                                       mode, stage, args.score_batch_size, config['max_length'])
                trainer.add_callback(updater)
            print(f'[{dataset}] {experiment}: independently initialize from M0', flush=True)
            selection = fit_in_ram(trainer, config['early_stopping_patience'])
            if updater is not None:
                trainer.remove_callback(EpochWeights)
            record(experiment, finish(trainer, tokenizer, frames, config, stage, dataset, True, selection))
            del trainer, model, updater
            release_gpu()

        if not args.skip_e2:
            run_stage('E2')
        R = assigned_label_probabilities('ProsusAI/finbert', synthetic, dataset,
                                         args.score_batch_size, 512)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        seed_embeddings = encoder.encode(frames['seed'].sentence.tolist(),
                                         normalize_embeddings=True, batch_size=args.score_batch_size)
        embeddings = encoder.encode(synthetic.sentence.tolist(), normalize_embeddings=True,
                                    batch_size=args.score_batch_size)
        N = 1.0 - np.clip(embeddings @ seed_embeddings.T, -1.0, 1.0).max(axis=1)
        del encoder
        release_gpu()
        scores = {'R': R, 'L': L0, 'N': N}
        synthetic.assign(**scores).to_csv(folder / 'initial_scores.csv', index=False)
        for variant in args.variants:
            for mode in args.modes:
                if mode in ('dynamic', 'dynamic_legacy') and 'L' not in variant:
                    fixed = 'legacy' if mode == 'dynamic_legacy' else 'static'
                    print(
                        f'[SKIP] {variant} has no L; {mode} duplicates {fixed}.',
                        flush=True,
                    )
                    continue
                suffix = {
                    'legacy': 'legacy',
                    'static': 'static01',
                    'dynamic': 'dynamic01',
                    'dynamic_legacy': 'dynamic052',
                }[mode]
                run_stage(f'E3_{variant}_{suffix}', variant, mode, scores)
        del m0, tokenizer, run_stage
        release_gpu()
    print(f'Saved comparison: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
