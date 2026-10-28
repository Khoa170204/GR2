"""Train fresh E1 and independent E3 ablations. Keep all trained weights in RAM."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import TrainerCallback, EarlyStoppingCallback, set_seed

from data_utils import LABELS, prepare_data, prepare_synthetic, write_json, normalize_label
from train_seed import build_seed_trainer
from training import build_stage_trainer
from scoring import assigned_label_probabilities, make_weights, release_gpu


class BestInRAM(TrainerCallback):
    def __init__(self, patience):
        self.patience = patience
        self.best = -np.inf
        self.weights = None
        self.bad = 0
        self.epoch = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        value = float(metrics['eval_f1'])
        if not np.isfinite(value):
            raise ValueError('Non-finite validation macro-F1')
        if value > self.best:
            self.best, self.bad, self.epoch = value, 0, state.epoch
            self.weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
            if self.bad >= self.patience:
                control.should_training_stop = True
        return control


def fit_in_ram(trainer, patience):
    trainer.args.save_strategy = type(trainer.args.save_strategy)('no')
    trainer.args.load_best_model_at_end = False
    trainer.remove_callback(EarlyStoppingCallback)
    best = BestInRAM(patience)
    trainer.add_callback(best)
    trainer.train()
    if best.weights is None:
        raise RuntimeError('No validation-best weights captured')
    trainer.model.load_state_dict(best.weights)
    trainer.remove_callback(BestInRAM)
    best.weights = None
    return {'best_epoch': best.epoch, 'best_validation_macro_f1': best.best,
            'model_saved': False}


def test_and_save(trainer, tokenizer, frames, config, folder, dataset, weighted, selection):
    test = frames['test'][['sentence', 'label']].copy()
    if weighted:
        test['weight'] = 1.0
    data = Dataset.from_pandas(test, preserve_index=False).map(
        lambda batch: tokenizer(batch['sentence'], truncation=True,
                                max_length=config['max_length']),
        batched=True, remove_columns=['sentence'])
    prediction = trainer.predict(data, metric_key_prefix='test')
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.argmax(logits, axis=-1)
    report = classification_report(
        prediction.label_ids, predicted, labels=[0, 1, 2],
        target_names=[LABELS[dataset][i] for i in range(3)],
        output_dict=True, zero_division=0)
    write_json(folder / 'test_report.json', report)
    write_json(folder / 'test_metrics.json', prediction.metrics)
    write_json(folder / 'training_history.json', trainer.state.log_history)
    write_json(folder / 'selection.json', selection)
    frames['test'].assign(prediction=predicted).to_csv(folder / 'test_predictions.csv', index=False)
    return {'test_accuracy': report['accuracy'],
            'test_macro_f1': report['macro avg']['f1-score']}


def score_student(model, tokenizer, frame, dataset, batch_size, max_length):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    columns = {normalize_label(name, dataset): int(i)
               for i, name in model.config.id2label.items()}
    result = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            encoded = tokenizer(batch.sentence.tolist(), padding=True, truncation=True,
                                max_length=max_length, return_tensors='pt').to(device)
            probs = model(**encoded).logits.float().softmax(-1)
            indices = torch.tensor([columns[int(x)] for x in batch.label], device=device)
            result.extend(probs.gather(1, indices[:, None]).squeeze(1).cpu().tolist())
    model.cpu()
    release_gpu()
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['phrasebank', 'twitter'],
                        default=['phrasebank', 'twitter'])
    parser.add_argument('--variants', nargs='+', choices=['R', 'L', 'N', 'RL', 'RN', 'LN', 'RLN'],
                        default=['R', 'L', 'N', 'RL', 'RN', 'LN', 'RLN'])
    parser.add_argument('--output-root', type=Path, default=Path('output/ablation_ram_01'))
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=24266)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--twitter-train-file', type=Path)
    parser.add_argument('--twitter-validation-file', type=Path)
    parser.add_argument('--phrasebank-file', type=Path, default=Path(
        'data/financial_phrasebank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'))
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.num_epochs < 1:
        parser.error('Invalid learning rate or epochs')
    if bool(args.twitter_train_file) != bool(args.twitter_validation_file):
        parser.error('Provide both local Twitter files')
    for dataset in args.datasets:
        if (args.output_root / dataset).exists():
            parser.error(f'{args.output_root / dataset} already exists; choose a new output root')
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / 'test_results.csv'
    for dataset in args.datasets:
        folder = args.output_root / dataset
        seed_folder = folder / 'E1'
        seed_folder.mkdir(parents=True)
        config = dict(dataset=dataset, train_seed=args.train_seed, split_seed=args.split_seed,
                      learning_rate=args.learning_rate, num_epochs=args.num_epochs,
                      model='distilbert-base-uncased', model_revision='main', freeze_layers=4,
                      max_length=512, train_batch_size=16, eval_batch_size=32,
                      weight_decay=0.1, label_smoothing=0.1, early_stopping_patience=10)
        write_json(folder / 'run_config.json', {**config, 'variants': args.variants,
                   'selection': 'validation_macro_f1', 'trained_model_storage': 'RAM only'})
        frames, audit = prepare_data(
            dataset=dataset, seed_path=Path('outputs') / f'seed_{dataset}_data_clustered.jsonl',
            split_seed=args.split_seed, phrasebank_file=args.phrasebank_file,
            twitter_train_file=args.twitter_train_file,
            twitter_validation_file=args.twitter_validation_file, expected_seeds=105)
        write_json(folder / 'data_audit.json', audit)
        synthetic, removed = prepare_synthetic(
            Path('outputs') / f'synthetic_{dataset}_data_seed_clustered.jsonl', dataset, frames)
        write_json(folder / 'excluded_synthetic.json', removed)
        print(f'[{dataset}] Train fresh E1/M0', flush=True)
        trainer, tokenizer, _, _, _ = build_seed_trainer(SimpleNamespace(**config), seed_folder, frames)
        selection = fit_in_ram(trainer, config['early_stopping_patience'])
        metrics = test_and_save(trainer, tokenizer, frames, config, seed_folder, dataset, False, selection)
        # Independent CPU copy: optimizer and Trainer retain no reference to this M0.
        m0 = copy.deepcopy(trainer.model).cpu()
        del trainer
        release_gpu()

        def record(experiment, values):
            row = pd.DataFrame([{'dataset': dataset, 'experiment': experiment,
                                 'train_seed': args.train_seed, **values}])
            if csv_path.is_file():
                row = pd.concat([pd.read_csv(csv_path), row], ignore_index=True)
            row.to_csv(csv_path, index=False)
            print(f'{dataset} {experiment}: {values}', flush=True)

        record('E1', metrics)
        L = 1.0 - score_student(m0, tokenizer, synthetic, dataset, 32, config['max_length'])
        R = assigned_label_probabilities('ProsusAI/finbert', synthetic, dataset, 32, 512)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        seed_embeddings = encoder.encode(frames['seed'].sentence.tolist(),
                                         normalize_embeddings=True, batch_size=32)
        embeddings = encoder.encode(synthetic.sentence.tolist(), normalize_embeddings=True, batch_size=32)
        N = 1.0 - np.clip(embeddings @ seed_embeddings.T, -1.0, 1.0).max(axis=1)
        del encoder
        release_gpu()
        scores = {'R': R, 'L': L, 'N': N}
        synthetic.assign(R=R, L=L, N=N).to_csv(folder / 'component_scores.csv', index=False)
        for variant in args.variants:
            stage = folder / f'E3_{variant}'
            stage.mkdir()
            components = [scores[c] if c in variant else np.ones(len(synthetic)) for c in 'RLN']
            q, weights = make_weights(*components, 0.5, 2.0)
            synthetic.assign(Q=q, weight=weights).to_csv(stage / 'synthetic_weights.csv', index=False)
            train = pd.concat([frames['seed'].assign(weight=1.0),
                               synthetic[['sentence', 'label']].assign(weight=weights)], ignore_index=True)
            set_seed(args.train_seed)
            model = copy.deepcopy(m0)
            trainer = build_stage_trainer(args, stage, config, frames, train, model, tokenizer)
            print(f'[{dataset}] E3_{variant}: initialize from the same M0', flush=True)
            selection = fit_in_ram(trainer, config['early_stopping_patience'])
            metrics = test_and_save(trainer, tokenizer, frames, config, stage, dataset, True, selection)
            record(f'E3_{variant}', metrics)
            del trainer, model
            release_gpu()
        del m0, tokenizer
        release_gpu()
    print(f'Saved test results: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
