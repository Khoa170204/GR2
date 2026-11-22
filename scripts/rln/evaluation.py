"""Validation-best selection, metrics, and test predictions."""
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import TrainerCallback

from data_utils import LABELS
from run_io import write_json


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


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
