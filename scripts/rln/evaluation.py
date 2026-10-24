"""Final test inference and result caching."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support,
)

from data_utils import (
    read_json, write_json, fingerprint, model_fingerprint, frame_sha256, LABELS,
)
from scoring import release_gpu


def evaluate_test(
    folder,
    test,
    dataset,
    config,
    batch_size,
):
    import torch
    from sklearn.metrics import (
        classification_report,
    )
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    checkpoint = folder / "best_model"
    signature = fingerprint({'model': model_fingerprint(checkpoint), 'test': frame_sha256(test)})
    marker = (
        folder / "pipeline_test_complete.json"
    )
    if (
        marker.exists()
        and read_json(marker)["signature"]
        == signature
    ):
        return read_json(folder / 'pipeline_test_report.json')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
        .to(device)
        .eval()
    )
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            encoded = tokenizer(
                test.iloc[start:start + batch_size]['sentence'].tolist(),
                padding=True,
                truncation=True,
                max_length=config["max_length"],
                return_tensors="pt",
            ).to(device)
            predictions.extend(model(**encoded).logits.argmax(-1).cpu().tolist())
    report = classification_report(
        test["label"],
        predictions,
        labels=[0, 1, 2],
        target_names=[LABELS[dataset][index] for index in range(3)],
        output_dict=True,
        zero_division=0,
    )
    write_json(folder / 'pipeline_test_report.json', report)
    test.assign(prediction=predictions).to_csv(
        folder / "pipeline_test_predictions.csv",
        index=False,
    )
    write_json(marker, {'signature': signature})
    del model, tokenizer
    release_gpu()
    return report


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


def save_seed_result(args, output, frames, trainer, tokenizer, validation_data, tokenize, names, result):
    best_path = output / "best_model"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    write_json(output / 'training_history.json', trainer.state.log_history)
    write_json(output / 'train_metrics.json', result.metrics)
    validation_metrics = trainer.evaluate(eval_dataset=validation_data)
    write_json(output / 'validation_metrics.json', validation_metrics)
    summary = {
        "status": "m0_saved",
        "dataset": args.dataset,
        "train_seed": args.train_seed,
        "best_model": str(best_path),
        "best_validation_macro_f1": (
            trainer.state.best_metric
        ),
        "best_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
        "test_evaluated": False,
    }
    write_json(output / 'summary.json', summary)
    if args.evaluate_test:
        prediction = trainer.predict(tokenize(frames['test']))
        predicted = np.argmax(prediction.predictions, axis=-1)
        report = classification_report(
            prediction.label_ids,
            predicted,
            labels=[0, 1, 2],
            target_names=[names[index] for index in range(3)],
            output_dict=True,
            zero_division=0,
        )
        write_json(output / 'test_classification_report.json', report)
        write_json(output / 'test_metrics.json', prediction.metrics)
        table = frames["test"].copy()
        table["prediction"] = predicted
        table.to_csv(output / 'test_predictions.csv', index=False)
        summary["test_evaluated"] = True
    summary["status"] = "complete"
    write_json(output / 'summary.json', summary)
    print(f'M0 SAVED: {best_path}', flush=True)
