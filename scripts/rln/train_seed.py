"""E1: CLI, snapshots, seed-only model training and M0 persistence."""

import argparse
import importlib.metadata
import subprocess
from pathlib import Path


from data_utils import LABELS, file_sha256, write_json, prepare_data
from evaluation import compute_metrics, save_seed_result


def parse_args():
    parser = argparse.ArgumentParser(description='Train RLN initial student M0.')
    parser.add_argument('--dataset', required=True, choices=['phrasebank', 'twitter'])
    parser.add_argument('--seed-jsonl', type=Path, default=None)
    parser.add_argument(
        "--phrasebank-file",
        type=Path,
        default=Path('data/financial_phrasebank/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'),
    )
    parser.add_argument('--twitter-train-file', type=Path, default=None)
    parser.add_argument('--twitter-validation-file', type=Path, default=None)
    parser.add_argument('--split-seed', type=int, default=24266)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--expected-seeds', type=int, default=105)
    parser.add_argument('--output-root', type=Path, default=Path('outputs_rln'))
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--freeze-layers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--model', default='distilbert-base-uncased')
    parser.add_argument('--model-revision', default='main')
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Check and save data without training.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test after validation selection.",
    )
    args = parser.parse_args()
    positive_values = [
        args.num_epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.early_stopping_patience,
        args.expected_seeds,
        args.max_length,
    ]
    if min(positive_values) <= 0:
        parser.error('Epochs, batch sizes, patience, seed count and max length must be positive.')
    if (
        not 0 <= args.label_smoothing < 1
        or args.freeze_layers < 0
        or args.learning_rate <= 0
        or args.weight_decay < 0
    ):
        parser.error("Invalid training hyperparameters.")
    if args.seed_jsonl is None:
        filename = (
            "seed_phrasebank_data_clustered.jsonl"
            if args.dataset == "phrasebank"
            else "seed_twitter_data_clustered.jsonl"
        )
        args.seed_jsonl = Path("outputs") / filename
    return args


def prepare_seed_run(args):
    stage = (
        "data_check"
        if args.prepare_only
        else "seed_only"
    )
    output = (
        args.output_root
        / args.dataset
        / stage
        / f"train_seed_{args.train_seed}"
    )
    if output.exists():
        raise FileExistsError(f'{output} already exists. Use a new --output-root.')
    frames, audit = prepare_data(
        dataset=args.dataset,
        seed_path=args.seed_jsonl,
        split_seed=args.split_seed,
        phrasebank_file=args.phrasebank_file,
        twitter_train_file=args.twitter_train_file,
        twitter_validation_file=(
            args.twitter_validation_file
        ),
        expected_seeds=args.expected_seeds,
    )
    print("Data counts:", audit["counts"], flush=True)
    print('Seed class counts:', audit['class_counts']['seed'], flush=True)
    output.mkdir(parents=True)
    snapshot = output / "data_snapshot"
    snapshot.mkdir()
    for name, frame in frames.items():
        frame.to_json(snapshot / f'{name}.jsonl', orient='records', lines=True, force_ascii=False)
    write_json(output / "data_audit.json", audit)
    config = {
        name: (
            str(value.resolve())
            if isinstance(value, Path)
            else value
        )
        for name, value in vars(args).items()
    }
    config.update({
        "checkpoint_metric": "validation_macro_f1",
        "data_sha256": audit["sha256"],
        "script_sha256": {
            path.name: file_sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    })
    try:
        config["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        config["git_commit"] = None
    write_json(output / 'run_config.json', config)
    return output, frames


def build_seed_trainer(args, output, frames):
    import torch
    from datasets import Dataset
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        set_seed,
    )
    set_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    versions = {
        package: importlib.metadata.version(package)
        for package in (
            "torch",
            "transformers",
            "tokenizers",
            "datasets",
            "accelerate",
            "scikit-learn",
        )
    }
    write_json(
        output / "environment.json",
        {
            "versions": versions,
            "cuda": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    )
    print(
        "Device:",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        flush=True,
    )
    names = LABELS[args.dataset]
    tokenizer = (
        DistilBertTokenizerFast.from_pretrained(args.model, revision=args.model_revision)
    )
    model = (
        DistilBertForSequenceClassification
        .from_pretrained(
            args.model,
            revision=args.model_revision,
            num_labels=3,
            id2label=names,
            label2id={label: index for index, label in names.items()},
        )
    )
    if (
        args.max_length
        > model.config.max_position_embeddings
    ):
        raise ValueError('max-length exceeds model positional limit.')
    total_layers = len(model.distilbert.transformer.layer)
    if args.freeze_layers > total_layers:
        raise ValueError('freeze-layers exceeds model layer count.')
    for layer in (
        model.distilbert.transformer.layer[
            :args.freeze_layers
        ]
    ):
        for parameter in layer.parameters():
            parameter.requires_grad = False
    print(f'Frozen lower layers: {args.freeze_layers}', flush=True)
    def tokenize(frame):
        dataset = Dataset.from_pandas(frame[['sentence', 'label']], preserve_index=False)
        return dataset.map(
            lambda batch: tokenizer(batch['sentence'], truncation=True, max_length=args.max_length),
            batched=True,
            remove_columns=["sentence"],
        )
    train_data = tokenize(frames["seed"])
    validation_data = tokenize(frames['validation'])
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        seed=args.train_seed,
        data_seed=args.train_seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=(
            args.train_batch_size
        ),
        per_device_eval_batch_size=(
            args.eval_batch_size
        ),
        weight_decay=args.weight_decay,
        label_smoothing_factor=(
            args.label_smoothing
        ),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        warmup_ratio=0.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    return trainer, tokenizer, validation_data, tokenize, names


def main():
    args = parse_args()
    output, frames = prepare_seed_run(args)
    if args.prepare_only:
        print(f"DATA CHECK PASSED: {output}", flush=True)
        return
    trainer, tokenizer, validation_data, tokenize, names = build_seed_trainer(args, output, frames)
    result = trainer.train()
    save_seed_result(
        args, output, frames, trainer, tokenizer, validation_data, tokenize, names, result
    )


if __name__ == "__main__":
    main()
