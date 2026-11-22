"""Construct the seed-only and weighted DistilBERT trainers."""
import importlib.metadata

from data_utils import LABELS
from evaluation import compute_metrics
from run_io import write_json


def build_seed_trainer(args, output, frames):
    import torch
    from datasets import Dataset
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
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
        output_dir=str(output),
        seed=args.train_seed,
        data_seed=args.train_seed,
        eval_strategy="epoch",
        save_strategy="no",
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
        load_best_model_at_end=False,
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
    )
    return trainer, tokenizer, validation_data, tokenize, names


def build_stage_trainer(args, folder, config, frames, train, model, tokenizer):
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
    def tokenize(frame):
        data = Dataset.from_pandas(frame[['sentence', 'label', 'weight']], preserve_index=False)
        return data.map(
            lambda batch: tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=config["max_length"],
            ),
            batched=True,
            remove_columns=["sentence"],
        )
    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def collate(features):
        copies = [dict(item) for item in features]
        sample_weights = torch.tensor([item.pop('weight') for item in copies], dtype=torch.float32)
        batch = padding_collator(copies)
        batch["weight"] = sample_weights
        return batch
    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            inputs = dict(inputs)
            labels = inputs.pop("labels")
            sample_weights = (
                inputs.pop("weight").float()
            )
            outputs = model(**inputs)
            losses = F.cross_entropy(
                outputs.logits.float(),
                labels,
                reduction="none",
                label_smoothing=(
                    config["label_smoothing"]
                ),
            )
            loss = (sample_weights * losses).sum() / (
                sample_weights.sum().clamp_min(1e-12)
            )
            if return_outputs:
                return loss, outputs
            return loss
    validation = (
        frames['validation'].assign(weight=1.0)
    )
    training_args = TrainingArguments(
        output_dir=str(folder),
        seed=args.train_seed,
        data_seed=args.train_seed,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=(
            config["train_batch_size"]
        ),
        per_device_eval_batch_size=(
            config["eval_batch_size"]
        ),
        weight_decay=config["weight_decay"],
        # Label smoothing được tính trong custom loss.
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        warmup_ratio=0.0,
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_accumulation_steps=1,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=tokenize(train),
        eval_dataset=tokenize(validation),
        data_collator=collate,
        compute_metrics=compute_metrics,
    )
    trainer.model_accepts_loss_kwargs = False
    return trainer
