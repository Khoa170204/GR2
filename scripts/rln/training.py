"""E2/E3 model loading, weighted loss, Trainer, resume and stage persistence."""

import numpy as np
import pandas as pd

from data_utils import LABELS, fingerprint, read_json, write_json, model_fingerprint
from scoring import release_gpu
from evaluation import compute_metrics


def build_stage_model(args, dataset, checkpoint, config):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
    set_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    )
    actual = {int(index): name for index, name in model.config.id2label.items()}
    if actual != LABELS[dataset]:
        raise ValueError('M0 label mapping mismatch.')
    for layer in (
        model.distilbert.transformer.layer[
            :config["freeze_layers"]
        ]
    ):
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return model, tokenizer


def build_stage_trainer(args, folder, config, frames, train, model, tokenizer):
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
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
        output_dir=str(folder / 'checkpoints'),
        seed=args.train_seed,
        data_seed=args.train_seed,
        eval_strategy="epoch",
        save_strategy="epoch",
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
        load_best_model_at_end=True,
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
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])
        ],
    )
    trainer.model_accepts_loss_kwargs = False
    return trainer


def train_stage(args, dataset, stage, checkpoint, config, frames, synthetic, weights, contract):
    from transformers.trainer_utils import get_last_checkpoint
    folder = (
        args.output_root
        / dataset
        / stage
        / f"train_seed_{args.train_seed}"
    )
    signature = fingerprint({
        "stage": stage,
        "input": contract,
        "weights": np.asarray(weights).tolist(),
        "config": config,
        "loss": "sum(w*smoothedCE)/sum(w)",
    })
    marker = folder / "stage_contract.json"
    if folder.exists():
        if (
            not marker.exists()
            or read_json(marker)["signature"]
            != signature
        ):
            raise ValueError(f'{folder} belongs to another run/script. Do not overwrite it.')
        summary_path = folder / "summary.json"
        if (
            summary_path.exists()
            and read_json(summary_path)["status"]
            == "complete"
        ):
            model_fingerprint(folder / 'best_model')
            print(f'[{stage.upper()} SKIP] Already complete', flush=True)
            return folder
    folder.mkdir(parents=True, exist_ok=True)
    write_json(marker, {'signature': signature})
    write_json(
        folder / "run_config.json",
        {
            "stage": stage,
            "dataset": dataset,
            "m0_config": config,
            "input_contract": contract,
            "initialization": str(checkpoint),
            "loss": "sum(w*smoothedCE)/sum(w)",
            "optimizer_reinitialized": True,
        },
    )
    model, tokenizer = build_stage_model(args, dataset, checkpoint, config)
    seed = frames["seed"].copy()
    seed["weight"] = 1.0
    syn = synthetic[['sentence', 'label']].copy()
    syn["weight"] = np.asarray(weights)
    train = pd.concat([seed, syn], ignore_index=True)
    snapshots = folder / "data_snapshot"
    snapshots.mkdir(exist_ok=True)
    train.to_json(snapshots / 'train.jsonl', orient='records', lines=True, force_ascii=False)
    trainer = build_stage_trainer(args, folder, config, frames, train, model, tokenizer)
    checkpoints = folder / "checkpoints"
    last = (
        get_last_checkpoint(str(checkpoints))
        if checkpoints.exists()
        else None
    )
    print(f'[{stage.upper()}] Train from M0; resume={last}', flush=True)
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(folder / 'best_model'))
    tokenizer.save_pretrained(str(folder / 'best_model'))
    write_json(folder / 'training_history.json', trainer.state.log_history)
    metrics = trainer.evaluate()
    write_json(folder / 'validation_metrics.json', metrics)
    write_json(
        folder / "summary.json",
        {
            "status": "complete",
            "stage": stage,
            "best_validation_macro_f1": (
                metrics["eval_f1"]
            ),
            "initialization": str(checkpoint),
        },
    )
    del trainer, model, tokenizer
    release_gpu()
    return folder
