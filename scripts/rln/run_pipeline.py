"""One-command E1 -> E2 -> scoring -> E3, checkpoint reuse and comparison."""

import argparse
import subprocess
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import (
    read_json, write_json, frame_sha256, model_fingerprint,
    check_disjoint, prepare_synthetic,
)
from scoring import score_synthetic
from training import train_stage
from evaluation import evaluate_test


def parse_args():
    parser = argparse.ArgumentParser(description='Run/reuse E1, then E2, RLN scoring and E3.')
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["phrasebank", "twitter"],
        default=["phrasebank", "twitter"],
    )
    parser.add_argument('--output-root', type=Path, default=Path('output'))
    parser.add_argument('--split-seed', type=int, default=24266)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--reliability-model', default='ProsusAI/finbert')
    parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--weight-min', type=float, default=0.5)
    parser.add_argument('--weight-max', type=float, default=2.0)
    parser.add_argument('--score-batch-size', type=int, default=32)
    parser.add_argument('--twitter-train-file')
    parser.add_argument('--twitter-validation-file')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument(
        '--stages', nargs='+', choices=['E2', 'E3'], default=None,
        help='Run selected stages using an already completed E1. Default: full pipeline.',
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for E2/E3; default: inherit from E1.",
    )
    args = parser.parse_args()
    if args.learning_rate is not None:
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("--learning-rate must be finite and positive.")
    if (
        not 0 < args.weight_min <= args.weight_max
        or args.score_batch_size < 1
    ):
        parser.error('Invalid weight bounds or batch size.')
    if (
        bool(args.twitter_train_file)
        != bool(args.twitter_validation_file)
    ):
        parser.error('Provide both local Twitter files.')
    return args


def load_m0(args, dataset):
    run = (
        args.output_root
        / dataset
        / "seed_only"
        / f"train_seed_{args.train_seed}"
    )
    # Chỉ train E1 nếu chưa có thư mục lượt chạy.
    if not run.exists():
        if getattr(args, 'stages', None) is not None:
            raise FileNotFoundError(
                f'E1 is required before --stages: {run}. Run train_seed.py first.'
            )
        command = [
            sys.executable,
            str(Path(__file__).with_name('train_seed.py')),
            "--dataset", dataset,
            "--train-seed", str(args.train_seed),
            "--split-seed", str(args.split_seed),
            "--output-root", str(args.output_root),
        ]
        if (
            dataset == "twitter"
            and args.twitter_train_file
        ):
            command += [
                "--twitter-train-file",
                args.twitter_train_file,
                "--twitter-validation-file",
                args.twitter_validation_file,
            ]
        subprocess.run(command, check=True)
    summary_path = run / "summary.json"
    if (
        not summary_path.is_file()
        or read_json(summary_path)["status"]
        != "complete"
    ):
        raise ValueError(f'Incomplete M0 run: {run}. Keep it and inspect its training log.')
    config = read_json(run / 'run_config.json')
    if (
        config["dataset"] != dataset
        or config["train_seed"] != args.train_seed
        or config["split_seed"] != args.split_seed
    ):
        raise ValueError('Requested dataset/seeds do not match saved M0 config.')
    frames = {}
    for name in ("seed", "validation", "test"):
        frame = pd.read_json(run / 'data_snapshot' / f'{name}.jsonl', lines=True)
        if (
            frame_sha256(frame)
            != config["data_sha256"][name]
        ):
            raise ValueError(f'M0 snapshot changed: {name}')
        frames[name] = frame
    check_disjoint(frames)
    checkpoint = run / "best_model"
    digest = model_fingerprint(checkpoint)
    print(f'[E1 READY] Reuse M0: {checkpoint}', flush=True)
    return (run, checkpoint, config, frames, digest)


def main():
    args = parse_args()
    stages = set(args.stages if args.stages is not None else ['E2', 'E3'])
    rows = []
    for dataset in args.datasets:
        (m0_run, checkpoint, config, frames, digest) = load_m0(args, dataset)
        source = (
            Path("outputs")
            / (
                f"synthetic_{dataset}"
                "_data_seed_clustered.jsonl"
            )
        )
        synthetic, removed = prepare_synthetic(source, dataset, frames)
        prepared = (
            args.output_root
            / dataset
            / "pipeline_data"
            / f"train_seed_{args.train_seed}"
        )
        contract = {
            "m0": digest,
            "seed": frame_sha256(frames['seed']),
            "validation": frame_sha256(frames['validation']),
            "test": frame_sha256(frames['test']),
            "synthetic": frame_sha256(synthetic),
        }
        prepared.mkdir(parents=True, exist_ok=True)
        synthetic.to_json(
            prepared / "synthetic.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        write_json(prepared / 'excluded_synthetic.json', removed)
        write_json(prepared / 'contract.json', contract)
        print(
            f"[{dataset}] "
            f"seed={len(frames['seed'])}, "
            f"synthetic={len(synthetic)}, "
            f"excluded={len(removed)}",
            flush=True,
        )
        # E2: mọi mẫu có trọng số bằng 1.
        e2 = args.output_root / dataset / 'uniform' / f'train_seed_{args.train_seed}'
        e3 = args.output_root / dataset / 'weighted' / f'train_seed_{args.train_seed}'
        if 'E2' in stages:
            e2 = train_stage(
                args,
                dataset,
                "uniform",
                checkpoint,
                config,
                frames,
                synthetic,
                np.ones(len(synthetic)),
                contract,
            )
        # Chấm RLN bằng M0 cố định.
        if 'E3' in stages:
            scored, score_signature = (
                score_synthetic(args, dataset, checkpoint, frames, synthetic, contract)
            )
            if frame_sha256(scored) != frame_sha256(synthetic):
                raise ValueError('Scored sample order/content changed.')
            # E3: khởi tạo lại từ M0, KHÔNG từ E2.
            e3 = train_stage(
                args,
                dataset,
                "weighted",
                checkpoint,
                config,
                frames,
                synthetic,
                scored["weight"].to_numpy(),
                {**contract, 'scoring': score_signature},
            )
        for experiment, folder in (('E1', m0_run), ('E2', e2), ('E3', e3)):
            summary = folder / 'summary.json'
            if not summary.is_file() or read_json(summary).get('status') != 'complete':
                continue
            validation = read_json(folder / 'validation_metrics.json')
            row = {
                "dataset": dataset,
                "experiment": experiment,
                "validation_macro_f1": (
                    validation["eval_f1"]
                ),
                "validation_accuracy": (
                    validation["eval_accuracy"]
                ),
            }
            if args.evaluate_test:
                report = evaluate_test(
                    folder,
                    frames["test"],
                    dataset,
                    config,
                    args.score_batch_size,
                )
                row.update({
                    "test_macro_f1": (
                        report["macro avg"]["f1-score"]
                    ),
                    "test_accuracy": (
                        report["accuracy"]
                    ),
                })
            rows.append(row)
        comparison = args.output_root / f'comparison_train_seed_{args.train_seed}.csv'
        # Separate dataset commands must retain rows from earlier datasets.
        current = pd.DataFrame(rows)
        if comparison.is_file():
            old = pd.read_csv(comparison)
            keys = pd.MultiIndex.from_frame(current[['dataset', 'experiment']])
            keep = ~pd.MultiIndex.from_frame(old[['dataset', 'experiment']]).isin(keys)
            current = pd.concat([old.loc[keep], current], ignore_index=True)
        current.sort_values(['dataset', 'experiment']).to_csv(comparison, index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print('PIPELINE COMPLETE', flush=True)


if __name__ == "__main__":
    main()
