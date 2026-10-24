import hashlib
import json
import numbers
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS = {
    "phrasebank": {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    },
    "twitter": {
        0: "Bearish",
        1: "Bullish",
        2: "Neutral",
    },
}

TEXT_COLUMNS = (
    "sentence",
    "text",
    "tweet",
    "content",
    "document",
    "message",
    "headline",
)


def clean_text(text):
    """
    Chỉ dùng cho nhánh RLN mới.
    Giữ Unicode, dấu câu, ticker và chữ hoa/chữ thường.
    """
    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError("Empty text.")

    return text


def normalize_label(value, dataset):
    """Ánh xạ nhãn theo đúng dataset."""

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if (
            pd.notna(value)
            and float(value).is_integer()
            and int(value) in LABELS[dataset]
        ):
            return int(value)

        raise ValueError(f"Invalid numeric label: {value!r}")

    if isinstance(value, str):
        aliases = {
            "bearish": "negative",
            "bullish": "positive",
        }

        label = value.strip().lower()
        label = aliases.get(label, label)

        canonical = {
            aliases.get(name.lower(), name.lower()): index
            for index, name in LABELS[dataset].items()
        }

        if label in canonical:
            return canonical[label]

    raise ValueError(f"Invalid {dataset} label: {value!r}")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha256(frame):
    records = frame[["sentence", "label"]].to_dict("records")

    content = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_seed(path, dataset):
    """Đọc seed JSONL có các trường input và output."""

    records = []

    lines = Path(path).read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        row = json.loads(line)

        if (
            not isinstance(row, dict)
            or not {"input", "output"}.issubset(row)
        ):
            raise ValueError(
                f"Seed line {line_number}: expected input/output."
            )

        records.append({
            "sentence": clean_text(row["input"]),
            "label": normalize_label(row["output"], dataset),
        })

    frame = pd.DataFrame(
        records,
        columns=["sentence", "label"],
    )

    if frame.empty:
        raise ValueError("Seed file is empty.")

    if frame["sentence"].duplicated().any():
        raise ValueError(
            "Duplicate seed text. Review the JSONL file."
        )

    return frame


def load_phrasebank(path):
    """Đọc PhraseBank từ file local giống nguồn bạn đang dùng."""

    records = []

    lines = Path(path).read_text(
        encoding="latin-1"
    ).splitlines()

    for line in lines:
        if not line.strip():
            continue

        sentence, label = line.strip().rsplit("@", 1)

        records.append({
            "sentence": clean_text(sentence),
            "label": normalize_label(label, "phrasebank"),
        })

    frame = pd.DataFrame(records)

    if frame.empty:
        raise ValueError("PhraseBank file is empty.")

    return frame


def read_local_table(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

    if path.suffix.lower() == ".json":
        return pd.read_json(path)

    raise ValueError(
        "Local Twitter files must be CSV, JSONL, or JSON."
    )


def load_twitter(train_file=None, validation_file=None):
    """
    Giữ thứ tự ghép của pipeline cũ:
    original train -> original validation.
    """

    if bool(train_file) != bool(validation_file):
        raise ValueError(
            "Supply both Twitter train and validation files."
        )

    if train_file:
        first = read_local_table(train_file)
        second = read_local_table(validation_file)

    else:
        from datasets import load_dataset

        data = load_dataset(
            "zeroshot/twitter-financial-news-sentiment"
        )

        first = data["train"].to_pandas()
        second = data["validation"].to_pandas()

    frame = pd.concat(
        [first, second],
        ignore_index=True,
    )

    text_column = next(
        (
            name
            for name in TEXT_COLUMNS
            if name in frame.columns
        ),
        None,
    )

    if text_column is None:
        raise ValueError(
            f"No recognized Twitter text column: "
            f"{list(frame.columns)}"
        )

    label_column = (
        "label_text"
        if "label_text" in frame.columns
        else "label"
    )

    if label_column not in frame.columns:
        raise ValueError("Twitter data has no label column.")

    return pd.DataFrame({
        "sentence": frame[text_column].map(clean_text),
        "label": frame[label_column].map(
            lambda value: normalize_label(value, "twitter")
        ),
    })


def make_splits(frame, split_seed):
    """
    Giữ chia stratified 80/10/10.

    Không sắp xếp hoặc bỏ dòng trước khi chia, vì có thể
    làm thay đổi membership của seed đã chọn trước đây.
    """

    train, remainder = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame["label"],
        random_state=split_seed,
    )

    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        stratify=remainder["label"],
        random_state=split_seed,
    )

    return tuple(
        part.reset_index(drop=True)
        for part in (train, validation, test)
    )


def check_disjoint(named_frames):
    """Kiểm tra giao văn bản chính xác sau chuẩn hóa."""

    names = list(named_frames)

    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = (
                set(named_frames[left]["sentence"])
                & set(named_frames[right]["sentence"])
            )

            if overlap:
                raise ValueError(
                    f"Text overlap {left}/{right}: "
                    f"{len(overlap)}. Stop and review data."
                )


def prepare_data(
    dataset,
    seed_path,
    split_seed,
    phrasebank_file,
    twitter_train_file=None,
    twitter_validation_file=None,
    expected_seeds=105,
):
    if dataset == "phrasebank":
        full = load_phrasebank(phrasebank_file)
    else:
        full = load_twitter(
            twitter_train_file,
            twitter_validation_file,
        )

    train_pool, validation, test = make_splits(
        full,
        split_seed,
    )

    seeds = read_seed(seed_path, dataset)

    if len(seeds) != expected_seeds:
        raise ValueError(
            f"Expected {expected_seeds} seeds, "
            f"got {len(seeds)}."
        )

    if set(seeds["label"]) != {0, 1, 2}:
        raise ValueError(
            "Seed set must contain all three classes."
        )

    pool_pairs = set(zip(
        train_pool["sentence"],
        train_pool["label"],
    ))

    outside = [
        (text, label)
        for text, label in zip(
            seeds["sentence"],
            seeds["label"],
        )
        if (text, label) not in pool_pairs
    ]

    if outside:
        raise ValueError(
            f"{len(outside)} seed rows do not match "
            "labeled train-pool rows. Check dataset version, "
            "row order, label mapping and split seed."
        )

    check_disjoint({
        "seed": seeds,
        "validation": validation,
        "test": test,
    })

    snapshot = {
        "train_pool": train_pool,
        "seed": seeds,
        "validation": validation,
        "test": test,
    }

    audit = {
        "dataset": dataset,
        "split_seed": split_seed,
        "labels": LABELS[dataset],
        "seed_file_sha256": file_sha256(seed_path),
        "normalization": (
            "Collapse whitespace; retain Unicode, "
            "punctuation and case."
        ),
        "counts": {
            name: len(frame)
            for name, frame in snapshot.items()
        },
        "class_counts": {
            name: {
                int(label): int(count)
                for label, count in (
                    frame["label"]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            }
            for name, frame in snapshot.items()
        },
        "sha256": {
            name: frame_sha256(frame)
            for name, frame in snapshot.items()
        },
    }

    return snapshot, audit