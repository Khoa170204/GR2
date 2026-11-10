import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Configuration: same clustered-seed method as PhraseBank/Twitter.
SAMPLES_PER_CLASS = 35
USE_CLASS_BALANCED_CLUSTERING = True
SEED = 24266
SAVE_PATH = "./outputs/seed_finsent_data_clustered.jsonl"
DATA_DIR = Path("./data/finsent/prepared")
DATASET = "FinanceMTEB/FinSent"
REVISION = "68ee0f0abf596e371ef6a308f685071e3b737bbb"
CLASS_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    dataset = load_dataset(DATASET, revision=REVISION)
    frames = {}
    for split in ["train", "test"]:
        df = dataset[split].to_pandas()
        # Source numbers differ from GR2: use semantic names.
        names = df["label_text"].str.strip().str.lower()
        if df[["text", "label_text"]].isna().any().any():
            raise ValueError(f"Missing text/label in {split}.")
        if not names.isin(["negative", "neutral", "positive"]).all():
            raise ValueError(f"Unexpected labels in {split}.")
        frames[split] = pd.DataFrame({
            "sentence": df["text"].map(lambda s: " ".join(s.split())),
            "label": names.map({"negative": 0, "neutral": 1, "positive": 2}),
            "source_id": [f"{split}:{i}" for i in range(len(df))],
        })
    return frames["train"], frames["test"]


def stratified_split(train_data, test_data):
    # Preserve the official 1,000-row test; reserve 1,000 validation rows.
    train_df, val_df = train_test_split(
        train_data, test_size=1000,
        stratify=train_data["label"], random_state=SEED,
    )
    return tuple(df.reset_index(drop=True) for df in [train_df, val_df, test_data])


def save_splits(train_df, val_df, test_df):
    frames = {"train_pool": train_df, "validation": val_df, "test": test_df}
    keys = {}
    for name, frame in frames.items():
        keys[name] = frame["sentence"].map(lambda s: " ".join(s.casefold().split()))
        if keys[name].duplicated().any() or frame["sentence"].str.strip().eq("").any():
            raise ValueError(f"Duplicate/empty text in {name}.")
    names = list(keys)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if set(keys[left]) & set(keys[right]):
                raise ValueError(f"Text overlap: {left}/{right}.")
    contents = {f"{name}.jsonl": frame.to_json(
        orient="records", lines=True, force_ascii=False) for name, frame in frames.items()}
    # Never replace a different split left by an earlier run.
    for filename, content in contents.items():
        path = DATA_DIR / filename
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"{path} contains another split; inspect it before rerunning.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in contents.items():
        (DATA_DIR / filename).write_text(content, encoding="utf-8")
    manifest_path = DATA_DIR / "manifest.json"
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
              for name, text in contents.items()}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file_sha256") != hashes:
            raise ValueError("Existing manifest does not match the fixed splits.")
    else:
        manifest = {
            "settings": {"dataset": DATASET, "revision": REVISION, "split_seed": SEED,
                         "validation_size": 1000, "label_map": CLASS_NAMES,
                         "test_protocol": "original official test unchanged"},
            "counts": {name: len(frame) for name, frame in frames.items()},
            "class_counts": {name: frame.label.value_counts().sort_index().to_dict()
                             for name, frame in frames.items()},
            "file_sha256": hashes, "normalized_cross_set_overlap": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate_embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences, show_progress_bar=True)


def cluster_embeddings(embeddings, num_clusters):
    model = KMeans(n_clusters=num_clusters, random_state=SEED)
    model.fit(embeddings)
    return model


def closest_cluster_members(cluster_model, embeddings):
    closest = []
    for cluster_id, center in enumerate(cluster_model.cluster_centers_):
        members = np.flatnonzero(cluster_model.labels_ == cluster_id)
        if len(members) == 0:
            raise ValueError(f'Empty cluster {cluster_id}; inspect embeddings.')
        distances = np.linalg.norm(embeddings[members] - center, axis=1)
        closest.append(int(members[np.argmin(distances)]))
    return closest


def select_seed_data(df):
    if USE_CLASS_BALANCED_CLUSTERING:
        seed_frames = []
        for label in sorted(df["label"].unique()):
            subset = df[df["label"] == label].reset_index(drop=True)
            embeddings = generate_embeddings(subset["sentence"].tolist())
            cluster_model = cluster_embeddings(embeddings, SAMPLES_PER_CLASS)
            closest = closest_cluster_members(cluster_model, embeddings)
            seed_frames.append(subset.iloc[closest])
        seed_df = pd.concat(seed_frames).reset_index(drop=True)
    else:
        embeddings = generate_embeddings(df["sentence"].tolist())
        cluster_model = cluster_embeddings(embeddings, 105)
        closest = closest_cluster_members(cluster_model, embeddings)
        seed_df = df.iloc[closest].reset_index(drop=True)
    if seed_df["sentence"].duplicated().any():
        raise ValueError("Cluster representatives repeat; inspect before generating synthetic data.")
    return seed_df


def convert_to_instruction_format(row):
    return {
        "instruction": "Classify the sentiment of this financial analyst report sentence.",
        "input": row["sentence"],
        "output": CLASS_NAMES[int(row["label"])],
    }


def save_as_jsonl(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(convert_to_instruction_format(row), ensure_ascii=False) + "\n")
    print(f"Seed data saved to: {path}")


def main():
    set_seed()
    train_data, test_data = load_data()
    train_df, val_df, test_df = stratified_split(train_data, test_data)
    save_splits(train_df, val_df, test_df)
    print(f"Train pool: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}")
    seed_df = select_seed_data(train_df)
    print(f"Selected seed size: {len(seed_df)}")
    print(seed_df["label"].map(CLASS_NAMES).value_counts())
    save_as_jsonl(seed_df, SAVE_PATH)


if __name__ == "__main__":
    main()
