import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# === CONFIGURATION ===
SAMPLES_PER_CLASS = 35
USE_CLASS_BALANCED_CLUSTERING = True   # (unused now; we load seeds from file)
SEED = 24266
FIGSIZE = (12, 6)
SAVE_DPI = 300
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
SAVE_BBOX = 'tight'
SAVE_PAD  = 0.03
LIM_PAD_FRAC = 0.06

# Output directory.
# This will be changed in main() based on the dataset.
OUTPUT_DIR = "./outputs"

# Preferred local paths
DATASET = "phrasebank"

DATASET_FILES = {
    "phrasebank": {
        "seed": "./outputs/seed_data_phrasebank_random.jsonl",
        "synthetic": "./outputs/synthetic_phrasebank_data_seed_random.jsonl",
    },
    "twitter": {
        "seed": "./outputs/seed_data_twitter_random.jsonl",
        "synthetic": "./outputs/synthetic_twitter_data_seed_random.jsonl",
    },
}

os.makedirs("./outputs", exist_ok=True)


def set_seed(seed=24266):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# Reduce figure border by default
plt.rcParams.update({
    "savefig.bbox": SAVE_BBOX,
    "savefig.pad_inches": SAVE_PAD,
})


# === LABELS & COLORS ===

DATASET_LABELS = {
    "phrasebank": {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive"
    },
    "twitter": {
        "bearish": "Bearish",
        "bullish": "Bullish",
        "neutral": "Neutral"
    }
}

LABEL_NAMES = DATASET_LABELS[DATASET]

# IMPORTANT:
# Labels are strings in this file:
# PhraseBank: negative / neutral / positive
# Twitter: bearish / bullish / neutral
COLOR_ALL = {"negative": "lightcoral","neutral": "lightblue","positive": "lightgreen","bearish": "lightcoral","bullish": "lightgreen",}
COLOR_SEED = {"negative": "#E41A1C","neutral": "#377EB8","positive": "#4DAF4A","bearish": "#E41A1C","bullish": "#4DAF4A",}
COLOR_SEED_PC = {"negative": "darkred","neutral": "darkblue","positive": "darkgreen","bearish": "darkred","bullish": "darkgreen",}

COLOR_REAL = {"negative": "skyblue","neutral": "lightgray","positive": "lightgreen","bearish": "skyblue","bullish": "lightgreen",}
COLOR_SYN = {"negative": "blue","neutral": "black","positive": "green","bearish": "blue","bullish": "green",}
COLOR_SEED_X = {"negative": "red","neutral": "orange","positive": "purple","bearish": "red","bullish": "purple",}


# === HELPERS ===

def _new_fig():
    # layout='constrained' shrinks whitespace
    return plt.figure(
        figsize=FIGSIZE,
        layout='constrained'
    )


def _prep_ax_equal(ax):
    ax.set_aspect('equal', adjustable='box')


def _expand_limits(vmin, vmax, frac=LIM_PAD_FRAC):
    span = vmax - vmin
    pad = span * frac if span > 0 else frac
    return vmin - pad, vmax + pad

def _resolve_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"None of the candidate paths exist: {paths}"
    )

# === IO FUNCTIONS (JSONL) ===
def load_seed_jsonl(path):
    """
    Loads seed JSONL written by your generator.
    First line may be {"seed_used": ...}; we skip any line missing
    input/output.
    Returns DataFrame with columns: sentence, label.
    """

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "output" in obj:
                sent = str(
                    obj["input"]
                ).strip()
                out = str(
                    obj["output"]
                ).strip()
                label = out.lower()
                if label not in LABEL_NAMES:
                    label = None
                if label is not None:
                    rows.append({
                        "sentence": sent,
                        "label": label
                    })
            # else: skip metadata row
    if not rows:
        raise ValueError(
            f"No valid seed rows found in {path}"
        )
    return pd.DataFrame(rows)

def load_synth_jsonl(path):
    """
    Loads synthetic JSONL (input/output keys).
    Returns DataFrame with columns: sentence, label.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "input" in obj and "output" in obj:
                sent = str(
                    obj["input"]
                ).strip()
                out = str(
                    obj["output"]
                ).strip()
                label = out.lower()
                if label not in LABEL_NAMES:
                    label = None
                if label is not None:
                    rows.append({
                        "sentence": sent,
                        "label": label
                    })
    if not rows:
        raise ValueError(
            f"No valid synthetic rows found in {path}"
        )
    return pd.DataFrame(rows)

# === DATA FUNCTIONS ===
# def load_data():
#     dataset = load_dataset(
#         "takala/financial_phrasebank",
#         "sentences_allagree",
#         split="train",
#         trust_remote_code=True
#     )
#     return pd.DataFrame(dataset)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize random seed selection"
    )
    parser.add_argument(
        "--dataset",
        choices=["phrasebank", "twitter"],
        required=True
    )
    return parser.parse_args()

def load_data():
    print("Loading data...")
    if DATASET == "twitter":
        ds_train = load_dataset(
            "zeroshot/twitter-financial-news-sentiment",
            split="train",
            trust_remote_code=True
        )
        ds_val = load_dataset(
            "zeroshot/twitter-financial-news-sentiment",
            split="validation",
            trust_remote_code=True
        )
        train_df = pd.DataFrame(ds_train)
        val_df = pd.DataFrame(ds_val)
        df = pd.concat(
            [train_df, val_df],
            ignore_index=True
        )
        text_col = (
            "sentence"
            if "sentence" in df.columns
            else "text"
        )
        def normalize_label(x):
            if isinstance(x, str):
                mapping = {
                    "bearish": "bearish",
                    "bullish": "bullish",
                    "neutral": "neutral"
                }
                return mapping[
                    x.strip().lower()
                ]
            mapping = {
                0: "bearish",
                1: "bullish",
                2: "neutral"
            }
            return mapping[int(x)]
        df = pd.DataFrame({
            "sentence":
                df[text_col]
                .astype(str)
                .str.strip(),
            "label":
                df["label"]
                .apply(normalize_label)
        })
        print(f"Loaded {len(df)} samples")
        print("\nLabel distribution:")
        print(df["label"].value_counts())
        return df
    
    # =========================
    # PhraseBank
    # =========================
    file_path = ("data/financial_phrasebank/""FinancialPhraseBank-v1.0/""Sentences_AllAgree.txt")
    data = []
    try:
        with open(file_path,"r",encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sentence, label = (line.rsplit("@", 1))
                data.append({
                    "sentence": sentence,
                    "label": label.lower()
                })
    except UnicodeDecodeError:
        with open(file_path,"r",encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sentence, label = (line.rsplit("@", 1))
                data.append({
                    "sentence": sentence,
                    "label": label.lower()
                })

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    return df

def generate_embeddings(sentences):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(sentences,show_progress_bar=True)

# === PLOTTING ===
def plot_tsne_projection(
    df,
    seed_indices,
    embeddings
):
    reduced = TSNE(
        n_components=2,
        random_state=SEED
    ).fit_transform(embeddings)

    # padded limits
    x_min, x_max = _expand_limits(
        reduced[:, 0].min(),
        reduced[:, 0].max()
    )
    y_min, y_max = _expand_limits(
        reduced[:, 1].min(),
        reduced[:, 1].max()
    )
    _new_fig()
    ax = plt.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    _prep_ax_equal(ax)
    # All data
    for label in sorted(df['label'].unique()):
        idx = df[
            df['label'] == label
        ].index
        plt.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            c=COLOR_ALL[label],
            alpha=0.3,
            label=f"{LABEL_NAMES[label]} (All)"
        )
    # Overlay seeds
    for label in sorted(df['label'].unique()):
        seed_sub = [
            i for i in seed_indices
            if df.iloc[i]['label'] == label
        ]
        if seed_sub:
            plt.scatter(
                reduced[seed_sub, 0],
                reduced[seed_sub, 1],
                c=COLOR_SEED[label],
                s=70,
                alpha=1.0,
                label=f"{LABEL_NAMES[label]} (Seed)"
            )
    plt.title("t-SNE Projection of Embeddings ""with Random Seed Samples by Class",fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    output_path = os.path.join(OUTPUT_DIR,"tsne_projection_labeled_random_seed.png")
    plt.savefig(output_path,dpi=SAVE_DPI,bbox_inches=SAVE_BBOX,pad_inches=SAVE_PAD)
    plt.show()
    print(f"Saved: {output_path}")
    return (reduced,(x_min, x_max),(y_min, y_max))

def plot_tsne_projection_per_class(df,seed_indices,reduced,xlim,ylim):
    for label in sorted(df['label'].unique()):
        _new_fig()
        ax = plt.gca()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        _prep_ax_equal(ax)
        idx = df[
            df['label'] == label
        ].index
        seed_sub = [
            i for i in seed_indices
            if df.iloc[i]['label'] == label
        ]
        plt.scatter(reduced[idx, 0],reduced[idx, 1],c=COLOR_ALL[label],alpha=0.3,label=f"{LABEL_NAMES[label]} (All)")
        if seed_sub:
            plt.scatter(reduced[seed_sub, 0],reduced[seed_sub, 1],c=COLOR_SEED_PC[label],label=f"{LABEL_NAMES[label]} (Seed)")
        plt.title(f"t-SNE Projection : "f"{LABEL_NAMES[label]} "f"(Random Seeds)",fontsize=TITLE_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True)
        fname = os.path.join(OUTPUT_DIR,f"tsne_projection_"f"{LABEL_NAMES[label].lower()}"f"_filtered_random_seed.png")
        plt.savefig(
            fname,
            dpi=SAVE_DPI,
            bbox_inches=SAVE_BBOX,
            pad_inches=SAVE_PAD
        )

        plt.show()

        print(
            f"Saved: {fname}"
        )


def plot_tsne_combined_real_synthetic(
    real_df,
    synthetic_df
):

    all_sentences = (
        real_df["sentence"].tolist()
        +
        synthetic_df["sentence"].tolist()
    )

    all_embeddings = generate_embeddings(
        all_sentences
    )

    reduced = TSNE(
        n_components=2,
        random_state=SEED
    ).fit_transform(
        all_embeddings
    )

    real_len = len(real_df)

    # padded limits
    x_min, x_max = _expand_limits(
        reduced[:, 0].min(),
        reduced[:, 0].max()
    )

    y_min, y_max = _expand_limits(
        reduced[:, 1].min(),
        reduced[:, 1].max()
    )

    _new_fig()

    ax = plt.gca()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    _prep_ax_equal(ax)

    plt.scatter(
        reduced[:real_len, 0],
        reduced[:real_len, 1],
        c='skyblue',
        alpha=0.4,
        label="Real Data"
    )

    plt.scatter(
        reduced[real_len:, 0],
        reduced[real_len:, 1],
        c='orange',
        alpha=0.6,
        label="Synthetic Data"
    )

    plt.title(
        "t-SNE Projection: "
        "Real vs. Synthetic Data",
        fontsize=TITLE_FONTSIZE
    )

    plt.legend(
        fontsize=LEGEND_FONTSIZE
    )

    plt.grid(True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "tsne_projection_real_vs_synthetic.png"
    )

    plt.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches=SAVE_BBOX,
        pad_inches=SAVE_PAD
    )

    plt.show()

    print(
        f"Saved: {output_path}"
    )


def plot_tsne_per_class_real_synthetic(
    real_df,
    synthetic_df
):

    for label in sorted(
        real_df['label'].unique()
    ):

        real_class = (
            real_df[
                real_df["label"] == label
            ]
            .reset_index(drop=True)
        )

        synthetic_class = (
            synthetic_df[
                synthetic_df["label"] == label
            ]
            .reset_index(drop=True)
        )

        all_sentences = (
            real_class["sentence"].tolist()
            +
            synthetic_class["sentence"].tolist()
        )

        embeddings = generate_embeddings(
            all_sentences
        )

        reduced = TSNE(
            n_components=2,
            random_state=SEED
        ).fit_transform(
            embeddings
        )

        real_len = len(real_class)

        # padded limits
        x_min, x_max = _expand_limits(
            reduced[:, 0].min(),
            reduced[:, 0].max()
        )

        y_min, y_max = _expand_limits(
            reduced[:, 1].min(),
            reduced[:, 1].max()
        )

        _new_fig()

        ax = plt.gca()

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        _prep_ax_equal(ax)

        plt.scatter(
            reduced[:real_len, 0],
            reduced[:real_len, 1],
            c=COLOR_REAL[label],
            alpha=0.4,
            label=f"{LABEL_NAMES[label]} (Real)"
        )

        plt.scatter(
            reduced[real_len:, 0],
            reduced[real_len:, 1],
            c=COLOR_SYN[label],
            alpha=0.7,
            label=f"{LABEL_NAMES[label]} (Synthetic)"
        )

        plt.title(
            f"t-SNE Projection: "
            f"{LABEL_NAMES[label]} – "
            f"Real vs Synthetic",
            fontsize=TITLE_FONTSIZE
        )

        plt.legend(
            fontsize=LEGEND_FONTSIZE
        )

        plt.grid(True)

        fname = os.path.join(
            OUTPUT_DIR,
            f"tsne_projection_real_vs_synthetic_"
            f"{LABEL_NAMES[label].lower()}.png"
        )

        plt.savefig(
            fname,
            dpi=SAVE_DPI,
            bbox_inches=SAVE_BBOX,
            pad_inches=SAVE_PAD
        )

        plt.show()

        print(
            f"Saved: {fname}"
        )


def plot_tsne_per_class_real_synthetic_with_seed_overlay(
    real_df,
    synthetic_df,
    seed_df
):

    for label in sorted(
        real_df['label'].unique()
    ):

        real_class = (
            real_df[
                real_df["label"] == label
            ]
            .reset_index(drop=True)
        )

        synthetic_class = (
            synthetic_df[
                synthetic_df["label"] == label
            ]
            .reset_index(drop=True)
        )

        seed_class = (
            seed_df[
                seed_df["label"] == label
            ]
            .reset_index(drop=True)
        )

        all_sentences = (
            real_class["sentence"].tolist()
            +
            synthetic_class["sentence"].tolist()
        )

        all_embeddings = generate_embeddings(
            all_sentences
        )

        reduced = TSNE(
            n_components=2,
            random_state=SEED
        ).fit_transform(
            all_embeddings
        )

        real_len = len(real_class)
        synth_len = len(synthetic_class)

        # sanity check
        assert (
            reduced.shape[0]
            ==
            real_len + synth_len
        ), (
            "t-SNE rows != "
            "real_len + synth_len"
        )

        # padded limits
        x_min, x_max = _expand_limits(
            reduced[:, 0].min(),
            reduced[:, 0].max()
        )

        y_min, y_max = _expand_limits(
            reduced[:, 1].min(),
            reduced[:, 1].max()
        )

        _new_fig()

        ax = plt.gca()

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        _prep_ax_equal(ax)

        plt.scatter(
            reduced[:real_len, 0],
            reduced[:real_len, 1],
            c=COLOR_REAL[label],
            alpha=0.3,
            label=f"{LABEL_NAMES[label]} (Real)"
        )

        plt.scatter(
            reduced[real_len:, 0],
            reduced[real_len:, 1],
            c=COLOR_SYN[label],
            alpha=0.8,
            label=f"{LABEL_NAMES[label]} (Synthetic)"
        )

        # Overlay seed
        if not seed_class.empty:

            seed_mask = (
                real_class["sentence"]
                .isin(seed_class["sentence"])
            )

            seed_idx = np.flatnonzero(
                seed_mask.values
            )

            if seed_idx.size > 0:

                plt.scatter(
                    reduced[seed_idx, 0],
                    reduced[seed_idx, 1],
                    c=COLOR_SEED_X[label],
                    marker='x',
                    s=90,
                    label=f"{LABEL_NAMES[label]} (Seed)"
                )

        plt.title(
            f"t-SNE Projection: "
            f"{LABEL_NAMES[label]} – "
            f"Real, Synthetic, Seed (Random)",
            fontsize=TITLE_FONTSIZE
        )

        plt.legend(
            fontsize=LEGEND_FONTSIZE
        )

        plt.grid(True)

        fname = os.path.join(
            OUTPUT_DIR,
            f"tsne_projection_real_vs_synthetic_"
            f"{LABEL_NAMES[label].lower()}"
            f"_with_seed_overlay_random.png"
        )

        plt.savefig(
            fname,
            dpi=SAVE_DPI,
            bbox_inches=SAVE_BBOX,
            pad_inches=SAVE_PAD
        )

        plt.show()

        print(
            f"Saved: {fname}"
        )


# === MAIN ===

if __name__ == "__main__":

    args = parse_args()

    DATASET = args.dataset
    LABEL_NAMES = DATASET_LABELS[DATASET]

    # Separate output folder for each dataset
    OUTPUT_DIR = os.path.join(
        "./outputs",
        f"visualization_{DATASET}_random"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print(f"Dataset: {DATASET}")
    print("Method: RANDOM")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # -------------------------
    # Load real data
    # -------------------------
    print("Loading real data...")

    data = load_data()

    # -------------------------
    # Seed
    # -------------------------
    print("Loading random seed JSONL...")

    seed_path = DATASET_FILES[
        DATASET
    ]["seed"]

    seed_df = load_seed_jsonl(
        seed_path
    )

    print(
        f"Loaded seeds: "
        f"{len(seed_df)} from {seed_path}"
    )

    # -------------------------
    # Synthetic
    # -------------------------
    print(
        "Loading synthetic JSONL..."
    )

    syn_path = DATASET_FILES[
        DATASET
    ]["synthetic"]

    synthetic_df = load_synth_jsonl(
        syn_path
    )

    print(
        f"Loaded synthetic: "
        f"{len(synthetic_df)} from {syn_path}"
    )

    # -------------------------
    # Embeddings
    # -------------------------
    print(
        "Embedding all real sentences..."
    )

    full_embeddings = generate_embeddings(
        data["sentence"].tolist()
    )

    # -------------------------
    # Find seeds
    # -------------------------
    seed_mask_global = data[
        "sentence"
    ].isin(
        seed_df["sentence"]
    )

    seed_indices = data.index[
        seed_mask_global
    ].tolist()

    print(
        f"Matched {len(seed_indices)} "
        f"seed indices in real data."
    )

    # -------------------------
    # Visualizations
    # -------------------------
    print(
        "Generating visualizations..."
    )

    reduced, xlim, ylim = (
        plot_tsne_projection(
            data,
            seed_indices,
            full_embeddings
        )
    )

    plot_tsne_projection_per_class(
        data,
        seed_indices,
        reduced,
        xlim,
        ylim
    )

    print(
        "Generating real vs synthetic "
        "t-SNE plot..."
    )

    plot_tsne_combined_real_synthetic(
        data,
        synthetic_df
    )

    print(
        "Generating per-class "
        "real vs synthetic plots..."
    )

    plot_tsne_per_class_real_synthetic(
        data,
        synthetic_df
    )

    print(
        "Generating real/synthetic "
        "plots with random seed overlay..."
    )

    plot_tsne_per_class_real_synthetic_with_seed_overlay(
        data,
        synthetic_df,
        seed_df
    )

    print("Done.")