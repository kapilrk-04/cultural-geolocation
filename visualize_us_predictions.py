"""
For each (split, image_id) triplet across image_only / image_english / image_regional,
find cases where at least one experiment predicted a US variant and visualize + save them.
"""

import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

EXPERIMENTS = ["image_only", "image_english", "image_regional"]
RESULTS_DIR = Path("results")
DATA_DIR = Path("cammt_data/data")
SAVE_DIR = Path("us_prediction_images")

US_VARIANTS = re.compile(
    r"^(United States(?: of America)?|USA)$", re.IGNORECASE
)


def is_us(pred: str) -> bool:
    return bool(US_VARIANTS.match(pred.strip())) if pred else False


def load_all_results() -> dict:
    """
    Returns: {split: {id: {experiment: record}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    for exp in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp
        if not exp_dir.exists():
            print(f"Warning: {exp_dir} not found, skipping")
            continue
        for jsonl_path in sorted(exp_dir.glob("*.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    r = json.loads(line)
                    data[r["split"]][r["id"]][exp] = r
    return data


def find_us_triplets(data: dict) -> list:
    """
    Return list of (split, id, triplet_dict) where at least one experiment
    predicted a US variant. Only include IDs present in all 3 experiments.
    """
    triplets = []
    for split, ids in data.items():
        for rid, exps in ids.items():
            if len(exps) < 3:
                continue
            preds = {e: exps[e]["predicted"] for e in EXPERIMENTS}
            if any(is_us(p) for p in preds.values()):
                triplets.append((split, rid, exps))
    return triplets


def load_parquet_index(split: str) -> dict:
    parquet_path = DATA_DIR / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path, columns=["ID", "image", "English", "Category"])
    return df.set_index("ID").to_dict("index")


def get_pil_image(row_data: dict) -> Image.Image:
    img_field = row_data["image"]
    if isinstance(img_field, dict):
        raw = img_field.get("bytes") or img_field.get("path")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(img_field, bytes):
        return Image.open(io.BytesIO(img_field)).convert("RGB")
    raise ValueError(f"Unexpected image format: {type(img_field)}")


def pred_color(pred: str, gt: str) -> str:
    if is_us(pred):
        return "red"
    if pred == gt:
        return "green"
    return "orange"


def make_triplet_label(exp: str, pred: str, gt: str) -> str:
    short = {"image_only": "img_only", "image_english": "img_en", "image_regional": "img_reg"}
    marker = " ★" if is_us(pred) else ""
    return f"[{short[exp]}] {pred}{marker}"


def visualize_and_save(triplets: list, save_dir: Path, cols: int = 4):
    """
    One subplot per triplet. Each subplot shows the image with a 3-line title
    (one line per experiment). Images are also saved individually.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(triplets)
    if n == 0:
        print("No triplets found.")
        return

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 5.5))
    axes = axes.flatten() if n > 1 else [axes]

    parquet_cache: dict = {}

    for idx, (split, rid, exps) in enumerate(triplets):
        ax = axes[idx]

        if split not in parquet_cache:
            parquet_cache[split] = load_parquet_index(split)

        row_data = parquet_cache[split].get(rid)
        if row_data is None:
            ax.set_visible(False)
            continue

        img = get_pil_image(row_data)

        # Save individual image
        img_save_path = save_dir / f"{split}__{rid}.jpg"
        img.save(img_save_path)

        ax.imshow(img)
        ax.axis("off")

        gt = exps[EXPERIMENTS[0]]["ground_truth"]
        category = row_data.get("Category", "")
        english = row_data.get("English", "")
        if len(english) > 70:
            english = english[:67] + "..."

        # Header line
        ax.set_title(
            f"GT: {gt} | {category}\n{english}",
            fontsize=6.5, pad=3, color="black"
        )

        # Annotate predictions as coloured text below image
        lines = []
        colors = []
        for exp in EXPERIMENTS:
            rec = exps.get(exp)
            pred = rec["predicted"] if rec else "N/A"
            lines.append(make_triplet_label(exp, pred, gt))
            colors.append(pred_color(pred, gt))

        y_start = -0.03
        for line, color in zip(lines, colors):
            ax.text(
                0.5, y_start, line,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=6, color=color,
                clip_on=False
            )
            y_start -= 0.07

    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="red", label="US prediction (★)"),
        Patch(color="green", label="Correct prediction"),
        Patch(color="orange", label="Wrong (non-US) prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.suptitle(
        f"Triplets with ≥1 US prediction  (n={n})  —  images also saved to {save_dir}/",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    grid_path = save_dir / "grid_overview.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=150)
    print(f"Grid saved to {grid_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--split", type=str, default=None, help="Filter to one split")
    parser.add_argument("--save-dir", type=str, default=str(SAVE_DIR))
    args = parser.parse_args()

    print("Loading results...")
    data = load_all_results()

    print("Finding triplets with US prediction...")
    triplets = find_us_triplets(data)
    print(f"Total triplets: {len(triplets)}")

    # Summary
    split_counts = Counter(t[0] for t in triplets)
    print("\nBreakdown by split:")
    for split, count in sorted(split_counts.items(), key=lambda x: -x[1]):
        print(f"  {split:20s}: {count}")

    exp_counts = Counter()
    for _, _, exps in triplets:
        for exp in EXPERIMENTS:
            rec = exps.get(exp)
            if rec and is_us(rec["predicted"]):
                exp_counts[exp] += 1
    print("\nUS predictions per experiment:")
    for exp, count in exp_counts.items():
        print(f"  {exp:20s}: {count}")

    if args.split:
        triplets = [t for t in triplets if t[0] == args.split]
        print(f"\nFiltered to '{args.split}': {len(triplets)} triplets")

    visualize_and_save(triplets, Path(args.save_dir), cols=args.cols)
    print(f"\nIndividual images saved to {args.save_dir}/")
