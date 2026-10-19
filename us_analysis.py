"""
Find all (split, image_id) triplets across image_only / image_english / image_regional
where at least one experiment predicted a US variant. Save individual images and
per-split JSONL files (with English + regional text) to SAVE_DIR.
"""

import io
import json
import math
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EXPERIMENTS = ["image_only", "image_english", "image_regional"]
RESULTS_DIR = Path("results")
DATA_DIR = Path("cammt_data/data")
SAVE_DIR = Path("us_prediction_images")

US_VARIANTS = re.compile(r"^(United States(?: of America)?|USA)$", re.IGNORECASE)
SHORT = {"image_only": "img_only", "image_english": "img_en", "image_regional": "img_reg"}


def is_us(pred: str) -> bool:
    return bool(US_VARIANTS.match(pred.strip())) if pred else False


def load_all_results(results_dir: Path) -> dict:
    data = defaultdict(lambda: defaultdict(dict))
    for exp in EXPERIMENTS:
        exp_dir = results_dir / exp
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
    triplets = []
    for split, ids in data.items():
        for rid, exps in ids.items():
            if len(exps) < 3:
                continue
            if any(is_us(exps[e]["predicted"]) for e in EXPERIMENTS):
                triplets.append((split, rid, exps))
    return triplets


def load_parquet(split: str, columns: list, data_dir: Path) -> dict:
    path = data_dir / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path, columns=columns)
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
    if pred.strip().lower() == gt.strip().lower():
        return "green"
    return "orange"


def save_images(triplets: list, save_dir: Path, data_dir: Path):
    parquet_cache = {}
    for split, rid, exps in triplets:
        if split not in parquet_cache:
            parquet_cache[split] = load_parquet(split, ["ID", "image", "English", "Category"], data_dir)
        row_data = parquet_cache[split].get(rid)
        if row_data is None:
            print(f"  Warning: {rid} not found in {split} parquet")
            continue
        get_pil_image(row_data).save(save_dir / f"{split}__{rid}.jpg")
    return parquet_cache


def save_jsonl(triplets: list, save_dir: Path, data_dir: Path):
    by_split = defaultdict(list)
    for split, rid, exps in triplets:
        by_split[split].append((rid, exps))

    for split, items in sorted(by_split.items()):
        index = load_parquet(split, ["ID", "English", "regional", "Category"], data_dir)
        out_path = save_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for rid, exps in items:
                row = index.get(rid, {})
                record = {
                    "id": rid,
                    "split": split,
                    "ground_truth": exps[EXPERIMENTS[0]]["ground_truth"],
                    "category": row.get("Category", ""),
                    "english": row.get("English", ""),
                    "regional": row.get("regional", ""),
                    "predictions": {
                        exp: {
                            "predicted": exps[exp]["predicted"],
                            "correct": exps[exp]["correct"],
                            "is_us": is_us(exps[exp]["predicted"]),
                            "raw_output": exps[exp]["raw_output"],
                        }
                        for exp in EXPERIMENTS
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {out_path}  ({len(items)} records)")


def save_grid(triplets: list, parquet_cache: dict, save_dir: Path, cols: int = 4):
    n = len(triplets)
    if n == 0:
        return
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 5.5))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (split, rid, exps) in enumerate(triplets):
        ax = axes[idx]
        row_data = parquet_cache.get(split, {}).get(rid)
        if row_data is None:
            ax.set_visible(False)
            continue

        img = get_pil_image(row_data)
        ax.imshow(img)
        ax.axis("off")

        gt = exps[EXPERIMENTS[0]]["ground_truth"]
        category = row_data.get("Category", "")
        english = (row_data.get("English", "") or "")[:70]

        ax.set_title(f"GT: {gt} | {category}\n{english}", fontsize=6.5, pad=3)

        y = -0.03
        for exp in EXPERIMENTS:
            rec = exps.get(exp)
            pred = rec["predicted"] if rec else "N/A"
            marker = " ★" if is_us(pred) else ""
            ax.text(0.5, y, f"[{SHORT[exp]}] {pred}{marker}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=6, color=pred_color(pred, gt), clip_on=False)
            y -= 0.07

    for ax in axes[n:]:
        ax.set_visible(False)

    legend_elements = [
        mpatches.Patch(color="red",    label="US prediction (★)"),
        mpatches.Patch(color="green",  label="Correct prediction"),
        mpatches.Patch(color="orange", label="Wrong (non-US) prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.suptitle(f"Triplets with ≥1 US prediction  (n={n})", fontsize=12, fontweight="bold")
    plt.tight_layout()

    grid_path = save_dir / "grid_overview.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=150)
    print(f"Grid saved to {grid_path}")
    plt.close(fig)


def print_summary(triplets: list):
    print(f"\nTotal triplets with ≥1 US prediction: {len(triplets)}")

    split_counts = Counter(t[0] for t in triplets)
    print("\nBreakdown by split:")
    for split, count in sorted(split_counts.items(), key=lambda x: -x[1]):
        print(f"  {split:20s}: {count}")

    exp_counts = Counter()
    for _, _, exps in triplets:
        for exp in EXPERIMENTS:
            if is_us(exps[exp]["predicted"]):
                exp_counts[exp] += 1
    print("\nUS predictions per experiment:")
    for exp in EXPERIMENTS:
        print(f"  {exp:20s}: {exp_counts.get(exp, 0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--data-dir",    default=str(DATA_DIR))
    parser.add_argument("--save-dir",    default=str(SAVE_DIR))
    parser.add_argument("--cols",        type=int, default=4)
    parser.add_argument("--split",       default=None, help="Filter to one split")
    parser.add_argument("--no-grid",     action="store_true", help="Skip grid visualization")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data_dir    = Path(args.data_dir)
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all_results(results_dir)

    print("Finding US triplets...")
    triplets = find_us_triplets(data)

    if args.split:
        triplets = [t for t in triplets if t[0] == args.split]
        print(f"Filtered to '{args.split}'")

    print_summary(triplets)

    print(f"\nSaving images to {save_dir}/...")
    parquet_cache = save_images(triplets, save_dir, data_dir)
    print(f"  {len(triplets)} images saved.")

    print(f"\nWriting per-split JSONL files to {save_dir}/...")
    save_jsonl(triplets, save_dir, data_dir)

    if not args.no_grid:
        print("\nRendering grid...")
        save_grid(triplets, parquet_cache, save_dir, cols=args.cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
