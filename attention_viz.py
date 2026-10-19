"""
Attention visualization for US-predicted examples using Qwen2.5-VL + Visualizer.

Install deps first (if not already):
    pip install bytecode
    pip install git+https://github.com/luo3300612/Visualizer

Run from cammt_vl/:
    python attention_viz.py [--max-examples N] [--split SPLIT] [--out-dir PATH]

Outputs (in --out-dir, default /l/users/kapil.kavitha/attention_viz/):
    {split}__{id}.png   — per-example side-by-side heatmaps across 3 experiments
    summary_grid.png    — overview of all saved per-example plots
"""

# ── MUST be first — activates bytecode instrumentation before any model import ─
from visualizer import get_local
get_local.activate()
# ──────────────────────────────────────────────────────────────────────────────

import io, json, math, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Paths / constants ──────────────────────────────────────────────────────────
JSONL_DIR   = Path("us_prediction_images")
DATA_DIR    = Path("cammt_data/data")
MODEL_PATH  = "Qwen/Qwen2.5-VL-7B-Instruct"
EXPERIMENTS = ["image_only", "image_english", "image_regional"]

# ── Prompts (mirrored from cammt_experiments.py) ───────────────────────────────
def make_prompt(exp_name: str, ex: dict) -> str:
    regional = ex.get("regional") or ""
    english  = ex.get("english")  or ""
    if exp_name == "image_only":
        return (
            "Look at this image carefully. "
            "Based on ALL available visual clues (architecture, vegetation, signage, "
            "vehicles, clothing, environment, text visible in the image, etc.), "
            "predict which country this image was most likely taken in. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name. No explanation. /no_think'
        )
    elif exp_name == "image_english":
        return (
            "Look at this image and read the associated text carefully. "
            f'Text: "{english}" '
            "Based on ALL available clues, predict which country this image was most likely taken in. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name. No explanation. /no_think'
        )
    elif exp_name == "image_regional":
        return (
            "Look at this image and read the associated text carefully. "
            "The text is written in the local language. "
            f'Text: "{regional}" '
            "Based on ALL available clues, predict which country this image was most likely taken in. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name. No explanation. /no_think'
        )
    raise ValueError(f"Unknown experiment: {exp_name}")


# ── Visualizer hook: manual vision attention that exposes attn_weights ─────────

@get_local('attn_weights')
def _vision_eager_attn(module, q, k, v, scaling):
    """
    Drop-in replacement for eager_attention_forward in the vision encoder.
    Shapes: q/k/v are (1, num_heads, chunk_seq, head_dim).
    get_local captures attn_weights as (1, num_heads, seq, seq) numpy array.
    attn_weights is kept as float32 so numpy conversion succeeds.
    """
    scores      = torch.matmul(q, k.transpose(2, 3)) * scaling
    attn_weights = torch.softmax(scores.float(), dim=-1)           # float32 — captured by get_local
    attn_output  = torch.matmul(attn_weights.to(q.dtype), v).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def patch_vision_attention():
    """
    Replace Qwen2_5_VLVisionAttention.forward with a version that calls
    _vision_eager_attn so get_local can capture the attention weights.
    Must be called AFTER get_local.activate() but BEFORE loading the model.
    """
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = M.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # (heads, seq, head_dim) → (1, heads, seq, head_dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states   = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Split into per-image chunks and run attention on each
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks = torch.split(query_states, lengths, dim=2)
        k_chunks = torch.split(key_states,   lengths, dim=2)
        v_chunks = torch.split(value_states, lengths, dim=2)

        chunk_outs = [
            _vision_eager_attn(self, q, k, v, self.scaling)[0]
            for q, k, v in zip(q_chunks, k_chunks, v_chunks)
        ]
        attn_output = torch.cat(chunk_outs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    M.Qwen2_5_VLVisionAttention.forward = patched_forward
    print("Vision attention patched (eager, get_local enabled).")


# ── Data loading ───────────────────────────────────────────────────────────────

_parquet_img_cache: dict = {}

def load_image(split: str, rid: str, data_dir: Path = DATA_DIR) -> Image.Image:
    if split not in _parquet_img_cache:
        path = data_dir / f"{split}-00000-of-00001.parquet"
        df = pd.read_parquet(path, columns=["ID", "image"])
        _parquet_img_cache[split] = df.set_index("ID").to_dict("index")
    row = _parquet_img_cache[split].get(rid)
    if row is None:
        raise KeyError(f"{rid} not found in {split}")
    img_field = row["image"]
    if isinstance(img_field, dict):
        raw = img_field.get("bytes") or img_field.get("path")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(img_field, bytes):
        return Image.open(io.BytesIO(img_field)).convert("RGB")
    raise ValueError(f"Unexpected image format for {rid}")


def load_examples(jsonl_dir: Path, split_filter=None, max_n=None):
    examples = []
    for jf in sorted(jsonl_dir.glob("*.jsonl")):
        if split_filter and jf.stem != split_filter:
            continue
        with open(jf) as f:
            for line in f:
                examples.append(json.loads(line))
    return examples[:max_n] if max_n else examples


# ── Attention helpers ──────────────────────────────────────────────────────────

def attention_rollout(attn_list: list) -> np.ndarray:
    """
    Compute attention rollout across layers.
    attn_list: list of (1, heads, seq, seq) numpy arrays — one per vision layer.
    Returns: (seq, seq) float32 array.
    """
    n = attn_list[0].shape[-1]
    rollout = np.eye(n, dtype=np.float32)
    for a in attn_list:
        avg = a[0].mean(axis=0).astype(np.float32)   # (seq, seq)
        avg = avg + np.eye(n, dtype=np.float32)
        avg = avg / (avg.sum(axis=-1, keepdims=True) + 1e-8)
        rollout = avg @ rollout
    return rollout


def patches_to_heatmap(patch_vec: np.ndarray, gh: int, gw: int, img_h: int, img_w: int) -> np.ndarray:
    """Resize a (gh*gw,) patch attention vector to image dimensions."""
    p = patch_vec[: gh * gw].reshape(gh, gw).astype(np.float32)
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    hm = np.array(
        Image.fromarray((p * 255).astype(np.uint8)).resize(
            (img_w, img_h), resample=Image.BILINEAR
        )
    ) / 255.0
    return hm


def overlay(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    colored = cm.inferno(heatmap)[..., :3]
    return (alpha * colored + (1 - alpha) * img_np.astype(np.float32) / 255.0).clip(0, 1)


# ── Inference + capture ────────────────────────────────────────────────────────

def infer_and_capture(model, processor, image: Image.Image, prompt: str):
    """
    Run one forward pass for (image, prompt) and return captured vision attention maps.
    Returns:
        attn_list : list of (1, num_heads, n_patches, n_patches) numpy arrays per layer
        gh, gw    : patch grid height and width
    """
    content  = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs   = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        return [], 0, 0
    _, gh, gw = [int(x) for x in grid_thw[0].tolist()]

    get_local.clear()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=30)

    # get_local.cache['attn_weights'] contains one entry per captured call
    # (one per vision layer × one chunk per image)
    attn_list = list(get_local.cache.get("attn_weights", []))
    return attn_list, gh, gw


# ── Per-example visualization ──────────────────────────────────────────────────

def pred_color(pred: str, gt: str, is_us: bool) -> str:
    if is_us:
        return "red"
    if pred.strip().lower() == gt.strip().lower():
        return "green"
    return "orange"


def visualize_example(ex: dict, model, processor, out_dir: Path, data_dir: Path = DATA_DIR):
    rid, split, gt = ex["id"], ex["split"], ex["ground_truth"]

    try:
        image = load_image(split, rid, data_dir)
    except Exception as e:
        print(f"    skip {rid}: {e}")
        return None

    img_np = np.array(image)
    ih, iw = img_np.shape[:2]

    ncols = len(EXPERIMENTS) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4.5, 4.8))

    # Column 0: original image
    axes[0].imshow(img_np)
    axes[0].axis("off")
    english = (ex.get("english") or "")[:60]
    axes[0].set_title(f"GT: {gt} | {split}\n{english}", fontsize=7)

    for col, exp in enumerate(EXPERIMENTS, start=1):
        ax   = axes[col]
        pred_info = ex["predictions"].get(exp, {})
        pred      = pred_info.get("predicted", "N/A")
        is_us_    = pred_info.get("is_us", False)
        color     = pred_color(pred, gt, is_us_)
        star      = " ★" if is_us_ else ""

        prompt    = make_prompt(exp, ex)
        attn_list, gh, gw = infer_and_capture(model, processor, image, prompt)

        if attn_list and gh > 0:
            rollout    = attention_rollout(attn_list)     # (seq, seq)
            patch_attn = rollout.mean(axis=0)             # (seq,) — mean attention per patch
            heatmap    = patches_to_heatmap(patch_attn, gh, gw, ih, iw)
            ax.imshow(overlay(img_np, heatmap))
        else:
            ax.imshow(img_np)

        ax.axis("off")
        ax.set_title(f"{exp}\n{pred}{star}", fontsize=7, color=color)

    fig.suptitle(f"{split} | {rid}", fontsize=7, y=1.01)
    plt.tight_layout()
    out_path = out_dir / f"{split}__{rid}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"    saved {out_path.name}")
    return out_path


# ── Summary grid ───────────────────────────────────────────────────────────────

def save_summary_grid(saved_paths: list, out_dir: Path, cols: int = 3):
    valid = [p for p in saved_paths if p and p.exists()]
    if not valid:
        return
    rows = math.ceil(len(valid) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = axes.flatten() if len(valid) > 1 else [axes]
    for idx, path in enumerate(valid):
        axes[idx].imshow(plt.imread(str(path)))
        axes[idx].axis("off")
        axes[idx].set_title(path.stem.replace("__", "\n"), fontsize=5)
    for ax in axes[len(valid):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Attention maps — all US-prediction cases (n={len(valid)})\n"
        "Red=US pred ★  Green=correct  Orange=wrong non-US",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout()
    grid_path = out_dir / "summary_grid.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"Summary grid → {grid_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir",    default=str(JSONL_DIR))
    parser.add_argument("--data-dir",     default=str(DATA_DIR))
    parser.add_argument("--model-path",   default=MODEL_PATH)
    parser.add_argument("--out-dir",      default="/l/users/kapil.kavitha/attention_viz")
    parser.add_argument("--split",        default=None, help="Filter to one split")
    parser.add_argument("--max-examples", type=int,  default=None)
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch BEFORE loading model so the class method is replaced at init time
    patch_vision_attention()

    print(f"Loading model from {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",    # eager keeps our patched forward active
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512 * 512)
    processor.tokenizer.padding_side = "left"
    print("Model ready.\n")

    data_dir = Path(args.data_dir)
    examples = load_examples(Path(args.jsonl_dir), args.split, args.max_examples)
    print(f"Processing {len(examples)} examples...\n")

    saved_paths = []
    for i, ex in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] {ex['split']} | {ex['id']}")
        path = visualize_example(ex, model, processor, out_dir, data_dir)
        saved_paths.append(path)

    save_summary_grid(saved_paths, out_dir)
    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
