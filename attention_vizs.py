"""
Attention visualization for US-predicted examples using Qwen2.5-VL + Visualizer.

This version uses the Visualizer notebook-style functions directly
(grid-to-grid and head visualization), instead of a custom rollout overlay.

Run:
    python attention_vizs.py [--max-examples N] [--split SPLIT] [--out-dir PATH]
"""

# ── MUST be first — activates bytecode instrumentation before model import ──
from visualizer import get_local
get_local.activate()
# ─────────────────────────────────────────────────────────────────────────────

import io
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


JSONL_DIR   = Path("us_prediction_images")
DATA_DIR    = Path("cammt_data/data")
MODEL_PATH  = "Qwen/Qwen2.5-VL-7B-Instruct"
EXPERIMENTS = ["image_only", "image_english", "image_regional"]

_parquet_img_cache = {}


# ── Prompts ──────────────────────────────────────────────────────────────────
def make_prompt(exp_name: str, ex: dict) -> str:
    regional = ex.get("regional") or ""
    english = ex.get("english") or ""

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


# ── Visualizer hook: capture attention weights ───────────────────────────────
@get_local("attn_weights")
def _vision_eager_attn(module, q, k, v, scaling):
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling
    attn_weights = torch.softmax(scores.float(), dim=-1)
    attn_output = torch.matmul(attn_weights.to(q.dtype), v).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def patch_vision_attention():
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
        query_states, key_states = M.apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        # (seq, heads, dim) -> (1, heads, seq, dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states   = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks = torch.split(query_states, lengths, dim=2)
        k_chunks = torch.split(key_states, lengths, dim=2)
        v_chunks = torch.split(value_states, lengths, dim=2)

        outs = []
        for q, k, v in zip(q_chunks, k_chunks, v_chunks):
            out, _ = _vision_eager_attn(self, q, k, v, self.scaling)
            outs.append(out)

        attn_output = torch.cat(outs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    M.Qwen2_5_VLVisionAttention.forward = patched_forward
    print("Vision attention patched with Visualizer capture.")


# ── Notebook functions from Visualizer example ───────────────────────────────
def grid_show(to_shows, cols):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    axs = np.array(axs, dtype=object)
    if axs.ndim == 0:
        axs = axs.reshape(1, 1)
    elif axs.ndim == 1:
        if rows == 1:
            axs = axs.reshape(1, -1)
        else:
            axs = axs.reshape(-1, 1)

    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = "pad"
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    return fig


def visualize_head(att_map):
    fig, ax = plt.subplots()
    im = ax.imshow(att_map)
    ax.figure.colorbar(im, ax=ax)
    return fig


def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f"Head {i}"))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, "Head Average"))
    return grid_show(to_shows, cols=cols)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)
    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), "CLS", fill=(0, 0, 0))

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)

    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0:delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline="red", width=2)
    return image


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize(image.size)

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis("off")

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap="rainbow")
    ax[1].imshow(meta_mask)
    ax[1].axis("off")
    return fig


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize(image.size)

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis("off")

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / (np.max(mask) + 1e-8), alpha=alpha, cmap="rainbow")
    ax[1].axis("off")
    return fig


# ── Helpers to render notebook figs into arrays ───────────────────────────────
def fig_to_rgb_array(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    arr = np.array(buf)[..., :3].copy()
    plt.close(fig)
    return arr

def save_fig(fig, path: Path, dpi=140):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


# ── Visualization helpers ─────────────────────────────────────────────────────
def grid_show(to_shows, cols):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = np.array(axs, dtype=object)

    if axs.ndim == 0:
        axs = axs.reshape(1, 1)
    elif axs.ndim == 1:
        if rows == 1:
            axs = axs.reshape(1, -1)
        else:
            axs = axs.reshape(-1, 1)

    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = "pad"

            img = np.asarray(image, dtype=np.float32)

            # Contrast normalization (CRITICAL)
            vmin = np.percentile(img, 1)
            vmax = np.percentile(img, 99.5)
            if vmax <= vmin:
                vmin = float(img.min())
                vmax = float(img.max()) if float(img.max()) > float(img.min()) else float(img.min()) + 1e-8

            axs[i, j].imshow(img, vmin=vmin, vmax=vmax, interpolation="nearest")
            axs[i, j].set_title(title, fontsize=8)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])

    return fig


def visualize_head(att_map, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    x = np.asarray(att_map, dtype=np.float32)

    vmin = np.percentile(x, 1)
    vmax = np.percentile(x, 99.5)
    if vmax <= vmin:
        vmin = float(x.min())
        vmax = float(x.max()) if float(x.max()) > float(x.min()) else float(x.min()) + 1e-8

    im = ax.imshow(x, vmin=vmin, vmax=vmax, interpolation="nearest")

    if title:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig


def visualize_heads(att_map, cols=4):
    to_shows = []
    att_map = att_map.squeeze()

    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f"Head {i}"))

    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, "Head Average"))

    return grid_show(to_shows, cols=cols)


# ── Save all attention heads ──────────────────────────────────────────────────
def save_attention_heads(att_heads: np.ndarray, base_dir: Path, exp: str):
    """
    Saves:
      base_dir/
        {exp}_heads.png
        attention_heads/
          {exp}_head_00.png
          ...
          {exp}_head_average.png
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Overview grid
    heads_overview_path = base_dir / f"{exp}_heads.png"
    fig_heads = visualize_heads(att_heads, cols=4)
    save_fig(fig_heads, heads_overview_path)
    print(f"      saved heads overview -> {heads_overview_path}")

    # Individual heads
    heads_dir = base_dir / "attention_heads"
    heads_dir.mkdir(parents=True, exist_ok=True)

    for h in range(att_heads.shape[0]):
        head_path = heads_dir / f"{exp}_head_{h:02d}.png"
        fig = visualize_head(att_heads[h], title=f"{exp} | Head {h}")
        save_fig(fig, head_path)
        print(f"      saved attention head  -> {head_path}")

    # Average head
    avg_path = heads_dir / f"{exp}_head_average.png"
    fig_avg = visualize_head(att_heads.mean(axis=0), title=f"{exp} | Head Average")
    save_fig(fig_avg, avg_path)
    print(f"      saved attention avg   -> {avg_path}")


def save_fig(fig, path: Path, dpi=140):
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


# ── Data loading ──────────────────────────────────────────────────────────────
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


# ── Attention helpers ─────────────────────────────────────────────────────────
def infer_and_capture(model, processor, image: Image.Image, prompt: str):
    content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        raise RuntimeError("image_grid_thw missing from processor output")

    _, gh, gw = [int(x) for x in grid_thw[0].tolist()]

    get_local.clear()
    with torch.no_grad():
        _ = model(**inputs)

    raw = list(get_local.cache.get("_vision_eager_attn", []))
    if not raw:
        raise RuntimeError("No attention tensors captured under key '_vision_eager_attn'")

    attn_list = []
    for x in raw:
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)
        if x.ndim == 4 and x.shape[-1] == x.shape[-2]:
            attn_list.append(x)

    if not attn_list:
        raise RuntimeError("No valid 4D attention tensors found")

    return attn_list, gh, gw


def factor_grid(n: int):
    """
    Infer a 2D grid from seq length for local/window attention visualization.
    Chooses factor pair closest to square.
    """
    best = None
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            cand = (h, w)
            if best is None or abs(w - h) < abs(best[1] - best[0]):
                best = cand
    return best if best is not None else (1, n)


def choose_attention_chunk(attn_list: list):
    """
    Qwen often yields window/local attention chunks with varying seq lengths.
    Pick the largest square attention tensor.
    Returns:
        att: (heads, seq, seq)
    """
    best = max(attn_list, key=lambda x: x.shape[-1])
    return best[0]


def choose_query_index(att_heads: np.ndarray):
    """
    att_heads: (heads, seq, seq)
    Choose a query patch with strongest mean outgoing attention.
    """
    mat = att_heads.mean(axis=0)
    q_idx = int(np.argmax(mat.mean(axis=1)))
    return q_idx


def render_visualizer_panel(image: Image.Image, att_heads: np.ndarray, mode="grid"):
    """
    Uses the notebook-style Visualizer functions and returns an RGB image array.
    mode:
      - "grid": grid-to-grid attention from a selected query patch
      - "heads": per-head attention grid
    """
    seq_len = att_heads.shape[-1]
    local_grid = factor_grid(seq_len)

    if mode == "heads":
        fig = visualize_heads(att_heads, cols=4)
        return fig_to_rgb_array(fig), local_grid, None

    q_idx = choose_query_index(att_heads)
    head_idx = 0

    fig = visualize_grid_to_grid(
        att_heads[head_idx],
        q_idx,
        image,
        grid_size=local_grid,
        alpha=0.65,
    )
    return fig_to_rgb_array(fig), local_grid, q_idx


def pred_color(pred: str, gt: str, is_us_flag: bool) -> str:
    if is_us_flag:
        return "red"
    if pred.strip().lower() == gt.strip().lower():
        return "green"
    return "orange"

# ── Summary grid ───────────────────────────────────────────────────────────── def save_summary_grid(saved_paths: list, out_dir: Path, cols: int = 3): valid = [p for p in saved_paths if p and p.exists()] if not valid: return rows = math.ceil(len(valid) / cols) fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5)) axes = np.array(axes, dtype=object) if axes.ndim == 0: axes = axes.reshape(1) else: axes = axes.reshape(-1) for idx, path in enumerate(valid): axes[idx].imshow(plt.imread(str(path))) axes[idx].axis("off") axes[idx].set_title(path.stem.replace("__", "\n"), fontsize=5) for ax in axes[len(valid):]: ax.set_visible(False) fig.suptitle( f"Visualizer grid-to-grid attention — all US-prediction cases (n={len(valid)})", fontsize=9, fontweight="bold", ) plt.tight_layout() grid_path = out_dir / "summary_grid.png" plt.savefig(grid_path, bbox_inches="tight", dpi=100) plt.close(fig) print(f"Summary grid -> {grid_path}")

def visualize_example(ex: dict, model, processor, out_dir: Path, data_dir: Path = DATA_DIR):
    rid = ex["id"]
    split = ex["split"]
    gt = ex["ground_truth"]

    try:
        image = load_image(split, rid, data_dir)
    except Exception as e:
        print(f"    skip {rid}: {e}")
        return None

    img_np = np.array(image)

    ncols = len(EXPERIMENTS) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 6.5, 6))

    sample_dir = out_dir / f"{split}__{rid}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    original_path = sample_dir / "original.jpg"
    Image.fromarray(img_np).save(original_path)
    print(f"      saved original -> {original_path}")

    english = (ex.get("english") or "")[:60]
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title(
        f"Original\nGT: {gt} | split: {split}\n{english}",
        fontsize=8,
    )

    prompt_labels = {
        "image_only": "Prompt: image only",
        "image_english": "Prompt: image + English text",
        "image_regional": "Prompt: image + regional text",
    }

    for col, exp in enumerate(EXPERIMENTS, start=1):
        ax = axes[col]
        pred_info = ex["predictions"].get(exp, {})
        pred = pred_info.get("predicted", "N/A")
        is_us_flag = pred_info.get("is_us", False)
        color = pred_color(pred, gt, is_us_flag)
        star = " ★" if is_us_flag else ""

        prompt = make_prompt(exp, ex)

        try:
            attn_list, gh, gw = infer_and_capture(model, processor, image, prompt)
            att_heads = choose_attention_chunk(attn_list)

            panel_img, local_grid, q_idx = render_visualizer_panel(
                image=image,
                att_heads=att_heads,
                mode="grid",
            )
            ax.imshow(panel_img)
            ax.set_title(
                f"{prompt_labels[exp]}\nPred: {pred}{star}\nchunk={att_heads.shape[-1]} grid={local_grid[0]}x{local_grid[1]} q={q_idx}",
                fontsize=8,
                color=color,
            )

            exp_dir = sample_dir / exp
            exp_dir.mkdir(parents=True, exist_ok=True)

            grid_path = exp_dir / f"{exp}_grid_to_grid.png"
            fig_grid = visualize_grid_to_grid(
                att_heads[0],
                q_idx,
                image,
                grid_size=local_grid,
                alpha=0.65,
            )
            save_fig(fig_grid, grid_path)
            print(f"      saved grid            -> {grid_path}")

            save_attention_heads(att_heads, exp_dir, exp)

        except Exception as e:
            print(f"      failed {exp} -> {sample_dir}: {e}")
            ax.imshow(img_np)
            ax.set_title(
                f"{prompt_labels[exp]}\nPred: {pred}{star}\nERR: {str(e)[:55]}",
                fontsize=8,
                color=color,
            )

        ax.axis("off")

    fig.suptitle(f"{split} | {rid}", fontsize=10, y=1.03)
    plt.tight_layout()

    comparison_path = sample_dir / "comparison.png"
    plt.savefig(comparison_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"      saved compare -> {comparison_path}")

    return comparison_path


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", default=str(JSONL_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--out-dir", default="attention_vizls")
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_vision_attention()

    print(f"Loading model from {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=512 * 512,
    )
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