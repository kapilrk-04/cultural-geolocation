#!/usr/bin/env python3
"""
Step-4 analysis script for Qwen2.5-VL.

What it does:
- supports:
    * image_only
    * image_english
    * image_regional
    * textonly_english
    * textonly_regional
- uses the experiment prompts directly
- loads examples from JSONL
- loads image from parquet for image-based experiments
- runs generation with:
    * output_scores=True
    * output_attentions=True
- saves per experiment:
    * probability trace plot
    * debug json with:
        - generated tokens
        - actual generated token probabilities
        - candidate probabilities per step
        - top tokens per step
    * step-4 logit info json
    * step-4 attention-to-image visualization (image-based exps only)
- saves per example:
    * step4_us_vs_ethiopia_comparison.png
    * step4_us_vs_ethiopia_comparison.json
    * step4_logit_margin_comparison.png
    * step4_logit_margin_comparison.json

Run:
    python qwen_step4_analysis.py \
        --jsonl-dir us_prediction_images \
        --data-dir cammt_data/data \
        --out-dir step4_analysis_outputs \
        --split amh_eth \
        --max-examples 1
"""

import io
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

EXPERIMENTS = [
    "image_only",
    "image_english",
    "image_regional",
    "textonly_english",
    "textonly_regional",
]

_parquet_img_cache = {}


# ── prompts ───────────────────────────────────────────────────────────────────
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

    elif exp_name == "textonly_regional":
        return (
            "Read the following text carefully. "
            "It is written in a local language and describes something culturally specific. "
            "Based ONLY on linguistic and cultural clues in the text, "
            "predict which country this text most likely originates from. "
            f'Text: "{regional}" '
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
            "No explanation, no extra text. /no_think"
        )

    elif exp_name == "textonly_english":
        return (
            "Read the following text carefully. "
            "Based on cultural and contextual clues in the text, "
            "predict which country this text most likely originates from. "
            f'Text: "{english}" '
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
            "No explanation, no extra text. /no_think"
        )

    raise ValueError(f"Unknown experiment: {exp_name}")


def experiment_uses_image(exp_name: str) -> bool:
    return exp_name in {"image_only", "image_english", "image_regional"}


# ── data loading ──────────────────────────────────────────────────────────────
def load_examples(jsonl_dir: Path, split_filter=None, max_n=None):
    examples = []
    for jf in sorted(jsonl_dir.glob("*.jsonl")):
        if split_filter and jf.stem != split_filter:
            continue
        with open(jf) as f:
            for line in f:
                examples.append(json.loads(line))
    return examples[:max_n] if max_n else examples


def load_image(split: str, rid: str, data_dir: Path) -> Image.Image:
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

    raise ValueError(f"Unexpected image format for {rid}: {type(img_field)}")


# ── input building ────────────────────────────────────────────────────────────
def build_inputs(processor, image, prompt: str, device, use_image: bool):
    if use_image:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(device)
    else:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            return_tensors="pt",
        ).to(device)

    return messages, inputs


# ── generation ────────────────────────────────────────────────────────────────
def generate_with_scores(model, inputs, max_new_tokens=20):
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False,
        do_sample=False,
    )


def generate_with_scores_and_attentions(model, inputs, max_new_tokens=20):
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
        do_sample=False,
    )


def decode_generated_tokens(processor, generation_output, prompt_len):
    seq = generation_output.sequences[0]
    gen_ids = seq[prompt_len:]
    gen_texts = [
        processor.tokenizer.decode([int(t)], skip_special_tokens=False)
        for t in gen_ids
    ]
    return gen_ids.tolist(), gen_texts


# ── probability trace helpers ─────────────────────────────────────────────────
def tokenize_candidate_first_token(tokenizer, text: str):
    variants = [text, " " + text]
    out = []
    seen = set()

    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if ids:
            tok_id = ids[0]
            if tok_id not in seen:
                out.append(tok_id)
                seen.add(tok_id)

    if not out:
        raise ValueError(f"Could not tokenize candidate text: {text!r}")
    return out


def get_top_tokens_per_step(processor, generation_output, top_k=10):
    tokenizer = processor.tokenizer
    scores = generation_output.scores
    sequences = generation_output.sequences[0]
    num_steps = len(scores)
    generated_ids = sequences[-num_steps:].tolist()

    step_info = []
    for step_idx, step_scores in enumerate(scores):
        probs = F.softmax(step_scores[0].float(), dim=-1)

        actual_id = generated_ids[step_idx]
        actual_prob = float(probs[actual_id].item())
        actual_text = tokenizer.decode([actual_id], skip_special_tokens=False)

        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[0]))
        top_tokens = []
        for tok_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
            top_tokens.append({
                "token_id": int(tok_id),
                "token_text": tokenizer.decode([int(tok_id)], skip_special_tokens=False),
                "probability": float(prob),
            })

        step_info.append({
            "step": int(step_idx),
            "actual_generated_token_id": int(actual_id),
            "actual_generated_token_text": actual_text,
            "actual_generated_token_probability": actual_prob,
            "top_tokens": top_tokens,
        })

    return step_info


def trace_candidate_first_token_probs(processor, generation_output, candidates):
    tokenizer = processor.tokenizer
    scores = generation_output.scores
    sequences = generation_output.sequences[0]
    num_steps = len(scores)

    generated_ids = sequences[-num_steps:].tolist()

    candidate_token_ids = {
        cand: tokenize_candidate_first_token(tokenizer, cand)
        for cand in candidates
    }

    trace = {cand: [] for cand in candidates}
    actual_token_probs = []
    candidate_probs_per_step = []

    for step_idx, step_scores in enumerate(scores):
        probs = F.softmax(step_scores[0].float(), dim=-1)

        actual_id = generated_ids[step_idx]
        actual_prob = float(probs[actual_id].item())
        actual_token_probs.append(actual_prob)

        per_step = {"step": int(step_idx), "candidate_probabilities": {}}
        for cand, first_token_ids in candidate_token_ids.items():
            cand_prob = max(float(probs[tok_id].item()) for tok_id in first_token_ids)
            trace[cand].append(cand_prob)
            per_step["candidate_probabilities"][cand] = cand_prob

        candidate_probs_per_step.append(per_step)

    return trace, actual_token_probs, candidate_token_ids, candidate_probs_per_step


def run_probability_trace(
    model,
    processor,
    image,
    prompt,
    candidate_countries,
    use_image: bool,
    max_new_tokens=20,
    top_k_tokens=10,
):
    _, inputs = build_inputs(
        processor=processor,
        image=image,
        prompt=prompt,
        device=model.device,
        use_image=use_image,
    )
    prompt_len = inputs.input_ids.shape[1]

    gen_out = generate_with_scores(model, inputs, max_new_tokens=max_new_tokens)
    gen_ids, gen_token_texts = decode_generated_tokens(processor, gen_out, prompt_len)

    (
        trace,
        actual_token_probs,
        candidate_token_ids,
        candidate_probs_per_step,
    ) = trace_candidate_first_token_probs(
        processor=processor,
        generation_output=gen_out,
        candidates=candidate_countries,
    )

    top_tokens_per_step = get_top_tokens_per_step(
        processor=processor,
        generation_output=gen_out,
        top_k=top_k_tokens,
    )

    debug = {
        "use_image": use_image,
        "prompt": prompt,
        "generated_ids": gen_ids,
        "generated_tokens": gen_token_texts,
        "candidate_countries": candidate_countries,
        "candidate_first_token_ids": candidate_token_ids,
        "actual_generated_token_probabilities": actual_token_probs,
        "candidate_probabilities_per_step": candidate_probs_per_step,
        "top_tokens_per_step": top_tokens_per_step,
    }
    return trace, actual_token_probs, debug


def save_logit_trace_plot(trace, actual_token_probs, save_path, title="Logit / probability trace"):
    steps = list(range(len(actual_token_probs)))

    plt.figure(figsize=(7, 4.5))
    for cand, vals in trace.items():
        plt.plot(steps, vals, label=cand)

    plt.plot(steps, actual_token_probs, linestyle="--", label="actual generated token")

    plt.xlabel("Generation step")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def extract_country_probs_at_step(debug, step_idx=4):
    out = {"United States": 0.0, "Ethiopia": 0.0}

    per_step = debug.get("candidate_probabilities_per_step", [])
    for row in per_step:
        if row.get("step") == step_idx:
            cand_probs = row.get("candidate_probabilities", {})
            out["United States"] = float(cand_probs.get("United States", 0.0))
            out["Ethiopia"] = float(cand_probs.get("Ethiopia", 0.0))
            break

    return out


def save_step4_country_comparison_plot(
    step4_probs_by_config,
    save_path,
    title="Step 4 probability comparison",
):
    configs = list(step4_probs_by_config.keys())
    us_probs = [step4_probs_by_config[cfg].get("United States", 0.0) for cfg in configs]
    et_probs = [step4_probs_by_config[cfg].get("Ethiopia", 0.0) for cfg in configs]

    x = list(range(len(configs)))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], us_probs, width=width, label="United States")
    plt.bar([i + width / 2 for i in x], et_probs, width=width, label="Ethiopia")

    plt.xticks(x, configs, rotation=20, ha="right")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── step-4 logit helpers ──────────────────────────────────────────────────────
def get_candidate_first_token_ids(tokenizer, text: str):
    variants = [text, " " + text]
    out = []
    seen = set()
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if ids:
            tok_id = ids[0]
            if tok_id not in seen:
                out.append(tok_id)
                seen.add(tok_id)
    if not out:
        raise ValueError(f"Could not tokenize candidate text: {text!r}")
    return out


def extract_step_logit_info(processor, generation_output, step_idx, candidates):
    tokenizer = processor.tokenizer
    scores = generation_output.scores
    if step_idx >= len(scores):
        raise ValueError(f"step_idx={step_idx} out of range, only {len(scores)} steps")

    step_scores = scores[step_idx][0].float()
    step_probs = torch.softmax(step_scores, dim=-1)

    candidate_ids = {
        cand: get_candidate_first_token_ids(tokenizer, cand)
        for cand in candidates
    }

    result = {
        "step": int(step_idx),
        "candidate_logits": {},
        "candidate_probabilities": {},
        "candidate_token_ids": candidate_ids,
        "margins": {},
    }

    for cand, ids in candidate_ids.items():
        best_logit = max(float(step_scores[i].item()) for i in ids)
        best_prob = max(float(step_probs[i].item()) for i in ids)
        result["candidate_logits"][cand] = best_logit
        result["candidate_probabilities"][cand] = best_prob

    if "Ethiopia" in result["candidate_logits"] and "United States" in result["candidate_logits"]:
        eth_logit = result["candidate_logits"]["Ethiopia"]
        us_logit = result["candidate_logits"]["United States"]
        eth_prob = result["candidate_probabilities"]["Ethiopia"]
        us_prob = result["candidate_probabilities"]["United States"]

        result["margins"]["ethiopia_minus_us_logit"] = eth_logit - us_logit
        result["margins"]["ethiopia_minus_us_prob"] = eth_prob - us_prob

    return result


def save_step4_logit_margin_plot(step4_logit_info_by_config, save_path, title):
    configs = list(step4_logit_info_by_config.keys())
    margins = [
        step4_logit_info_by_config[cfg]["margins"].get("ethiopia_minus_us_logit", 0.0)
        for cfg in configs
    ]

    x = list(range(len(configs)))
    plt.figure(figsize=(10, 5))
    plt.bar(x, margins)
    plt.axhline(0.0, linestyle="--")
    plt.xticks(x, configs, rotation=20, ha="right")
    plt.ylabel("Logit margin (Ethiopia - United States)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── step-4 attention helpers ──────────────────────────────────────────────────
def find_image_token_span(inputs, processor):
    tok = processor.tokenizer
    vocab = tok.get_vocab()
    ids = inputs.input_ids[0].tolist()

    vision_start_id = vocab.get("<|vision_start|>")
    vision_end_id = vocab.get("<|vision_end|>")

    if vision_start_id is None or vision_end_id is None:
        return None

    start_positions = [i for i, x in enumerate(ids) if x == vision_start_id]
    end_positions = [i for i, x in enumerate(ids) if x == vision_end_id]

    if not start_positions or not end_positions:
        return None

    block_start = start_positions[0]
    block_end = next(i for i in end_positions if i > block_start)

    return block_start + 1, block_end


def extract_step_attention(processor, generation_output, step_idx, layer=-1):
    if step_idx >= len(generation_output.attentions):
        raise ValueError(f"step_idx={step_idx} out of range for attentions")

    step_attn_layers = generation_output.attentions[step_idx]
    layer_attn = step_attn_layers[layer]   # (batch, heads, q_len, kv_len)
    arr = layer_attn[0].detach().float().cpu().numpy()
    arr = arr.mean(axis=0)  # (q_len, kv_len)

    att_row = arr[-1]
    return att_row


def attention_vec_to_heatmap(att_vec, grid_thw, image_size):
    t, h, w = [int(x) for x in grid_thw[0].tolist()]
    full_needed = t * h * w
    actual = att_vec.shape[0]

    # Case 1: exact match
    if actual == full_needed:
        arr = att_vec.reshape(t, h, w).mean(axis=0)

    # Case 2: common merged-grid case, e.g. 1216 -> 304 = 19x16
    elif t == 1 and actual * 4 == full_needed and h % 2 == 0 and w % 2 == 0:
        arr = att_vec.reshape(1, h // 2, w // 2).mean(axis=0)

    # Case 3: try generic factorization for single-image case
    elif t == 1:
        found = False
        for hh in range(1, int(actual ** 0.5) + 1):
            if actual % hh == 0:
                ww = actual // hh
                # prefer aspect ratio close to original
                if abs((ww / hh) - (w / h)) < 0.5:
                    arr = att_vec.reshape(hh, ww)
                    found = True
                    break
        if not found:
            raise ValueError(f"Could not reshape attention vec of length {actual} into plausible grid")
    else:
        raise ValueError(f"attention vec too short: {actual} < {full_needed}")

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    hm = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        image_size, resample=Image.BILINEAR
    )
    hm = np.asarray(hm).astype(np.float32) / 255.0
    return hm


def overlay_heatmap_on_image(image, heatmap, alpha=0.65):
    base = np.asarray(image).astype(np.float32) / 255.0
    colored = plt.get_cmap("inferno")(heatmap)[..., :3].astype(np.float32)
    out = alpha * colored + (1 - alpha) * base
    return np.clip(out, 0, 1)


def save_step4_attention_visualization(
    processor,
    inputs,
    generation_output,
    image,
    save_path,
    step_idx=4,
    layer=-1,
):
    image_span = find_image_token_span(inputs, processor)
    if image_span is None:
        raise ValueError("No image token span found; likely a text-only example")

    att_row = extract_step_attention(processor, generation_output, step_idx=step_idx, layer=layer)
    start, end = image_span
    img_att = att_row[start:end]

    heatmap = attention_vec_to_heatmap(img_att, inputs["image_grid_thw"], image.size)
    vis = overlay_heatmap_on_image(image, heatmap, alpha=0.65)

    plt.figure(figsize=(6, 6))
    plt.imshow(vis)
    plt.axis("off")
    plt.title(f"Step {step_idx} attention to image tokens")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_step4_analysis(
    model,
    processor,
    image,
    prompt,
    use_image,
    candidates,
    max_new_tokens=20,
    step_idx=4,
):
    _, inputs = build_inputs(
        processor=processor,
        image=image,
        prompt=prompt,
        device=model.device,
        use_image=use_image,
    )

    gen_out = generate_with_scores_and_attentions(
        model=model,
        inputs=inputs,
        max_new_tokens=max_new_tokens,
    )

    step4_logit_info = extract_step_logit_info(
        processor=processor,
        generation_output=gen_out,
        step_idx=step_idx,
        candidates=candidates,
    )

    return inputs, gen_out, step4_logit_info


# ── main per-example runner ───────────────────────────────────────────────────
def run_example(example, model, processor, data_dir: Path, out_dir: Path, max_new_tokens: int):
    rid = example["id"]
    split = example["split"]
    gt = example.get("ground_truth", "")

    sample_dir = out_dir / f"{split}__{rid}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    image = None
    try:
        image = load_image(split, rid, data_dir)
        image.save(sample_dir / "original.jpg")
    except Exception as e:
        print(f"warning: image unavailable for {split}/{rid}: {e}")

    step4_probs_by_config = {}
    step4_logit_info_by_config = {}

    for exp in EXPERIMENTS:
        use_image = experiment_uses_image(exp)
        if use_image and image is None:
            print(f"skip {split}/{rid} {exp}: image unavailable")
            continue

        prompt = make_prompt(exp, example)

        pred_info = example.get("predictions", {}).get(exp, {})
        pred = pred_info.get("predicted", None)

        candidate_countries = []
        for c in [pred, gt, "United States", "Ethiopia"]:
            if c and c not in candidate_countries:
                candidate_countries.append(c)

        if not candidate_countries:
            print(f"skip {split}/{rid} {exp}: no candidate countries")
            continue

        try:
            # probability trace
            trace, actual_token_probs, debug = run_probability_trace(
                model=model,
                processor=processor,
                image=image if use_image else None,
                prompt=prompt,
                candidate_countries=candidate_countries,
                use_image=use_image,
                max_new_tokens=max_new_tokens,
            )

            plot_path = sample_dir / f"{exp}_probability_trace.png"
            debug_path = sample_dir / f"{exp}_probability_trace_debug.json"

            save_logit_trace_plot(
                trace=trace,
                actual_token_probs=actual_token_probs,
                save_path=plot_path,
                title=f"{split} | {rid} | {exp}",
            )

            with open(debug_path, "w") as f:
                json.dump(debug, f, indent=2, ensure_ascii=False)

            step4_probs_by_config[exp] = extract_country_probs_at_step(debug, step_idx=4)

            # step-4 logit + attention analysis
            inputs, gen_out, step4_logit_info = run_step4_analysis(
                model=model,
                processor=processor,
                image=image if use_image else None,
                prompt=prompt,
                use_image=use_image,
                candidates=candidate_countries,
                max_new_tokens=max_new_tokens,
                step_idx=4,
            )

            step4_logit_info_by_config[exp] = step4_logit_info

            with open(sample_dir / f"{exp}_step4_logit_info.json", "w") as f:
                json.dump(step4_logit_info, f, indent=2, ensure_ascii=False)

            if use_image:
                try:
                    save_step4_attention_visualization(
                        processor=processor,
                        inputs=inputs,
                        generation_output=gen_out,
                        image=image,
                        save_path=sample_dir / f"{exp}_step4_attention_to_image.png",
                        step_idx=4,
                        layer=-1,
                    )
                except Exception as e:
                    print(f"warning: could not save step-4 attention viz for {exp}: {e}")

            print(f"saved {plot_path}")

        except Exception as e:
            print(f"error for {split}/{rid} {exp}: {e}")

    # save cross-config step-4 probability comparison
    if step4_probs_by_config:
        comparison_plot_path = sample_dir / "step4_us_vs_ethiopia_comparison.png"
        comparison_json_path = sample_dir / "step4_us_vs_ethiopia_comparison.json"

        save_step4_country_comparison_plot(
            step4_probs_by_config=step4_probs_by_config,
            save_path=comparison_plot_path,
            title=f"{split} | {rid} | step 4: United States vs Ethiopia",
        )

        with open(comparison_json_path, "w") as f:
            json.dump(step4_probs_by_config, f, indent=2, ensure_ascii=False)

        print(f"saved {comparison_plot_path}")

    # save cross-config step-4 logit margin comparison
    if step4_logit_info_by_config:
        margin_plot_path = sample_dir / "step4_logit_margin_comparison.png"
        margin_json_path = sample_dir / "step4_logit_margin_comparison.json"

        save_step4_logit_margin_plot(
            step4_logit_info_by_config=step4_logit_info_by_config,
            save_path=margin_plot_path,
            title=f"{split} | {rid} | step 4 logit margin (Ethiopia - United States)",
        )

        with open(margin_json_path, "w") as f:
            json.dump(step4_logit_info_by_config, f, indent=2, ensure_ascii=False)

        print(f"saved {margin_plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=str, default="us_prediction_images")
    parser.add_argument("--data-dir", type=str, default="cammt_data/data")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--out-dir", type=str, default="step4_analysis_outputs")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    jsonl_dir = Path(args.jsonl_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print("Model ready.")

    examples = load_examples(jsonl_dir, split_filter=args.split, max_n=args.max_examples)
    print(f"Loaded {len(examples)} examples.")

    for i, ex in enumerate(examples, 1):
        print(f"[{i}/{len(examples)}] {ex['split']} | {ex['id']}")
        run_example(
            example=ex,
            model=model,
            processor=processor,
            data_dir=data_dir,
            out_dir=out_dir,
            max_new_tokens=args.max_new_tokens,
        )

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()