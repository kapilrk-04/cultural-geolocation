#!/usr/bin/env python3
"""
Step-4 focus analysis for Qwen2.5-VL.

Purpose:
- analyze what the model focuses on at the decision step
- highlight:
    1) step-4 logit margin (Ethiopia - United States)
    2) top attended tokens at step 4
    3) image-vs-text attention mass at step 4

Supports:
    * image_only
    * image_english
    * image_regional
    * textonly_english
    * textonly_regional

Outputs per example:
    - {exp}_step4_focus.json
    - {exp}_step4_attention_mass.png
    - step4_logit_margin_comparison.png
    - step4_logit_margin_comparison.json
    - step4_focus_summary.txt
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


def make_prompt_parts(exp_name: str, ex: dict):
    regional = ex.get("regional") or ""
    english = ex.get("english") or ""

    if exp_name == "image_only":
        instruction = (
            "Look at this image carefully. "
            "Based on ALL available visual clues (architecture, vegetation, signage, "
            "vehicles, clothing, environment, text visible in the image, etc.), "
            "predict which country this image was most likely taken in. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name. No explanation. /no_think'
        )
        aux_text = ""
    elif exp_name == "image_english":
        instruction = (
            "Look at this image and read the associated text carefully. "
            "Based on ALL available clues, predict which country this image was most likely taken in. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name. No explanation. /no_think'
        )
        aux_text = f'Text: "{english}"'
    elif exp_name == "image_regional":
        instruction = (
            "Look at this image and read the associated text carefully. "
            "The text is written in the local language. "
            "Based on ALL available clues, predict which country this image was most likely taken in. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name. No explanation. /no_think'
        )
        aux_text = f'Text: "{regional}"'
    elif exp_name == "textonly_english":
        instruction = (
            "Read the following text carefully. "
            "Based on cultural and contextual clues in the text, "
            "predict which country this text most likely originates from. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
            "No explanation, no extra text. /no_think"
        )
        aux_text = f'Text: "{english}"'
    elif exp_name == "textonly_regional":
        instruction = (
            "Read the following text carefully. "
            "It is written in a local language and describes something culturally specific. "
            "Based ONLY on linguistic and cultural clues in the text, "
            "predict which country this text most likely originates from. "
            'Respond with ONLY a JSON object: {"country": "<country_name>"} '
            'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
            "No explanation, no extra text. /no_think"
        )
        aux_text = f'Text: "{regional}"'
    else:
        raise ValueError(exp_name)

    return instruction, aux_text


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


# ── token helpers ─────────────────────────────────────────────────────────────
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


def safe_decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return f"<tok_{token_id}>"


def get_effective_step_idx(generation_output, preferred_step=4):
    """
    Use preferred_step if available, otherwise fall back to the last generation step.
    """
    num_steps = len(generation_output.scores)
    if num_steps == 0:
        raise ValueError("No generation steps found")
    return min(preferred_step, num_steps - 1)


# ── step analysis ─────────────────────────────────────────────────────────────
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

    top_probs, top_ids = torch.topk(step_probs, k=10)
    result["top_tokens"] = [
        {
            "token_id": int(tok_id),
            "token_text": safe_decode_token(tokenizer, int(tok_id)),
            "probability": float(prob),
            "logit": float(step_scores[int(tok_id)].item()),
        }
        for tok_id, prob in zip(top_ids.tolist(), top_probs.tolist())
    ]

    return result


# ── attention analysis ────────────────────────────────────────────────────────
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


def extract_step_attention_row(generation_output, step_idx, layer=-1):
    if step_idx >= len(generation_output.attentions):
        raise ValueError(f"step_idx={step_idx} out of range for attentions")

    step_attn_layers = generation_output.attentions[step_idx]
    layer_attn = step_attn_layers[layer]   # (batch, heads, q_len, kv_len)
    arr = layer_attn[0].detach().float().cpu().numpy()
    arr = arr.mean(axis=0)
    return arr[-1]


def get_top_attended_tokens(att_row, inputs, processor, top_k=15, generated_ids=None):
    prompt_ids = inputs.input_ids[0].tolist()
    all_ids = prompt_ids + (list(generated_ids) if generated_ids else [])
    tokenizer = processor.tokenizer

    vals = []
    for i, score in enumerate(att_row):
        tok_id = all_ids[i] if i < len(all_ids) else -1
        vals.append({
            "position": int(i),
            "token_id": int(tok_id),
            "token_text": safe_decode_token(tokenizer, int(tok_id)) if tok_id >= 0 else "<gen>",
            "attention": float(score),
        })

    vals.sort(key=lambda x: x["attention"], reverse=True)
    return vals[:top_k]


def split_attention_mass(att_row, inputs, processor):
    image_span = find_image_token_span(inputs, processor)
    total = float(np.sum(att_row)) if len(att_row) else 0.0

    if image_span is None:
        return {
            "total_attention_mass": total,
            "image_attention_mass": 0.0,
            "text_attention_mass": total,
            "image_attention_fraction": 0.0,
            "text_attention_fraction": 1.0 if total > 0 else 0.0,
        }

    start, end = image_span
    image_mass = float(np.sum(att_row[start:end]))
    text_mass = float(total - image_mass)

    return {
        "total_attention_mass": total,
        "image_attention_mass": image_mass,
        "text_attention_mass": text_mass,
        "image_attention_fraction": image_mass / total if total > 0 else 0.0,
        "text_attention_fraction": text_mass / total if total > 0 else 0.0,
    }


def split_attention_mass_detailed(att_row, inputs, processor, image_span=None, instruction_len=0, aux_len=0):
    ids = inputs.input_ids[0].tolist()
    input_len = len(ids)
    att_row = att_row[:input_len]

    total = float(np.sum(att_row)) if len(att_row) else 0.0

    masses = {
        "total_attention_mass": total,
        "image_attention_mass": 0.0,
        "instruction_attention_mass": 0.0,
        "aux_text_attention_mass": 0.0,
        "other_text_attention_mass": 0.0,
    }

    text_mask = np.ones(input_len, dtype=bool)

    if image_span is not None:
        s, e = image_span
        masses["image_attention_mass"] = float(np.sum(att_row[s:e]))
        text_mask[s:e] = False

    text_indices = np.where(text_mask)[0]

    inst_end = min(instruction_len, len(text_indices))
    aux_end = min(inst_end + aux_len, len(text_indices))

    inst_idx = text_indices[:inst_end]
    aux_idx = text_indices[inst_end:aux_end]
    other_idx = text_indices[aux_end:]

    masses["instruction_attention_mass"] = float(np.sum(att_row[inst_idx])) if len(inst_idx) else 0.0
    masses["aux_text_attention_mass"] = float(np.sum(att_row[aux_idx])) if len(aux_idx) else 0.0
    masses["other_text_attention_mass"] = float(np.sum(att_row[other_idx])) if len(other_idx) else 0.0

    for k in list(masses.keys()):
        if k.endswith("_mass") and k != "total_attention_mass":
            frac_key = k.replace("_mass", "_fraction")
            masses[frac_key] = masses[k] / total if total > 0 else 0.0

    return masses


def save_attention_mass_plot(mass_info, save_path, title):
    labels = ["image", "text"]
    vals = [
        mass_info.get("image_attention_fraction", 0.0),
        mass_info.get("text_attention_fraction", 0.0),
    ]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, vals)
    plt.ylim(0, 1)
    plt.ylabel("Attention fraction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── summary plots ─────────────────────────────────────────────────────────────
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

    step4_logit_info_by_config = {}
    summary_lines = [f"Example: {split} | {rid}", f"Ground truth: {gt}", ""]

    for exp in EXPERIMENTS:
        use_image = experiment_uses_image(exp)
        if use_image and image is None:
            print(f"skip {split}/{rid} {exp}: image unavailable")
            continue

        instruction, aux_text = make_prompt_parts(exp, example)
        prompt = instruction if not aux_text else f"{instruction} {aux_text}"

        pred_info = example.get("predictions", {}).get(exp, {})
        pred = pred_info.get("predicted", None)

        candidate_countries = []
        for c in [pred, gt, "United States", "Ethiopia"]:
            if c and c not in candidate_countries:
                candidate_countries.append(c)

        if "United States" not in candidate_countries:
            candidate_countries.append("United States")
        if "Ethiopia" not in candidate_countries:
            candidate_countries.append("Ethiopia")

        try:
            _, inputs = build_inputs(
                processor=processor,
                image=image if use_image else None,
                prompt=prompt,
                device=model.device,
                use_image=use_image,
            )
            prompt_len = inputs.input_ids.shape[1]

            gen_out = generate_with_scores_and_attentions(
                model=model,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
            )
            gen_ids, gen_tokens = decode_generated_tokens(processor, gen_out, prompt_len)

            effective_step = get_effective_step_idx(gen_out, preferred_step=4)
            print(
                f"{exp}: num_generation_steps={len(gen_out.scores)} "
                f"effective_step={effective_step} generated_tokens={gen_tokens}"
            )

            step_logit_info = extract_step_logit_info(
                processor=processor,
                generation_output=gen_out,
                step_idx=effective_step,
                candidates=candidate_countries,
            )
            step4_logit_info_by_config[exp] = step_logit_info

            att_row = extract_step_attention_row(gen_out, step_idx=effective_step, layer=-1)
            top_attended = get_top_attended_tokens(
                att_row, inputs, processor, top_k=15,
                generated_ids=gen_ids[:effective_step],
            )
            mass_info = split_attention_mass(att_row, inputs, processor)

            image_span = find_image_token_span(inputs, processor)
            tok = processor.tokenizer
            instruction_len = len(tok.encode(instruction, add_special_tokens=False))
            aux_len = len(tok.encode(aux_text, add_special_tokens=False)) if aux_text else 0
            detailed_mass_info = split_attention_mass_detailed(
                att_row, inputs, processor,
                image_span=image_span,
                instruction_len=instruction_len,
                aux_len=aux_len,
            )

            focus_report = {
                "experiment": exp,
                "predicted_country": pred,
                "generated_ids": gen_ids,
                "generated_tokens": gen_tokens,
                "effective_step_used": effective_step,
                "step_logit_info": step_logit_info,
                "step_top_attended_tokens": top_attended,
                "step_attention_mass": mass_info,
                "step_attention_mass_detailed": detailed_mass_info,
            }

            with open(sample_dir / f"{exp}_step4_focus.json", "w") as f:
                json.dump(focus_report, f, indent=2, ensure_ascii=False)

            save_attention_mass_plot(
                mass_info,
                sample_dir / f"{exp}_step4_attention_mass.png",
                title=f"{exp}: step {effective_step} attention mass",
            )

            margin = step_logit_info["margins"].get("ethiopia_minus_us_logit", None)
            summary_lines.append(f"[{exp}]")
            summary_lines.append(f"  predicted: {pred}")
            summary_lines.append(f"  effective step used: {effective_step}")
            summary_lines.append(f"  step margin (eth-us): {margin}")
            summary_lines.append(
                f"  attention fraction image={detailed_mass_info['image_attention_fraction']:.4f} "
                f"instruction={detailed_mass_info['instruction_attention_fraction']:.4f} "
                f"aux={detailed_mass_info['aux_text_attention_fraction']:.4f} "
                f"other={detailed_mass_info['other_text_attention_fraction']:.4f}"
            )
            summary_lines.append("  top attended tokens:")
            for row in top_attended[:10]:
                tok = row["token_text"].replace("\n", "\\n")
                summary_lines.append(
                    f"    pos={row['position']:>4} att={row['attention']:.6f} tok={tok!r}"
                )
            summary_lines.append("")

            print(f"saved {sample_dir / f'{exp}_step4_focus.json'}")

        except Exception as e:
            print(f"error for {split}/{rid} {exp}: {e}")

    if step4_logit_info_by_config:
        save_step4_logit_margin_plot(
            step4_logit_info_by_config=step4_logit_info_by_config,
            save_path=sample_dir / "step4_logit_margin_comparison.png",
            title=f"{split} | {rid} | step logit margin (Ethiopia - United States)",
        )

        with open(sample_dir / "step4_logit_margin_comparison.json", "w") as f:
            json.dump(step4_logit_info_by_config, f, indent=2, ensure_ascii=False)

        print(f"saved {sample_dir / 'step4_logit_margin_comparison.png'}")

    with open(sample_dir / "step4_focus_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print(f"saved {sample_dir / 'step4_focus_summary.txt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-dir", type=str, default="us_prediction_images")
    parser.add_argument("--data-dir", type=str, default="cammt_data/data")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--out-dir", type=str, default="step4_focus_outputs")
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