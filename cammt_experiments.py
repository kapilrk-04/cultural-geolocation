import os
import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "./cammt_data"
OUTPUT_DIR = "./results"
BATCH_SIZE = 4

SPLIT_TO_COUNTRY = {
    "es_mex": "Mexico",
    "bn_india": "India",
    "om_eth": "Ethiopia",
    "ur_india": "India",
    "ig_nga": "Nigeria",
    "ur_pak": "Pakistan",
    "zh_ch": "China",
    "es_ecu": "Ecuador",
    "sw_ken": "Kenya",
    "kor_sk": "South Korea",
    "ru_rus": "Russia",
    "ta_india": "India",
    "amh_eth": "Ethiopia",
    "jp_jap": "Japan",
    "fil_phl": "Philippines",
    "ms_mys": "Malaysia",
    "bg_bg": "Bulgaria",
    "es_chl": "Chile",
    "pt_brz": "Brazil",
    "ar_egy": "Egypt",
    "ind_ind": "Indonesia",
    "mr_india": "India",
    "es_arg": "Argentina",
}

PROMPTS = {
    "image_only": (
        "Look at this image carefully. "
        "Based on ALL available visual clues (architecture, vegetation, signage, "
        "vehicles, clothing, environment, text visible in the image, etc.), "
        "predict which country this image was most likely taken in. "
        'Respond with ONLY a JSON object: {"country": "<country_name>"} '
        'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
        "No explanation, no extra text. /no_think"
    ),
    "image_regional": lambda t: (
        "Look at this image and read the associated text carefully. "
        "The text is written in the local language about something visible in the image "
        "and may contain cultural, linguistic, or geographic clues. "
        f'Text: "{t}" '
        "Based on ALL available clues (image, architecture, vegetation, signage, "
        "vehicles, environment, and the text), predict which country this image was most likely taken in. "
        'Respond with ONLY a JSON object: {"country": "<country_name>"} '
        'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
        "No explanation, no extra text. /no_think"
    ),
    "image_english": lambda t: (
        "Look at this image and read the associated text carefully. "
        f'Text: "{t}" '
        "Based on ALL available clues (image, architecture, vegetation, signage, "
        "vehicles, environment, and the text), predict which country this image was most likely taken in. "
        'Respond with ONLY a JSON object: {"country": "<country_name>"} '
        'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
        "No explanation, no extra text. /no_think"
    ),
    "textonly_regional": lambda t: (
        "Read the following text carefully. "
        "It is written in a local language and describes something culturally specific. "
        "Based ONLY on linguistic and cultural clues in the text, "
        "predict which country this text most likely originates from. "
        f'Text: "{t}" '
        'Respond with ONLY a JSON object: {"country": "<country_name>"} '
        'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
        "No explanation, no extra text. /no_think"
    ),
    "textonly_english": lambda t: (
        "Read the following text carefully. "
        "Based on cultural and contextual clues in the text, "
        "predict which country this text most likely originates from. "
        f'Text: "{t}" '
        'Respond with ONLY a JSON object: {"country": "<country_name>"} '
        'Use the English country name (e.g., "Japan", "Brazil", "South Korea"). '
        "No explanation, no extra text. /no_think"
    ),
}

USES_IMAGE = {"image_only", "image_regional", "image_english"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_messages(exp_name, sample):
    """Build the messages list for a single sample."""
    regional = sample.get("regional") or ""
    english = sample.get("English") or ""

    if exp_name == "image_only":
        text = PROMPTS["image_only"]
        content = [{"type": "image", "image": sample["image"]}, {"type": "text", "text": text}]
    elif exp_name == "image_regional":
        text = PROMPTS["image_regional"](regional)
        content = [{"type": "image", "image": sample["image"]}, {"type": "text", "text": text}]
    elif exp_name == "image_english":
        text = PROMPTS["image_english"](english)
        content = [{"type": "image", "image": sample["image"]}, {"type": "text", "text": text}]
    elif exp_name == "textonly_regional":
        text = PROMPTS["textonly_regional"](regional)
        content = [{"type": "text", "text": text}]
    elif exp_name == "textonly_english":
        text = PROMPTS["textonly_english"](english)
        content = [{"type": "text", "text": text}]
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")

    return [{"role": "user", "content": content}]


def parse_country(raw: str) -> str:
    """Extract country from model output; return raw string if parsing fails."""
    # Try strict JSON first
    try:
        obj = json.loads(raw.strip())
        return obj.get("country", raw).strip()
    except json.JSONDecodeError:
        pass
    # Fallback: regex for {"country": "..."}
    m = re.search(r'"country"\s*:\s*"([^"]+)"', raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def checkpoint_path(output_dir: str, exp_name: str, split_name: str) -> Path:
    return Path(output_dir) / exp_name / f"{split_name}.jsonl"


def load_checkpoint(path: Path) -> set:
    """Return set of already-processed IDs from a checkpoint file."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment(exp_name, split_name, split_data, model, processor, output_dir):
    out_path = checkpoint_path(output_dir, exp_name, split_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_checkpoint(out_path)
    ground_truth = SPLIT_TO_COUNTRY[split_name]

    samples = [s for s in split_data if s["ID"] not in done_ids]
    if not samples:
        print(f"  [skip] {exp_name}/{split_name} — already complete")
        return

    correct = 0
    total = 0

    with open(out_path, "a") as f_out:
        pbar = tqdm(range(0, len(samples), BATCH_SIZE), desc=f"{exp_name}/{split_name}", unit="batch")
        for i in pbar:
            batch = samples[i : i + BATCH_SIZE]

            texts = []
            vision_inputs_list = []
            for sample in batch:
                messages = build_messages(exp_name, sample)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)

                if exp_name in USES_IMAGE:
                    image_inputs, _ = process_vision_info(messages)
                    vision_inputs_list.append(image_inputs)
                else:
                    vision_inputs_list.append(None)

            all_images = []
            for v in vision_inputs_list:
                if v:
                    all_images.extend(v)

            if all_images:
                inputs = processor(text=texts, images=all_images, padding=True, return_tensors="pt").to(model.device)
            else:
                inputs = processor(text=texts, padding=True, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=30)
            torch.cuda.empty_cache()

            generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            raw_outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for sample, raw in zip(batch, raw_outputs):
                predicted = parse_country(raw)
                is_correct = predicted.strip().lower() == ground_truth.lower()
                correct += int(is_correct)
                total += 1

                record = {
                    "id": sample["ID"],
                    "split": split_name,
                    "experiment": exp_name,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "correct": is_correct,
                    "raw_output": raw,
                }
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

            pbar.set_postfix(acc=f"{correct}/{total}")

    print(f"  Done {exp_name}/{split_name} — acc: {correct}/{total} = {correct/total:.3f}")


def summarize_results(output_dir: str):
    """Print per-experiment accuracy summary across all splits."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp_name in PROMPTS:
        exp_dir = Path(output_dir) / exp_name
        if not exp_dir.exists():
            continue
        total_correct = 0
        total_samples = 0
        for jsonl in sorted(exp_dir.glob("*.jsonl")):
            with open(jsonl) as f:
                records = [json.loads(l) for l in f if l.strip()]
            if not records:
                continue
            c = sum(r["correct"] for r in records)
            t = len(records)
            total_correct += c
            total_samples += t
            print(f"  {exp_name:25s} | {jsonl.stem:15s} | {c}/{t} = {c/t:.3f}")
        if total_samples:
            print(f"  {'':25s}   {'TOTAL':15s} | {total_correct}/{total_samples} = {total_correct/total_samples:.3f}")
        print()


def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--experiments", nargs="+", default=list(PROMPTS.keys()),
                        choices=list(PROMPTS.keys()), help="Which experiments to run")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Which dataset splits to run (default: all)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    # Load dataset
    print(f"Loading dataset from {args.data_path} ...")
    ds = load_dataset(args.data_path)
    splits = args.splits or list(ds.keys())
    unknown = set(splits) - set(SPLIT_TO_COUNTRY)
    if unknown:
        raise ValueError(f"Unknown splits (no country mapping): {unknown}")
    print(f"Dataset loaded: {len(splits)} splits, e.g. {splits[:3]}")

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded.")

    print("Loading processor ...")
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=512*512)
    processor.tokenizer.padding_side = "left"
    print("Processor loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    for exp_name in args.experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*60}")
        for split_name in splits:
            run_experiment(
                exp_name=exp_name,
                split_name=split_name,
                split_data=ds[split_name],
                model=model,
                processor=processor,
                output_dir=args.output_dir,
            )

    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
