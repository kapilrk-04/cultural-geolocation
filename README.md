# CaMMT-VL: Country Geolocation Probing of Qwen2.5-VL

This project looks at how **Qwen2.5-VL-7B-Instruct** guesses the country of origin of culturally specific images from the [CaMMT](https://huggingface.co/datasets/villacu/cammt) dataset. It compares four input settings: image only, image + English caption, image + regional-language caption, and regional caption only. It also asks why the model so often answers **"United States"**, using attention maps and logit/probability traces.

## Experiments

| Experiment | Input to the model |
|---|---|
| `image_only` | Image only |
| `image_english` | Image + English caption |
| `image_regional` | Image + caption in the local language |
| `textonly_regional` | Local-language caption only (no image) |
| `textonly_english` | English caption only (used in the analysis scripts) |

Each prompt asks the model to reply with `{"country": "<country_name>"}`. The prediction is compared against the country mapped from the dataset split (for example `amh_eth` → Ethiopia, `es_mex` → Mexico; the full map is `SPLIT_TO_COUNTRY` in [cammt_experiments.py](cammt_experiments.py)).

### Results (all splits)

| Experiment |  Accuracy | "United States" predictions |
|---|---|---|
| `image_only` | 0.561 | 433 |
| `image_english` | 0.760 | 172 |
| `image_regional` |  0.859 | 24 |
| `textonly_regional` |  0.803 | 9 |

`image_regional` has no results for `es_arg` (22 of the 23 splits). Per-example predictions are in `results/<experiment>/<split>.jsonl`, one record per line:

```json
{"id": "...", "split": "amh_eth", "experiment": "image_only", "ground_truth": "Ethiopia",
 "predicted": "United States", "correct": false, "raw_output": "{\"country\": \"United States\"}"}
```

## Repository layout

| File | Purpose |
|---|---|
| [cammt_project.ipynb](cammt_project.ipynb) | Downloads the CaMMT dataset from the Hugging Face Hub into `./cammt_data` |
| [cammt_experiments.py](cammt_experiments.py) | Main batched inference for every experiment and split; writes `results/` and prints an accuracy summary |
| [us_analysis.py](us_analysis.py) | Finds examples where any image experiment predicted the US; saves the images and per-split JSONL to `us_prediction_images/`, plus a grid plot |
| [visualize_us_predictions.py](visualize_us_predictions.py) | Earlier version of the US-prediction visualization |
| [attention_viz.py](attention_viz.py) | Attention heatmaps for the US-predicted examples (custom overlay) |
| [attention_vizs.py](attention_vizs.py) | Attention grid-to-grid and per-head plots using the Visualizer library; writes `attention_vizls/` |
| [prob_viz.py](prob_viz.py) | Token probability traces and the step-4 US-vs-true-country logit comparison |
| [logit_viz.py](logit_viz.py) | Step-4 focus analysis: logit margin, most-attended tokens, image-vs-text attention mass |
| [create_zip.py](create_zip.py) | Packs a random sample of attention visualizations into a zip |
| `us_preds.png`, `use_for_pres.zip` | Saved figures used in the presentation |

## Setup

A CUDA GPU is required. The inference script loads the model with `flash_attention_2` in bfloat16.

```bash
pip install torch transformers datasets huggingface_hub qwen-vl-utils flash-attn \
            pandas pyarrow pillow matplotlib numpy tqdm
# Needed only for the attention scripts:
pip install bytecode git+https://github.com/luo3300612/Visualizer
```

If the Hugging Face Hub needs authentication, set the token as an environment variable. Never write it into the code or the notebook:

```bash
export HF_TOKEN=<your_token>
```

## Usage

Run everything from this directory.

```bash
# 1. Download the data (run the first cell of cammt_project.ipynb) -> ./cammt_data

# 2. Run the inference experiments
python cammt_experiments.py                                   # all experiments, all splits
python cammt_experiments.py --experiments image_only --splits amh_eth es_mex --batch_size 8

# 3. Collect the examples where the model predicted the US
python us_analysis.py                                         # -> us_prediction_images/

# 4. Interpretability analyses on those examples
python attention_vizs.py --split amh_eth --max-examples 5     # -> attention_vizls/
python prob_viz.py       --split amh_eth --max-examples 1     # -> step4_analysis_outputs/
python logit_viz.py      --split amh_eth --max-examples 1     # -> step4_focus_outputs/
```

Note: the default `--out-dir` in `attention_viz.py` is a cluster path (`/l/users/...`). Pass `--out-dir` when running it anywhere else.
