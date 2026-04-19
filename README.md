# Qwen2SAM_ZS

Zero-Shot VLM → SAM3 texture segmentation baseline. No training, no custom
bridges — just Qwen3-VL-8B describing distinct surface regions and SAM3's
native text encoder + Semantic Seg Head turning each phrase into a heatmap.
Conflicts are resolved by Softmax + Argmax over `(K+1, H, W)` logits
(textures + a static "dustbin" channel); evaluation uses Hungarian matching
on the IoU cost matrix.

The region count `K` is set **per-sample** from the GT:

- **RWTD** → always K=2.
- **ADE20k_DeTexture** → K = number of GT textures for that image (1..6).

## Install

```bash
# 1. clone this repo
git clone https://github.com/<your-gh-user>/Qwen2SAM_ZS.git
cd Qwen2SAM_ZS

# 2. clone SAM3 somewhere and point to it
git clone https://github.com/facebookresearch/sam3.git ~/sam3
pip install -e ~/sam3
export SAM3_ROOT=~/sam3   # optional; defaults to /home/aviad/sam3

# 3. install python deps
pip install -r requirements.txt
```

Weights download on first run from Hugging Face:

- `Qwen/Qwen3-VL-8B-Instruct`
- SAM3 (loaded via `build_sam3_image_model(load_from_HF=True)`)

## Datasets

The script expects unified metadata files. Defaults baked into the
[registry](./evaluate_zero_shot_pipeline.py) point at:

| Key                 | Metadata path                                      |
| ------------------- | -------------------------------------------------- |
| `rwtd`              | `/home/aviad/datasets/RWTD/metadata.json`          |
| `rwtd_phase1`       | `/home/aviad/RWTD/metadata_phase1.json` (legacy)   |
| `ade20k_detexture`  | `/home/aviad/datasets/ADE20k_DeTexture/metadata.json` |
| `ade20k_textured`   | `/home/aviad/datasets/ADE20K_textured_images/metadata.json` |
| `custom`            | pass `--metadata <path>` (+ optional `--schema`)   |

Unified schema per entry:

```json
{
  "image_path": "/abs/path/image.jpg",
  "id": "sample_id",
  "textures": [
    {"description": "…", "mask_path": "/abs/path/mask_0.png"},
    {"description": "…", "mask_path": "/abs/path/mask_1.png"}
  ]
}
```

Override any dataset's metadata path with `--metadata` (useful if your paths
differ from the defaults).

## Run

```bash
# RWTD (K=2 for every sample)
python evaluate_zero_shot_pipeline.py --dataset rwtd

# ADE20k_DeTexture (K varies per sample)
python evaluate_zero_shot_pipeline.py --dataset ade20k_detexture --limit 50

# Custom metadata
python evaluate_zero_shot_pipeline.py --dataset custom \
    --metadata /path/to/metadata.json --output_dir /tmp/my_eval
```

Useful flags:

| Flag                  | Default                          | Notes                                    |
| --------------------- | -------------------------------- | ---------------------------------------- |
| `--dataset`           | `rwtd`                           | registry key or `custom`                 |
| `--metadata`          | from registry                    | override the metadata path               |
| `--output_dir`        | `./eval_results`                 | results + visualizations                 |
| `--image_size`        | `1008`                           | SAM3 input resolution                    |
| `--dustbin_logit`     | `0.0`                            | 0.0 ≡ sigmoid p=0.5                      |
| `--max_textures`      | `10`                             | safety cap on the parser                 |
| `--min_gt_area_frac`  | `0.0`                            | drop tiny GT masks                       |
| `--samples`           | —                                | comma-separated ids                      |
| `--limit`             | —                                | process first N samples                  |
| `--no_vis`            | off                              | disable per-sample PNGs                  |
| `--vis_every`         | `1`                              | save every Nth visualization             |

## Outputs

Per dataset, the script writes `eval_results/<dataset>/`:

- `zero_shot_results.json` — summary + per-sample records
  (`panoptic_iou`, `matched_mean_iou`, `assignment`, `descs`, …).
- `vis/<id>.png` — a dynamic grid (top row: image + GT overlay + pred
  overlay + info; then one row per predicted texture with logits, pred mask,
  matched GT mask, and a contour overlay).

## Pipeline summary

1. **Qwen3-VL-8B** — prompted for exactly `K = K_GT` distinct surface
   regions. Returns `TEXTURE_1 … TEXTURE_K` phrases.
2. **SAM3** — for each phrase, `backbone.forward_text` → Multimodal Decoder
   → Semantic Seg Head → `(H, W)` logits.
3. **Softmax + Argmax with dustbin** — stack `(K+1, H, W)` logits (K
   textures + static `dustbin_logit`), softmax across channels, argmax.
4. **Hungarian (K × M)** — `scipy.optimize.linear_sum_assignment` on
   `1 - IoU` between predicted classes and GT masks. With K=M the
   assignment is a square permutation, so panoptic IoU and matched mean IoU
   coincide.
