# cellpose-sam microglia segmentation

Segmentation pipeline for microglia in 2-channel fluorescence microscopy (stitched tilescans from insula cortex), built on top of [Cellpose-SAM](https://github.com/MouseLand/cellpose) (`cpsam` model, v4).

- **Input**: stitched `.tif` files, shape `(2, H, W)` uint8 — channel 0 = DAPI (nuclei), channel 1 = IBA1 (microglia).
- **Output**: `uint16` label masks, annotated overview PNGs, and per-cell IBA1 crops.

## Setup

```bash
conda env create -f environment.yml
```

This pins `torch==2.9.1+cu126` for NVIDIA GPUs with CUDA 12.6 (tested on RTX 3060). All scripts are invoked under the env:

```bash
conda run -n cellpose python -u <script.py>
```

Always pass `-u` — without it `conda run` buffers stdout and nothing prints until the process finishes.

## Workflow

The pipeline runs in three stages. Start each new experiment at stage 1 — model behaviour varies across datasets and the "best" parameters from one dataset are rarely optimal for another.

### 1. Tune — narrow down parameters on tile-sized crops

Runs a parameter grid on a small sample of crops from a handful of images. Fast (minutes) because it uses tile-sized inputs, not full images.

```bash
conda run -n cellpose python -u tune_parameters.py                                 # phase 1 (high-impact)
conda run -n cellpose python -u tune_parameters.py --phase 2 --diameter 150 --cellprob -2.0
```

Phases: **1** varies `diameter × cellprob_threshold` (biggest levers); **2** varies `flow_threshold × pix_filter`; **3** covers low-impact params. Any individual parameter can be overridden with a comma-separated CLI flag, e.g. `--diameter 100,150,200`.

Outputs:
- `tune/phase<N>/<sample>__<combo>_labelled_segmentations.png` — one annotated preview per (crop × combo)
- `tune/phase<N>/summary.csv` — one row per (crop, combo) with `n_cells`, `size_min/max/mean/median` in pixels

See [PARAMETERS.md](PARAMETERS.md) for what each parameter does, default sweep ranges, and symptoms of bad values.

### 2. Validate — sanity-check on a few full images

Once a combo looks good on crops, run it through the full tiled pipeline on a few randomly chosen images to confirm it holds up at scale.

```bash
conda run -n cellpose python -u validate_parameters.py \
    --diameter 150 --cellprob -2.0 --flow 1.0 --pix-filter 500 --tnb 100
```

Outputs labelled full-image overviews to `validate/` + prints cell counts and mask-size stats per image.

### 3. Run — full segmentation over the dataset

```bash
conda run -n cellpose python -u run_segmentation.py \
    --diameter 150 --cellprob-threshold -2.0 --flow-threshold 1.0 --tile-norm-blocksize 100
```

Outputs:
- `masks/<image>_mask.tif` — uint16 label arrays
- `labelled/<image>_labelled_segmentations.png` — annotated overviews
- `exports/<image>/cell_XXXX.tif` — per-cell square IBA1 crops with background zeroed

## Critical: full-image tiling

**Do not run `model.eval()` directly on the full 2037×2037 image.** Cellpose's internal multi-patch stitching breaks on large inputs — masks collapse to nucleus-sized blobs. Always use `segment_tiled()` from `bens_cellpose_utils.py`, which chunks the image into 384px inner tiles (with 64px context), evaluates each independently, and stitches masks with globally unique IDs.

`tune_parameters.py` deliberately runs at tile size (448px = 384 inner + 64 context × 2) so results transfer directly to the full tiled pipeline. `run_segmentation.py` and `validate_parameters.py` both use `segment_tiled()`.

## Scripts

| Script | Purpose |
|---|---|
| `tune_parameters.py` | Phased parameter-grid search on tile-sized crops → `tune/` |
| `validate_parameters.py` | Sanity-check chosen params on N random full images → `validate/` |
| `run_segmentation.py` | Full pipeline: segment all images, write masks + labelled PNGs + per-cell crops |
| `evaluate_segmentation.py` | Compute per-image QC metrics → `metrics.csv` |
| `visualize_results.py` | Overlay existing masks on raw images → `vis/` |
| `bens_cellpose_utils.py` | Shared utilities: `segment_tiled`, `mask_filter_fixed`, `mask_size_stats`, export/overlay helpers |

## Cellpose v4 API notes

- `model_type` and `channels` are deprecated — omit them.
- Pass images as `(H, W, 3)` with `channel_axis=2`: IBA1 in slot 0, DAPI in slot 1.
- `normalize={'normalize': True, 'tile_norm_blocksize': 100}` for local-contrast normalization; `normalize=True` (plain) falls back to global.
- `CellposeModel(gpu=True)` loads the `cpsam` weights by default.

## Other documentation

- [PARAMETERS.md](PARAMETERS.md) — full reference for every tunable parameter.
- [plans.md](plans.md) — in-progress work and known issues.
- [CLAUDE.md](CLAUDE.md) — instructions for AI coding assistants working in this repo.

## Known issues

- Labelled overview PNGs occasionally fail to render on full images (`save_segmentation_img_w_mask_ns_fixed`). `run_segmentation.py` wraps the call in try/except — watch for `WARNING: labelled image failed:` lines.
- tqdm progress bars don't render under `conda run` (no TTY); per-tile progress uses `print(..., flush=True)` instead.
