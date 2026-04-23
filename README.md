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
- `tune/phase<N>/<combo>.png` — one contact-sheet PNG per parameter combo, showing every sampled crop side-by-side under that combo (grid rows default to one-per-image)
- `tune/phase<N>/all_combos.pdf` — same contact sheets collected into a single multi-page PDF for easy sequential browsing. Opt out with `--no-pdf`.
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

## CLI reference

All scripts use `argparse`, so `--help` on any of them prints the full list. The tables below cover flags that affect behaviour day-to-day.

### Common flags

Shared across most scripts with identical meaning.

| Flag | Default | Scripts | Description |
|---|---|---|---|
| `--image-dir` | `images` | tune, validate, run, visualize | Folder containing input `.tif` files |
| `--iba1-channel` | `1` | tune, validate, run, visualize | IBA1 channel index in source images |
| `--dapi-channel` | `0` | tune, validate, run, visualize | DAPI channel index in source images |
| `--label-fontsize` | `2` | tune, validate, run | Font size for cell-ID labels on overview images |
| `--tile-inner-size` | `384` | validate, run | Pixels per tile that end up in the stitched output |
| `--tile-context` | `64` | validate, run | Extra context pixels around each tile |
| `--batch-size` | `8` | validate, run | Tiles per model forward pass |
| `--seed` | `0` | tune, validate | RNG seed for image/region selection |

Parameter flags (`--diameter`, `--cellprob` / `--cellprob-threshold`, `--flow` / `--flow-threshold`, `--pix-filter`, `--tnb` / `--tile-norm-blocksize`, `--niter`) are documented in [PARAMETERS.md](PARAMETERS.md). Note that `tune` and `validate` use the short names (`--cellprob`, `--flow`, `--tnb`) while `run_segmentation` uses the long names (`--cellprob-threshold`, `--flow-threshold`, `--tile-norm-blocksize`).

### `tune_parameters.py`

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `tune` | Root output dir; previews written to `<output-dir>/phase<N>/` |
| `--phase` | `1` | Preset grid to run: 1 (`diameter × cellprob`), 2 (`flow × pix_filter`), 3 (`tnb × niter`) |
| `--n-images` | `4` | Images sampled from `--image-dir` |
| `--regions-per-image` | `2` | Region centres chosen per image |
| `--crops-per-region` | `3` | Jittered crops around each region centre |
| `--crop-size` | `448` | Tile size in px (matches `segment_tiled`: 384 inner + 64 context × 2) |
| `--jitter` | `200` | Max random offset (px) of crops around a region centre |

Parameter overrides accept comma-separated lists (`--diameter 100,150,200`) and replace that parameter's values in the phase grid.

Diameter sweeps have two extra shortcuts (precedence: `--diameter` > `--diameter-center` > `--diameter-preset`):

| Flag | Default | Description |
|---|---|---|
| `--diameter-preset` | — | Named list: `small=[25,50,75,100]`, `medium=[75,112,150,188,225]`, `large=[200,300,400,500]`, `wide=[50,150,250,350,450]` |
| `--diameter-center` | — | Centre value; auto-generates `linspace(center*(1-spread), center*(1+spread), n)` — e.g. `--diameter-center 150` → `[75, 112.5, 150, 187.5, 225]` |
| `--diameter-n` | `5` | Number of values for `--diameter-center` |
| `--diameter-spread` | `0.5` | Fractional half-width for `--diameter-center` |
| `--no-pdf` | — | Skip writing `all_combos.pdf` (PNG contact sheets are still written) |

### `validate_parameters.py`

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `validate` | Where labelled full-image overviews are written |
| `--n-images` | `3` | Full images sampled (respects `--seed`) |
| `--diameter` | `150` | See PARAMETERS.md |
| `--cellprob` | `-2.0` | `cellprob_threshold` |
| `--flow` | `1.0` | `flow_threshold` |
| `--pix-filter` | `500` | Minimum mask size in px |
| `--tnb` | `100` | `tile_norm_blocksize` (0 = global normalization) |
| `--niter` | `None` | Dynamics iterations; `None` = model default |

### `run_segmentation.py`

| Flag | Default | Description |
|---|---|---|
| `--diameter` | `150` | See PARAMETERS.md |
| `--cellprob-threshold` | `-2.0` | Cell probability threshold |
| `--flow-threshold` | `1.0` | Flow error threshold |
| `--tile-norm-blocksize` | `100` | Local-contrast block size (0 = global) |
| `--pix-filter` | `500` | Minimum mask size in px |
| `--mask-dir` | `masks` | Output dir for `_mask.tif` files |
| `--labelled-dir` | `labelled` | Output dir for annotated overview PNGs |
| `--exports-dir` | `exports` | Output dir for per-cell IBA1 crops |
| `--cell-padding` | `20` | Padding (px) around each cell bounding box in per-cell crops |

### `visualize_results.py`

| Flag | Default | Description |
|---|---|---|
| `--mask-dir` | `masks` | Directory with `*_mask.tif` files |
| `--output-dir` | `vis` | Where 3-panel overlay PNGs are written |

Also accepts `--image-dir`, `--iba1-channel`, `--dapi-channel` from the common flags.

### `evaluate_segmentation.py`

| Flag | Default | Description |
|---|---|---|
| `--mask-dir` | `masks` | Directory with `*_mask.tif` files |
| `--csv` | `metrics.csv` | CSV path for per-image QC metrics |

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
