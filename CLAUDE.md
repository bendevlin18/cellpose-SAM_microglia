# cellpose-sam microglia segmentation

## Environment

All Python scripts must be run under the `cellpose` conda environment:

```bash
conda run -n cellpose python -u <script.py>
```

Always use `-u` (unbuffered). Without it, `conda run` pipes stdout and Python buffers all output silently until the process ends — nothing prints in real time. Never call `python` directly — always prefix with `conda run -n cellpose`.

## Project overview

Goal: segment microglia in 2-channel fluorescence microscopy images (stitched tilescans from insula cortex).

- **Images**: `images/` — 22 stitched `.tif` files, all (2, 2037×2037) uint8
  - Channel 0: DAPI (nuclei) — confirmed
  - Channel 1: IBA1 (microglia marker) — confirmed
- **Model**: `cpsam` (CellPose-SAM v4) — only available model; `CellposeModel(gpu=True)` loads it by default
- **GPU**: NVIDIA RTX 3060, CUDA 12.6, torch 2.9.1

## Key scripts

| Script | Purpose |
|--------|---------|
| `run_segmentation.py` | Full pipeline: segment all images, save masks + labelled PNGs + per-cell exports |
| `parameter_sweep.py` | Initial grid-search (diameter × cellprob_threshold) on a 512px crop |
| `parameter_sweep2.py` | Focused sweep (flow_threshold × tile_norm_blocksize) after fixing diameter |
| `generalization_sweep.py` | Test chosen params across multiple images, sweep cellprob_threshold |
| `diagnose_scale.py` | Compare crop vs full-image vs tiled segmentation to debug scale issues |
| `evaluate_segmentation.py` | Per-image QC metrics → metrics.csv |
| `visualize_results.py` | Overlay existing masks on raw images → vis/ |
| `bens_cellpose_utils.py` | Core utility functions (tiling, masking, export, labelling) |

## Best parameters (validated on this dataset)

```
diameter=150, flow_threshold=1.0, cellprob_threshold=-2.0, tile_norm_blocksize=100, pix_filter=500
```

## Critical: full-image tiling

**Do NOT run the model on the full 2037×2037 image directly.** The model's internal multi-patch stitching breaks down on large images — masks collapse to nucleus-sized blobs. Always use `segment_tiled()` from `bens_cellpose_utils.py`, which tiles the image into 384px inner chunks (448px with context) and stitches masks with globally unique IDs.

```python
masks = segment_tiled(model, img_chw, iba1_ch=1, dapi_ch=0,
                      inner_size=384, context=64, **eval_kwargs)
```

## Cellpose v4 API notes

- `model_type` and `channels` args are **deprecated** — omit them
- Pass images as `(H, W, 3)` with `channel_axis=2`; IBA1 in slot 0, DAPI in slot 1
- `normalize={'normalize': True, 'tile_norm_blocksize': 100}` for local contrast normalization

## Output structure (from run_segmentation.py)

```
masks/      *_mask.tif          — uint16 label arrays
labelled/   *_labelled_segmentations.png  — annotated overview PNGs (IBA1 + mask overlay + cell ID labels)
exports/    <image_stem>/<image_stem>_cell_XXXX.tif   — per-cell square IBA1 crops, background zeroed, 20px padding
```

## Known issues / in-progress

- `labelled/` PNGs may silently fail — run_segmentation.py now wraps the call in try/except and prints the error. Check output for `WARNING: labelled image failed:` lines.
- `export_segmented_images` is confirmed working (smoke tested).
- tqdm does not render in `conda run` (no TTY); replaced with `print(..., flush=True)` per-tile progress.
