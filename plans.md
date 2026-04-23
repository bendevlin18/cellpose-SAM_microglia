# Plans

## Parameter-finding tool

**Goal:** A clean, reusable tool for finding optimal segmentation parameters at the start of each new experiment, since model behavior varies significantly across image datasets.

**Background:** The current workflow required several manual iterations (parameter_sweep.py → parameter_sweep2.py → generalization_sweep.py) to land on good parameters. A good tool would compress this into a single well-designed script.

**Desired behavior:**
- Accept a reference image (or auto-pick one from the dataset)
- Run a configurable parameter grid (flow_threshold, cellprob_threshold, diameter, tile_norm_blocksize, pix_filter, niter)
- Save one annotated output image per parameter combination with the parameter values embedded in the filename (so you can flip through them quickly in a file browser)
- Print a summary table of cell counts per combo
- Optionally auto-suggest the combo with the "most reasonable" cell count (e.g. closest to expected density)

**Starting point:** `preview_cellpose_params()` in `bens_cellpose_utils.py` is the intended basis — it already does grid iteration and filename labeling. It needs:
- Channel handling updated for cellpose v4 API (no `channels=` arg, use `channel_axis=2`)
- Must use `segment_tiled()` instead of direct `model.eval()` — direct eval on large images breaks (see CLAUDE.md)
- Integration with `save_segmentation_img_w_mask_ns_fixed` for output
- A clean CLI wrapper so it can be run standalone without editing code
- Default param grid tuned for microglia: `ft=1.0, cp=-2.0, d=150, tnb=100`

## Exports: include source image name in per-cell filenames

Currently `export_segmented_images` writes `exports/<image_stem>/cell_XXXX.tif`, so the source image is only encoded in the directory. Change the filenames themselves to include the image stem (e.g. `<image_stem>_cell_XXXX.tif`) so the files remain identifiable after being copied out of their subdirectory or flattened into a single pool for downstream analysis.

- Update `export_segmented_images` to accept an `image_stem` (or derive from a passed path) and prepend it to each tif filename.
- Update `run_segmentation.py` to pass the stem through.
- Keep the per-image subdirectory structure — this is additive naming, not a layout change.

## Labelled image bug

`save_segmentation_img_w_mask_ns_fixed` was not producing output during the full pipeline run. The cause is unknown — `run_segmentation.py` now has try/except around the call that prints the error message. Next run will reveal the actual exception. Likely candidates:
- matplotlib backend issue (no `matplotlib.use('Agg')` in `bens_cellpose_utils.py`)
- Memory issue rendering a 2037×2037 figure at 300 DPI with hundreds of text labels
- Silent path/permission issue

Fix attempt pending — need to see the error message from the next run first.
