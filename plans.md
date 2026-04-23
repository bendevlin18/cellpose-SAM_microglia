# Plans

## CNN filter on exported cells

**Goal:** Add a post-processing step that scores each per-cell crop in `exports/` so downstream analysis can restrict to fully segmented, high-quality microglia and drop off-targets (dim/low-signal cells, bare processes, over- or under-segmented fragments).

**Inputs:** `exports/<image_stem>/<image_stem>_cell_XXXX.tif` produced by `run_segmentation.py`.

**Output:** per-cell scores (e.g. `exports/scores.csv` with columns `image, cell_id, score, keep`), or a split into `exports_good/` and `exports_off_target/` subdirs — decide during implementation.

**Open questions:**
- How to build the training set: hand-curate a few hundred crops into good / off-target folders? Start with a small labelled subset and iterate?
- Architecture: small CNN from scratch (ResNet-ish), or fine-tune a pretrained backbone on these ~100×100 grayscale IBA1 crops?
- Binary classifier vs. continuous score? A score lets you pick a threshold post-hoc.
- Integration: standalone `classify_exports.py` that consumes `exports/` and writes scores, or a flag on `run_segmentation.py` that runs inline after export?
