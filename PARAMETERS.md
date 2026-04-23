# Cellpose-SAM tuning parameters

Reference sheet for `tune_parameters.py` and `validate_parameters.py`. Each parameter is tagged with an impact tier — higher tiers get more values in the default sweep grid, lower tiers are held fixed unless explicitly varied.

Known-good for this dataset (stitched microglia tilescans, 2037×2037 uint8):
```
diameter=150, cellprob_threshold=-2.0, flow_threshold=1.0,
pix_filter=500, tile_norm_blocksize=100, niter=None
```

---

## High impact (phase 1)

### `diameter`
**Default:** 150
**Phase-1 sweep:** `[75, 150, 225, 300]`
**What it does:** Expected cell diameter in pixels. The model rescales input so cells land at its trained internal size.
**Why wide-ranging:** Scales linearly with acquisition resolution — a 2× magnification or pixel-size change doubles the effective diameter. A dataset-agnostic sweep needs to cover ~0.5× to 2× of known-good.
**Symptoms:**
- Too small → cells fragment into multiple sub-masks; lots of tiny objects.
- Too large → adjacent cells merge; soma + processes collapsed into one big blob.

### `cellprob_threshold`
**Default:** -2.0
**Phase-1 sweep:** `[-3.0, -2.0, -1.0, 0.0]`
**What it does:** Minimum cell-probability for a pixel to be included in a mask. Lower = more permissive, higher = stricter.
**Why wide-ranging:** Controls the recall/precision tradeoff directly; the single biggest lever for cell count. Microglia often need negative thresholds because their processes have low cellprob signal.
**Symptoms:**
- Too low → background/noise pulled into masks; over-segmentation; very high cell count.
- Too high → only bright somas detected; processes truncated or missed entirely.

---

## Medium impact (phase 2)

### `flow_threshold`
**Default:** 1.0
**Phase-2 sweep:** `[0.4, 1.0, 1.5, 2.0]`
**What it does:** Max allowable error between predicted flow and mask-derived flow. Rejects mask candidates whose flows don't match the model's prediction.
**Symptoms:**
- Too low (strict) → filters out valid cells with slightly noisy flows; lower count, missing real microglia.
- Too high (permissive, >3) → keeps low-quality mask candidates; jagged/irregular mask boundaries.

### `pix_filter`
**Default:** 500
**Phase-2 sweep:** `[250, 500, 1000, 2000]`
**What it does:** Post-processing — removes mask labels whose pixel area is below this threshold. Not a cellpose eval arg; applied via `mask_filter_fixed`.
**Symptoms:**
- Too low → tiny noise blobs and mask fragments kept as "cells", inflating count.
- Too high → valid small/faint microglia deleted, under-counting.

---

## Low impact (phase 3, rarely needed)

### `tile_norm_blocksize`
**Default:** 100
**Phase-3 sweep:** `[0, 50, 100, 200]` (0 = global normalization)
**What it does:** Controls local-contrast normalization. Smaller blocks give stronger local equalization; 0 disables local norm and uses global percentiles.
**Why low impact here:** Dataset has relatively uniform illumination after stitching — global and local norm produce similar results. Can matter on datasets with large intensity gradients.
**Symptoms:**
- Too small → faint cells artificially brightened, but background noise also amplified.
- Too large / 0 → dim regions under-segmented when global contrast is dominated by bright areas.

### `niter`
**Default:** None (auto)
**Phase-3 sweep:** `[None, 200, 500, 1000]`
**What it does:** Number of flow-dynamics iterations for mask construction. Auto-selected from diameter by default.
**Symptoms:**
- Too few → elongated cells (microglia processes!) truncated mid-process.
- Too many → minor accuracy gains at higher cost; rarely changes results meaningfully above ~500.

---

## Phased tuning workflow

1. **Phase 1** (`tune_parameters.py --phase 1`): Grid over diameter × cellprob_threshold, others fixed. Pick best combo by inspecting labelled PNGs + summary.csv (cell count + median mask size stable across crops).
2. **Phase 2** (`tune_parameters.py --phase 2 --diameter <P1winner> --cellprob <P1winner>`): Grid over flow_threshold × pix_filter. Usually only small refinements.
3. **Phase 3** (rare, only if phases 1/2 look off): Sweep tile_norm_blocksize / niter.
4. **Validate** (`validate_parameters.py --diameter ... --cellprob ...`): Run the chosen params on 3 random full images via `segment_tiled()`. Check labelled overviews look right before launching the full analysis.
