"""
Step 2 of the tuning workflow: validate chosen parameters on a few full images.

After picking a combo from tune_parameters.py, run it through segment_tiled() on
a handful of randomly chosen full images to sanity-check before committing to the
full analysis. Saves labelled overviews + prints per-image cell counts and timing.

Usage:
    conda run -n cellpose python -u validate_parameters.py \
        --diameter 150 --cellprob -2.0 --flow 1.0 --pix-filter 500 --tnb 100
"""
import argparse
import sys
import time
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

import tifffile
from pathlib import Path
from cellpose.models import CellposeModel
from bens_cellpose_utils import (mask_filter_fixed, segment_tiled,
                                  save_segmentation_img_w_mask_ns_fixed,
                                  mask_size_stats, combo_label,
                                  apply_config_defaults, maybe_save_config)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', default='images')
    p.add_argument('--output-dir', default='validate')
    p.add_argument('--n-images', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--diameter', type=float, default=150)
    p.add_argument('--cellprob', dest='cellprob_threshold', type=float, default=-2.0)
    p.add_argument('--flow', dest='flow_threshold', type=float, default=1.0)
    p.add_argument('--pix-filter', type=int, default=500)
    p.add_argument('--tnb', dest='tile_norm_blocksize', type=int, default=100)
    p.add_argument('--niter', type=int, default=None)

    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    p.add_argument('--tile-inner-size', type=int, default=384)
    p.add_argument('--tile-context', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--label-fontsize', type=float, default=2)
    apply_config_defaults(p)
    args = p.parse_args()
    maybe_save_config(args)
    return args


def main():
    args = parse_args()
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(img_dir.glob('*.tif'))
    if not tif_files:
        print(f'No .tif files in {img_dir}', flush=True)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.choice(len(tif_files),
                            size=min(args.n_images, len(tif_files)),
                            replace=False)
    chosen = [tif_files[i] for i in chosen_idx]

    params = {
        'diameter': args.diameter,
        'cellprob_threshold': args.cellprob_threshold,
        'flow_threshold': args.flow_threshold,
        'pix_filter': args.pix_filter,
        'tile_norm_blocksize': args.tile_norm_blocksize,
        'niter': args.niter,
    }
    label = combo_label(params)
    print(f'Validating combo: {label}')
    print(f'  {params}')
    print(f'Images ({len(chosen)}, seed={args.seed}):')
    for p in chosen:
        print(f'  - {p.name}')
    print()

    norm_param = (True if args.tile_norm_blocksize == 0
                  else {'normalize': True, 'tile_norm_blocksize': args.tile_norm_blocksize})

    model = CellposeModel(gpu=True)

    t_total = time.time()
    for idx, tif_path in enumerate(chosen, 1):
        t0 = time.time()
        print(f'[{idx}/{len(chosen)}] {tif_path.name}')
        img = tifffile.imread(str(tif_path))  # (2, H, W)
        h, w = img.shape[1], img.shape[2]

        masks = segment_tiled(
            model, img,
            iba1_ch=args.iba1_channel,
            dapi_ch=args.dapi_channel,
            inner_size=args.tile_inner_size,
            context=args.tile_context,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            normalize=norm_param,
            niter=args.niter,
            batch_size=args.batch_size,
        )
        masks = mask_filter_fixed(masks, pix_size=args.pix_filter)
        stats = mask_size_stats(masks)
        elapsed = time.time() - t0

        img_hwc = np.zeros((h, w, 3), dtype=img.dtype)
        img_hwc[:, :, 0] = img[args.iba1_channel]
        img_hwc[:, :, 1] = img[args.dapi_channel]

        overview_name = f'{tif_path.stem}__{label}.tif'
        try:
            out_path = save_segmentation_img_w_mask_ns_fixed(
                img_hwc, masks,
                file_name=overview_name, odir=str(out_dir),
                label_fontsize=args.label_fontsize,
            )
            print(f'  labelled -> {out_path}', flush=True)
        except Exception as e:
            print(f'  WARNING: labelled image failed: {e}', flush=True)

        print(f'  n_cells={stats["n_cells"]}  '
              f'size med={stats["size_median"]:.0f}px '
              f'(min={stats["size_min"]}, max={stats["size_max"]}, '
              f'mean={stats["size_mean"]:.0f})  '
              f'| {elapsed:.0f}s\n')

    print(f'Done in {(time.time() - t_total)/60:.1f} min.')
    print(f'Overviews -> {out_dir}/')


if __name__ == '__main__':
    main()
