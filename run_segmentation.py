"""
Run cpsam segmentation on all images and save masks + labelled overview PNGs.

Best parameters found for this dataset:
    diameter=150, flow_threshold=1.0, cellprob_threshold=-2.0, tile_norm_blocksize=100

Usage:
    conda run -n cellpose python run_segmentation.py
    conda run -n cellpose python run_segmentation.py --cellprob -1.5
"""
import argparse
import sys
import time
import numpy as np

# Force line-buffered stdout so prints appear immediately under conda run
sys.stdout.reconfigure(line_buffering=True)
import tifffile
from pathlib import Path
from cellpose.models import CellposeModel
from bens_cellpose_utils import (mask_filter_fixed, save_segmentation_img_w_mask_ns_fixed,
                                  export_segmented_images, segment_tiled)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--diameter', type=float, default=150)
    p.add_argument('--flow', dest='flow_threshold', type=float, default=1.0)
    p.add_argument('--cellprob', dest='cellprob_threshold', type=float, default=-2.0)
    p.add_argument('--tnb', dest='tile_norm_blocksize', type=int, default=100)
    p.add_argument('--pix-filter', type=int, default=500,
                   help='Remove masks smaller than this many pixels')
    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    p.add_argument('--image-dir', default='images')
    p.add_argument('--mask-dir', default='masks')
    p.add_argument('--labelled-dir', default='labelled',
                   help='Output directory for annotated overview PNGs')
    p.add_argument('--exports-dir', default='exports',
                   help='Output directory for per-cell tif crops (one subdir per image)')
    p.add_argument('--cell-padding', type=int, default=20,
                   help='Pixels of padding around each cell bounding box')
    p.add_argument('--label-fontsize', type=float, default=2,
                   help='Font size for cell ID labels on overview images')
    p.add_argument('--tile-inner-size', type=int, default=384,
                   help='Each tile contributes this many pixels to the output (must match crop size that worked during tuning)')
    p.add_argument('--tile-context', type=int, default=64,
                   help='Extra context pixels around each tile for the model')
    p.add_argument('--batch-size', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    img_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    labelled_dir = Path(args.labelled_dir)
    exports_dir = Path(args.exports_dir)
    mask_dir.mkdir(exist_ok=True)
    labelled_dir.mkdir(exist_ok=True)
    exports_dir.mkdir(exist_ok=True)

    norm_param = {'normalize': True, 'tile_norm_blocksize': args.tile_norm_blocksize}
    model = CellposeModel(gpu=True)

    print(f'diameter={args.diameter}  flow_threshold={args.flow_threshold}  '
          f'cellprob_threshold={args.cellprob_threshold}  '
          f'tile_norm_blocksize={args.tile_norm_blocksize}  '
          f'pix_filter={args.pix_filter}')

    tif_files = sorted(img_dir.glob('*.tif'))
    n_images = len(tif_files)
    print(f'Found {n_images} images\n')

    cell_counts = []
    run_start = time.time()
    for img_idx, tif_path in enumerate(tif_files, 1):
        t0 = time.time()
        print(f'[{img_idx}/{n_images}] {tif_path.name}')
        img = tifffile.imread(str(tif_path))       # (2, H, W)
        h, w = img.shape[1], img.shape[2]

        img_hwc = np.zeros((h, w, 3), dtype=img.dtype)
        img_hwc[:, :, 0] = img[args.iba1_channel]  # IBA1 → slot 0
        img_hwc[:, :, 1] = img[args.dapi_channel]  # DAPI → slot 1

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
            batch_size=args.batch_size,
        )

        masks = mask_filter_fixed(masks, pix_size=args.pix_filter)

        n_cells = int(masks.max())
        cell_counts.append(n_cells)
        elapsed = time.time() - t0
        done = img_idx
        avg_s = (time.time() - run_start) / done
        eta_s = avg_s * (n_images - done)
        eta_min = eta_s / 60
        print(f'  {n_cells} cells  |  {elapsed:.0f}s this image  |  ETA {eta_min:.1f} min')

        tifffile.imwrite(
            str(mask_dir / f'{tif_path.stem}_mask.tif'),
            masks.astype(np.uint16),
        )

        print(f'  saving labelled overview...', flush=True)
        try:
            out_path = save_segmentation_img_w_mask_ns_fixed(
                img_hwc,
                masks,
                file_name=tif_path,
                odir=str(labelled_dir),
                label_fontsize=args.label_fontsize,
            )
            print(f'  labelled -> {out_path}', flush=True)
        except Exception as e:
            print(f'  WARNING: labelled image failed: {e}', flush=True)

        print(f'  exporting cell crops...', flush=True)
        cell_export_dir = exports_dir / tif_path.stem
        n_exported = export_segmented_images(
            img,
            masks,
            odir=cell_export_dir,
            iba1_channel=args.iba1_channel,
            padding=args.cell_padding,
            image_stem=tif_path.stem,
        )
        print(f'  -> {n_exported} cell crops exported to {cell_export_dir}/', flush=True)

    counts = np.array(cell_counts)
    print(f'\nDone.')
    print(f'Masks   -> {mask_dir}/')
    print(f'Labels  -> {labelled_dir}/')
    print(f'Exports -> {exports_dir}/')
    print(f'Cell counts: mean={counts.mean():.0f}, min={counts.min()}, max={counts.max()}')


if __name__ == '__main__':
    main()
