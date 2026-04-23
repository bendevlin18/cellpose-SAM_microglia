"""
Focused sweep fixing d=150, cp=0.0, 2ch input — varying flow_threshold and
tile normalization block size.

flow_threshold: higher = more permissive toward irregular/branched shapes
tile_norm_blocksize: normalizes in local windows to brighten dim process regions

Usage:
    conda run -n cellpose python parameter_sweep2.py
    conda run -n cellpose python parameter_sweep2.py --diameter 100  # try other diameters too
"""
import argparse
import csv
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose.models import CellposeModel
from cellpose.plot import mask_overlay


FLOW_THRESHOLDS = [0.4, 0.6, 0.8, 1.0]
# 0 = global normalization (default), otherwise local tile window in pixels
TILE_NORM_BLOCKSIZES = [0, 50, 100, 200]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='images/Stitch_1B3_insula_A01_G001.tif')
    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    p.add_argument('--diameter', type=float, default=150)
    p.add_argument('--cellprob-threshold', type=float, default=0.0)
    p.add_argument('--crop-size', type=int, default=512)
    p.add_argument('--output-dir', default='sweep')
    return p.parse_args()


def norm8(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def center_crop(arr_chw, size):
    h, w = arr_chw.shape[1], arr_chw.shape[2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr_chw[:, y0:y0+size, x0:x0+size]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    img = tifffile.imread(args.image)
    img = center_crop(img, args.crop_size)
    h, w = img.shape[1], img.shape[2]

    img_hwc = np.zeros((h, w, 3), dtype=img.dtype)
    img_hwc[:, :, 0] = img[args.iba1_channel]
    img_hwc[:, :, 1] = img[args.dapi_channel]
    bg = norm8(img[args.iba1_channel])

    print(f'Fixed: diameter={args.diameter}, cellprob_threshold={args.cellprob_threshold}')
    print(f'Sweeping flow_threshold × tile_norm_blocksize\n')

    model = CellposeModel(gpu=True)

    n_rows = len(FLOW_THRESHOLDS)
    n_cols = len(TILE_NORM_BLOCKSIZES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(
        f'd={args.diameter}  cp={args.cellprob_threshold}  2ch (IBA1+DAPI)\n'
        f'rows=flow_threshold  cols=tile_norm_blocksize (0=global)',
        fontsize=11,
    )

    results = []
    for i, ft in enumerate(FLOW_THRESHOLDS):
        for j, tnb in enumerate(TILE_NORM_BLOCKSIZES):
            norm_param = True if tnb == 0 else {'normalize': True, 'tile_norm_blocksize': tnb}
            masks, _, _ = model.eval(
                img_hwc,
                diameter=args.diameter,
                channel_axis=2,
                flow_threshold=ft,
                cellprob_threshold=args.cellprob_threshold,
                normalize=norm_param,
            )
            n_cells = int(masks.max())
            results.append((ft, tnb, n_cells))
            print(f'  flow={ft}, tile_norm={tnb:>3}: {n_cells} cells')

            ax = axes[i][j]
            ax.imshow(mask_overlay(bg, masks))
            ax.set_title(f'ft={ft} tnb={tnb}\n{n_cells} cells', fontsize=8)
            ax.axis('off')

    out_path = out_dir / f'sweep2_d{int(args.diameter)}.png'
    fig.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nGrid: {out_path}')

    csv_path = out_dir / 'sweep2_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['diameter', 'flow_threshold', 'tile_norm_blocksize', 'n_cells'])
        w.writerows([(args.diameter, ft, tnb, n) for ft, tnb, n in results])
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    main()
