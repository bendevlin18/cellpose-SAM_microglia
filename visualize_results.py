"""
Visualize segmentation masks overlaid on raw images.

Produces a 3-panel PNG per image: DAPI | IBA1 | cpsam mask overlay on IBA1.

Usage:
    conda run -n cellpose python visualize_results.py [--iba1-channel 1] [--dapi-channel 0]
"""
import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose.plot import mask_overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', default='images')
    p.add_argument('--mask-dir', default='masks')
    p.add_argument('--output-dir', default='vis')
    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    return p.parse_args()


def norm8(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def main():
    args = parse_args()
    img_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    mask_files = sorted(mask_dir.glob('*_mask.tif'))
    if not mask_files:
        print(f'No mask files found in {mask_dir}. Run run_segmentation.py first.')
        return

    print(f'Generating overlays for {len(mask_files)} images...')
    for mask_path in mask_files:
        stem = mask_path.stem.replace('_mask', '')
        img_path = img_dir / f'{stem}.tif'
        if not img_path.exists():
            print(f'  WARNING: no source image for {mask_path.name}')
            continue

        img = tifffile.imread(str(img_path))    # (2, H, W)
        masks = tifffile.imread(str(mask_path))  # (H, W) uint16
        n_cells = int(masks.max())

        dapi = norm8(img[args.dapi_channel])
        iba1 = norm8(img[args.iba1_channel])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{stem}  |  {n_cells} cells', fontsize=11)

        axes[0].imshow(dapi, cmap='gray')
        axes[0].set_title(f'Ch{args.dapi_channel} — DAPI')
        axes[0].axis('off')

        axes[1].imshow(iba1, cmap='gray')
        axes[1].set_title(f'Ch{args.iba1_channel} — IBA1')
        axes[1].axis('off')

        axes[2].imshow(mask_overlay(iba1, masks))
        axes[2].set_title(f'cpsam mask ({n_cells} cells)')
        axes[2].axis('off')

        out_path = out_dir / f'{stem}_overlay.png'
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  {stem}: {n_cells} cells')

    print(f'\nSaved to {out_dir}/')


if __name__ == '__main__':
    main()
