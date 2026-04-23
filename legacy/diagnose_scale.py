"""
Compare segmentation: 512 crop vs full image vs full image (tiled).

Usage:
    conda run -n cellpose python diagnose_scale.py
"""
import argparse
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose.models import CellposeModel
from cellpose.plot import mask_overlay
from bens_cellpose_utils import mask_filter_fixed, segment_tiled


DIAMETER = 150
FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -2.0
PIX_FILTER = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='images/Stitch_1B3_insula_A01_G001.tif')
    p.add_argument('--iba1-channel', type=int, default=1)
    p.add_argument('--dapi-channel', type=int, default=0)
    p.add_argument('--tile-norm-blocksize', type=int, default=100)
    p.add_argument('--crop-size', type=int, default=512)
    p.add_argument('--output-dir', default='sweep')
    return p.parse_args()


def norm8(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)


def build_hwc(iba1_hw, dapi_hw):
    h, w = iba1_hw.shape
    out = np.zeros((h, w, 3), dtype=iba1_hw.dtype)
    out[:, :, 0] = iba1_hw
    out[:, :, 1] = dapi_hw
    return out


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    norm_param = {'normalize': True, 'tile_norm_blocksize': args.tile_norm_blocksize}

    img = tifffile.imread(args.image)   # (2, H, W)
    H, W = img.shape[1], img.shape[2]
    cs = args.crop_size
    y0, x0 = (H - cs) // 2, (W - cs) // 2

    iba1 = img[args.iba1_channel]
    dapi = img[args.dapi_channel]

    model = CellposeModel(gpu=True)
    eval_kwargs = dict(diameter=DIAMETER, flow_threshold=FLOW_THRESHOLD,
                       cellprob_threshold=CELLPROB_THRESHOLD, normalize=norm_param)

    print('Running crop...')
    crop_hwc = build_hwc(iba1[y0:y0+cs, x0:x0+cs], dapi[y0:y0+cs, x0:x0+cs])
    crop_masks, _, _ = model.eval(crop_hwc, channel_axis=2, **eval_kwargs)
    crop_masks = mask_filter_fixed(crop_masks, PIX_FILTER)
    print(f'  Crop: {int(crop_masks.max())} cells')

    print('Running full image (no tiling)...')
    full_hwc = build_hwc(iba1, dapi)
    full_masks, _, _ = model.eval(full_hwc, channel_axis=2, **eval_kwargs)
    full_masks = mask_filter_fixed(full_masks, PIX_FILTER)
    print(f'  Full: {int(full_masks.max())} cells')

    print('Running full image (tiled)...')
    tiled_masks = segment_tiled(model, img,
                                iba1_ch=args.iba1_channel,
                                dapi_ch=args.dapi_channel,
                                inner_size=384, context=64,
                                **eval_kwargs)
    tiled_masks = mask_filter_fixed(tiled_masks, PIX_FILTER)
    print(f'  Tiled: {int(tiled_masks.max())} cells')

    def make_overlay(masks_full, iba1_bg, gy0, gx0, size):
        region = masks_full[gy0:gy0+size, gx0:gx0+size]
        bg = norm8(iba1_bg[gy0:gy0+size, gx0:gx0+size])
        return mask_overlay(bg, region)

    bg_crop = norm8(iba1[y0:y0+cs, x0:x0+cs])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{Path(args.image).stem} — same {cs}x{cs} central region shown in all panels',
                 fontsize=10)

    axes[0].imshow(mask_overlay(bg_crop, crop_masks))
    axes[0].set_title(f'Crop ({int(crop_masks.max())} cells)', fontsize=10)

    axes[1].imshow(make_overlay(full_masks, iba1, y0, x0, cs))
    axes[1].set_title(f'Full image, no tiling ({int(full_masks.max())} total)', fontsize=10)

    axes[2].imshow(make_overlay(tiled_masks, iba1, y0, x0, cs))
    axes[2].set_title(f'Full image, tiled ({int(tiled_masks.max())} total)', fontsize=10)

    for ax in axes:
        ax.axis('off')

    out_path = out_dir / f'diagnose_{Path(args.image).stem}.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
