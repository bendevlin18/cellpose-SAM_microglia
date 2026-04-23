import numpy as np
import tifffile
from cellpose import plot
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import itertools
from scipy.ndimage import center_of_mass, find_objects


def segment_tiled(model, img_chw, iba1_ch, dapi_ch,
                  inner_size=384, context=64, **eval_kwargs):
    """Segment a large image by tiling it into inner_size x inner_size chunks.

    Each tile is expanded by `context` pixels on each side so the model has
    surrounding context, but only the inner region is written to the output mask.
    This ensures every tile is processed identically to a standalone small crop,
    avoiding the multi-patch stitching artefacts cellpose produces on large images.

    Returns a (H, W) uint16 mask array with globally unique cell IDs.
    """
    H, W = img_chw.shape[1], img_chw.shape[2]
    full_masks = np.zeros((H, W), dtype=np.uint32)
    global_id = 0

    def tile_ranges(total, inner, ctx):
        for gy0 in range(0, total, inner):
            gy1 = min(gy0 + inner, total)
            ey0 = max(0, gy0 - ctx)
            ey1 = min(total, gy1 + ctx)
            iy0 = gy0 - ey0
            iy1 = gy1 - ey0
            yield gy0, gy1, ey0, ey1, iy0, iy1

    y_tiles = list(tile_ranges(H, inner_size, context))
    x_tiles = list(tile_ranges(W, inner_size, context))
    n_rows, n_cols = len(y_tiles), len(x_tiles)
    print(f'  Tiling: {n_rows}x{n_cols} = {n_rows * n_cols} tiles '
          f'(inner={inner_size}px, context={context}px)', flush=True)

    for ri, (gy0, gy1, ey0, ey1, iy0, iy1) in enumerate(y_tiles):
        for ci, (gx0, gx1, ex0, ex1, ix0, ix1) in enumerate(x_tiles):
            th, tw = ey1 - ey0, ex1 - ex0
            tile_hwc = np.zeros((th, tw, 3), dtype=img_chw.dtype)
            tile_hwc[:, :, 0] = img_chw[iba1_ch, ey0:ey1, ex0:ex1]
            tile_hwc[:, :, 1] = img_chw[dapi_ch, ey0:ey1, ex0:ex1]

            masks, _, _ = model.eval(tile_hwc, channel_axis=2, **eval_kwargs)

            inner = masks[iy0:iy1, ix0:ix1]
            cell_ids = np.unique(inner)
            cell_ids = cell_ids[cell_ids > 0]

            if len(cell_ids) > 0:
                lut = np.zeros(int(masks.max()) + 1, dtype=np.uint32)
                for cid in cell_ids:
                    global_id += 1
                    lut[cid] = global_id
                full_masks[gy0:gy1, gx0:gx1] = lut[inner]

        print(f'  row {ri + 1}/{n_rows} done  ({global_id} cells so far)', flush=True)

    return full_masks.astype(np.uint16)


def export_segmented_images(img_chw, masks, odir, iba1_channel=1, padding=20):
    """Export each segmented cell as a square-cropped tif of the IBA1 channel.

    img_chw : (C, H, W) uint8 array
    masks   : (H, W) uint16 label array
    """
    odir = Path(odir)
    odir.mkdir(parents=True, exist_ok=True)

    iba1 = img_chw[iba1_channel]
    H, W = iba1.shape

    # find_objects gets all bounding boxes in one pass — avoids per-cell full-array scan
    bboxes = find_objects(masks)
    n_exported = 0

    for cell_id, bbox in enumerate(bboxes, 1):
        if bbox is None:
            continue

        sy, sx = bbox
        y_min, y_max = sy.start, sy.stop - 1
        x_min, x_max = sx.start, sx.stop - 1

        cell_mask = masks[sy, sx] == cell_id

        y0 = max(0, y_min - padding)
        y1 = min(H, y_max + padding + 1)
        x0 = max(0, x_min - padding)
        x1 = min(W, x_max + padding + 1)

        h_crop = y1 - y0
        w_crop = x1 - x0
        size = max(h_crop, w_crop)

        square_img  = np.zeros((size, size), dtype=iba1.dtype)
        square_mask = np.zeros((size, size), dtype=bool)

        y_off = (size - h_crop) // 2
        x_off = (size - w_crop) // 2

        # Place cell_mask into the padded crop region
        padded_mask = np.zeros((h_crop, w_crop), dtype=bool)
        by0 = sy.start - y0
        bx0 = sx.start - x0
        padded_mask[by0:by0 + (sy.stop - sy.start),
                    bx0:bx0 + (sx.stop - sx.start)] = cell_mask

        square_img [y_off:y_off + h_crop, x_off:x_off + w_crop] = iba1[y0:y1, x0:x1]
        square_mask[y_off:y_off + h_crop, x_off:x_off + w_crop] = padded_mask
        square_img[~square_mask] = 0

        tifffile.imwrite(str(odir / f'cell_{cell_id:04d}.tif'), square_img)
        n_exported += 1

    return n_exported


def mask_filter_fixed(masks, pix_size):
    """Remove mask labels whose area (in pixels) is below pix_size."""
    counts = np.bincount(masks.ravel())
    small_labels = np.where(counts < pix_size)[0]
    small_labels = small_labels[small_labels > 0]
    lut = np.arange(counts.size, dtype=masks.dtype)
    lut[small_labels] = 0
    return lut[masks]


def mask_size_stats(masks):
    """Return per-label size stats (in pixels) for a label array.

    Returns dict with n_cells, size_min, size_max, size_mean, size_median.
    Zero/missing labels are ignored. An empty mask returns zeros.
    """
    counts = np.bincount(masks.ravel())
    sizes = counts[1:]
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return {'n_cells': 0, 'size_min': 0, 'size_max': 0,
                'size_mean': 0.0, 'size_median': 0.0}
    return {
        'n_cells': int(len(sizes)),
        'size_min': int(sizes.min()),
        'size_max': int(sizes.max()),
        'size_mean': float(sizes.mean()),
        'size_median': float(np.median(sizes)),
    }


def save_segmentation_img_w_mask_ns_fixed(img, maski, file_name, odir, label_fontsize=2):
    """Save a full annotated overlay PNG with each cell labeled by its mask ID.

    img: numpy array, either (C, H, W) or (H, W, C)
    maski: (H, W) uint16 mask array
    """
    img0 = img.copy()

    # Normalize to (H, W, C)
    if img0.ndim == 3 and img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))

    if img0.max() <= 50.0:
        img0 = np.uint8(np.clip(img0, 0, 1) * 255)

    if img0.ndim == 2:
        img0 = np.stack([img0, img0, img0], axis=-1)
    elif img0.shape[2] < 3:
        pad = np.zeros((*img0.shape[:2], 3 - img0.shape[2]), dtype=img0.dtype)
        img0 = np.concatenate([img0, pad], axis=2)

    overlay = plot.mask_overlay(img0[:, :, 0], maski)

    H, W = maski.shape
    my_dpi = 300
    fig = plt.figure(figsize=(W / my_dpi, H / my_dpi), dpi=my_dpi)
    plt.axis("off")
    plt.imshow(overlay)

    cell_ids = np.unique(maski)
    cell_ids = cell_ids[cell_ids > 0].tolist()
    if cell_ids:
        # center_of_mass computes all centroids in a single array pass
        centroids = center_of_mass(maski > 0, maski, cell_ids)
        for cell_id, (cy, cx) in zip(cell_ids, centroids):
            plt.text(int(cx), int(cy), str(cell_id),
                     color="white", fontsize=label_fontsize)

    stem = Path(file_name).stem
    out_path = os.path.join(odir, stem + "_labelled_segmentations.png")
    plt.savefig(out_path, dpi=my_dpi, bbox_inches="tight")
    plt.close()
    return out_path


def combo_label(params):
    """Canonical filename-safe label for a parameter combo."""
    niter = params.get('niter')
    niter_str = 'auto' if niter is None else str(niter)
    return (
        f"d{params['diameter']}"
        f"_cp{params['cellprob_threshold']}"
        f"_ft{params['flow_threshold']}"
        f"_pf{params['pix_filter']}"
        f"_tnb{params['tile_norm_blocksize']}"
        f"_ni{niter_str}"
    )


def combo_title(params):
    """Human-readable label for a parameter combo (use in plot titles, not filenames)."""
    niter = params.get('niter')
    niter_str = 'auto' if niter is None else str(niter)
    return (
        f"diameter={params['diameter']} | "
        f"cellprob_thresh={params['cellprob_threshold']} | "
        f"flow_thresh={params['flow_threshold']} | "
        f"pix_filter={params['pix_filter']} | "
        f"tile_norm={params['tile_norm_blocksize']} | "
        f"niter={niter_str}"
    )


def _render_overlay_to_ax(ax, img, maski, label_fontsize=2, title=None):
    """Draw a mask overlay + per-cell ID labels onto a matplotlib Axes."""
    img0 = img.copy()
    if img0.ndim == 3 and img0.shape[0] < 4:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.max() <= 50.0:
        img0 = np.uint8(np.clip(img0, 0, 1) * 255)
    if img0.ndim == 2:
        img0 = np.stack([img0, img0, img0], axis=-1)
    elif img0.shape[2] < 3:
        pad = np.zeros((*img0.shape[:2], 3 - img0.shape[2]), dtype=img0.dtype)
        img0 = np.concatenate([img0, pad], axis=2)

    overlay = plot.mask_overlay(img0[:, :, 0], maski)
    ax.imshow(overlay)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=7)

    cell_ids = np.unique(maski)
    cell_ids = cell_ids[cell_ids > 0].tolist()
    if cell_ids:
        centroids = center_of_mass(maski > 0, maski, cell_ids)
        for cell_id, (cy, cx) in zip(cell_ids, centroids):
            ax.text(int(cx), int(cy), str(cell_id),
                    color="white", fontsize=label_fontsize)


def preview_cellpose_params_tiled(
    model,
    crops,
    param_grid,
    output_dir,
    grid_cols=None,
    label_fontsize=2,
    tile_size_inches=3.0,
    dpi=150,
):
    """Run a parameter grid across many crops, saving one contact-sheet PNG per combo.

    crops      : list of dicts with keys {sample, image, region, crop, data}. `data` is
                 an (H, W, 3) array with IBA1 in channel 0, DAPI in channel 1.
    param_grid : dict mapping each of {diameter, cellprob_threshold, flow_threshold,
                 pix_filter, tile_norm_blocksize, niter} to a list of values.
    grid_cols  : columns in the output grid. Defaults to ceil(sqrt(n_crops)).

    Writes <output_dir>/<combo_label>.png per combo and returns one stats row per
    (crop, combo) with keys {sample, image, region, crop, combo, <params...>,
    n_cells, size_min, size_max, size_mean, size_median, error}.
    """
    keys = ['diameter', 'cellprob_threshold', 'flow_threshold',
            'pix_filter', 'tile_norm_blocksize', 'niter']
    for k in keys:
        if k not in param_grid:
            raise ValueError(f"param_grid missing required key: {k}")

    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    os.makedirs(output_dir, exist_ok=True)

    n_crops = len(crops)
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(n_crops)))
    grid_rows = int(np.ceil(n_crops / grid_cols))

    results = []
    for ci, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        label = combo_label(params)
        print(f"[{ci}/{len(combos)}] {label}", flush=True)

        tnb = params['tile_norm_blocksize']
        norm_param = (True if tnb == 0
                      else {'normalize': True, 'tile_norm_blocksize': tnb})

        fig, axes = plt.subplots(
            grid_rows, grid_cols,
            figsize=(grid_cols * tile_size_inches, grid_rows * tile_size_inches),
            dpi=dpi,
        )
        axes = np.atleast_2d(axes).reshape(grid_rows, grid_cols)
        fig.suptitle(combo_title(params), fontsize=10)

        for ti, crop in enumerate(crops):
            ax = axes[ti // grid_cols, ti % grid_cols]
            row = {
                'sample': crop['sample'], 'image': crop['image'],
                'region': crop['region'], 'crop': crop['crop'],
                'combo': label, **params,
                'n_cells': 0, 'size_min': 0, 'size_max': 0,
                'size_mean': 0.0, 'size_median': 0.0, 'error': '',
            }
            try:
                masks, _, _ = model.eval(
                    crop['data'],
                    diameter=params['diameter'],
                    channel_axis=2,
                    flow_threshold=params['flow_threshold'],
                    cellprob_threshold=params['cellprob_threshold'],
                    normalize=norm_param,
                    niter=params['niter'],
                )
                masks = mask_filter_fixed(masks, pix_size=params['pix_filter'])
                stats = mask_size_stats(masks)
                row.update(stats)
                _render_overlay_to_ax(
                    ax, crop['data'], masks,
                    label_fontsize=label_fontsize,
                    title=f"{crop['sample']}  n={stats['n_cells']}",
                )
            except Exception as e:
                row['error'] = str(e)
                ax.text(0.5, 0.5, f'ERROR\n{e}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=6, wrap=True)
                ax.set_axis_off()
            results.append(row)

        for extra in range(n_crops, grid_rows * grid_cols):
            axes[extra // grid_cols, extra % grid_cols].axis('off')

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = os.path.join(output_dir, f'{label}.png')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        cell_counts = [r['n_cells'] for r in results[-n_crops:]]
        print(f"  -> {out_path}  | cells: min={min(cell_counts)} "
              f"max={max(cell_counts)} mean={np.mean(cell_counts):.1f}",
              flush=True)

    return results
