"""
Delamination_Analysis_v2.py
===========================
Enhanced version of Delamination_Analysis.py with improved metrics for
detecting cortical delamination from fluorescence time-lapse movies.

Changes from v1
---------------
New per-frame summary metrics:
  cortex_peak_dist_px    : distance from boundary of the peak-intensity band.
                           Unlike the intensity-weighted centroid, this is not
                           diluted by cytoplasmic background signal.
                           ↑ = cortex peak moves inward (delamination)
                           ↓ = cortex peak moves toward membrane (recruitment)

  outer_2band_ratio      : mean(outermost 2 bands) / mean(bands 2-4).
                           More focused than outer/inner — avoids comparing
                           the membrane region to deep cytoplasm (noisy).
                           ↑ = membrane-proximal actin enrichment
                           ↓ = cortex retreating inward (delamination)

  profile_gradient_outer : (band[2].intensity − band[0].intensity) / 2 bands.
                           Negative = signal highest at membrane (normal cortex).
                           Becomes less negative / positive as cortex separates
                           from membrane (delamination signal).

  membrane_enrichment    : (band[0] − median(all bands)) / MAD.
                           How many robust SDs the outermost band stands above
                           the median band intensity.  High = sharp cortex at
                           membrane; falls as cortex detaches.

Retained from v1 (all columns still written to CSV):
  centroid_dist_px, outer_inner_ratio, peak_band,
  outer_half_frac, inner_half_frac, inner_enrichment, total_intensity

Approach
--------
1. Load image stack and ilastik segmentation mask
2. For each frame:
   a. Euclidean distance transform inside cell mask → distance from boundary
   b. Divide interior into N equal-width concentric bands
   c. Measure mean intensity per band
3. Compute all per-frame summary metrics
4. Save CSV + kymograph + metrics plot + band profiles overlay
5. Batch processing wrapper (unchanged API)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile
import h5py
from scipy.ndimage import (
    distance_transform_edt,
    binary_fill_holes,
    label as ndlabel,
)
from skimage.morphology import remove_small_objects
from tqdm import tqdm


# =============================================================================
# Data loading
# =============================================================================

def load_image_stack(image_path):
    """Load a 2D or 3D TIFF as (T, Y, X).  Single-frame images are wrapped."""
    arr = tifffile.imread(image_path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected TIFF shape {arr.shape}: expected 2-D or 3-D.")
    return arr.astype(np.float32)


def load_ilastik_mask(mask_path, threshold=0.5, erode=False, erosion_radius=5):
    """Load an ilastik probability HDF5 file and threshold to a binary mask stack."""
    from skimage.morphology import erosion, disk

    with h5py.File(mask_path, "r") as f:
        data = f["exported_data"][()]

    if data.ndim == 4:
        data = data[..., 1] if data.shape[-1] == 2 else data[..., 0]
    elif data.ndim == 3 and data.shape[-1] <= 3:
        data = data[..., 1] if data.shape[-1] == 2 else data[..., 0]
        data = data[np.newaxis, ...]

    binary = (data > threshold).astype(bool)

    if erode:
        selem = disk(erosion_radius)
        binary = np.stack([erosion(frame, selem) for frame in binary])

    return binary


# =============================================================================
# Core analysis
# =============================================================================

def _clean_mask(mask_frame):
    """Fill holes, keep largest connected component."""
    filled = binary_fill_holes(mask_frame)
    labeled, n = ndlabel(filled)
    if n == 0:
        return filled
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return (labeled == largest)


def _band_intensities(image, mask, n_bands):
    """
    Compute mean intensity inside N concentric distance bands.

    Returns
    -------
    band_means   : ndarray (n_bands,)   mean intensity per band (NaN if empty)
    band_edges   : ndarray (n_bands+1,) distance edges in pixels
    dist_map     : ndarray (Y, X)       distance-from-boundary map
    """
    dist_map = distance_transform_edt(mask)
    max_dist = dist_map[mask].max() if mask.any() else 0.0

    if max_dist == 0:
        return np.full(n_bands, np.nan), np.zeros(n_bands + 1), dist_map

    band_edges = np.linspace(0, max_dist, n_bands + 1)
    band_means = np.full(n_bands, np.nan)

    for b in range(n_bands):
        lo, hi = band_edges[b], band_edges[b + 1]
        band_px = mask & (dist_map >= lo) & (dist_map < hi)
        if band_px.any():
            band_means[b] = float(image[band_px].mean())

    return band_means, band_edges, dist_map


_CHANNEL_NAMES = {"C1": "Actin", "C2": "CRY2"}


def _parse_cell_label(cell_label):
    """Extract metadata from the standard image filename stem.

    Captures the timepoint (YYYYMMDD-HHMMSS) when present and includes it
    in the 'sample' field as "{timepoint}_{Pcode}" — matching the format
    produced by the polarization pipeline.  Files without a timestamp fall
    back to Pcode-only in 'sample'.
    """
    m = re.match(
        r"^(C\d+)_(?:(\d{8}-\d{6})_)?(block\d+)_(W\d+)_(P\d+)(?:_T\d+)?(?:_smooth)?$",
        cell_label,
    )
    if m:
        channel_code = m.group(1)
        channel      = _CHANNEL_NAMES.get(channel_code, channel_code)
        timepoint    = m.group(2)          # None when absent from filename
        block        = m.group(3)
        well         = m.group(4)
        pcode        = m.group(5)
        sample       = f"{timepoint}_{pcode}" if timepoint else pcode
    else:
        channel = block = well = sample = timepoint = pcode = None

    return dict(channel=channel, block=block, well=well,
                sample=sample, timepoint=timepoint)


# ---------------------------------------------------------------------------
# New metric helpers
# ---------------------------------------------------------------------------

def _cortex_peak_dist(band_means, band_edges):
    """
    Distance from the boundary to the centre of the peak-intensity band.

    Returns NaN when all bands are NaN.
    Unlike the intensity-weighted centroid this is not diluted by diffuse
    cytoplasmic signal and tracks exactly where the brightest cortical
    structure sits.
    """
    if np.all(np.isnan(band_means)):
        return np.nan
    idx = int(np.nanargmax(band_means))
    return float(0.5 * (band_edges[idx] + band_edges[idx + 1]))


def _outer_2band_ratio(band_means, n_bands):
    """
    Mean intensity of the 2 outermost bands divided by the mean of bands 2-4.

    More targeted than comparing the single outermost band to the single
    innermost band (which is dominated by cytoplasmic noise).
    Returns NaN when n_bands < 5.
    """
    if n_bands < 5:
        return np.nan
    outer2 = np.nanmean(band_means[:2])
    mid3   = np.nanmean(band_means[2:5])
    if np.isnan(outer2) or np.isnan(mid3) or mid3 == 0:
        return np.nan
    return float(outer2 / mid3)


def _profile_gradient_outer(band_means):
    """
    Slope of the intensity profile from band 0 to band 2 (px^{-1} units,
    normalised by max band mean so it is scale-independent).

    Negative  → intensity is highest at the membrane and drops inward (intact cortex).
    Near zero → cortex no longer concentrated at membrane (delamination).
    Positive  → inner bands brighter than outermost (collapsed cortex).

    Returns NaN when any of the first 3 bands is NaN or when max == 0.
    """
    if len(band_means) < 3 or np.any(np.isnan(band_means[:3])):
        return np.nan
    mx = np.nanmax(band_means)
    if mx == 0:
        return np.nan
    # (band[2] - band[0]) / (2 band steps)  — normalised by max
    return float((band_means[2] - band_means[0]) / (2 * mx))


def _membrane_enrichment(band_means):
    """
    Robust z-score of the outermost band relative to all bands.

    membrane_enrichment = (band[0] − median) / MAD

    High positive value → sharp actin peak at the cell boundary.
    Drops toward zero / negative as the cortex detaches from the membrane.
    Returns NaN when MAD == 0 or all bands are NaN.
    """
    if np.all(np.isnan(band_means)):
        return np.nan
    med = float(np.nanmedian(band_means))
    mad = float(np.nanmedian(np.abs(band_means - med)))
    if mad == 0:
        return np.nan
    return float((band_means[0] - med) / mad)


# =============================================================================
# Main analysis function
# =============================================================================

def analyze_delamination(
    image_path,
    mask_path,
    n_bands=10,
    mask_threshold=0.5,
    erode_mask=False,
    erosion_radius=5,
    output_dir=None,
    cell_label=None,
    condition_name=None,
):
    """
    Full delamination analysis for a single cell time-lapse.

    Parameters
    ----------
    image_path      : str   Path to 2-D/3-D TIFF (T, Y, X)
    mask_path       : str   Path to ilastik H5 probability mask
    n_bands         : int   Number of concentric shells from outline inward
    mask_threshold  : float Threshold for binary mask (default 0.5)
    erode_mask      : bool  Erode mask before analysis
    erosion_radius  : int   Erosion radius in pixels
    output_dir      : str   Folder to save results (created if needed)
    cell_label      : str   Identifier used in output file names
    condition_name  : str   Experiment identifier used to build Cell_ID

    Returns
    -------
    df_summary : pd.DataFrame  Per-frame summary metrics
    df_bands   : pd.DataFrame  Per-frame per-band intensities (wide format)
    """
    if cell_label is None:
        cell_label = os.path.splitext(os.path.basename(image_path))[0]

    meta = _parse_cell_label(cell_label)
    if condition_name is not None and meta["sample"] is not None:
        meta["Cell_ID"] = f"{condition_name}_{meta['sample']}"
    elif meta["sample"] is not None:
        meta["Cell_ID"] = meta["sample"]
    else:
        meta["Cell_ID"] = None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    stack     = load_image_stack(image_path)
    masks_raw = load_ilastik_mask(mask_path, threshold=mask_threshold,
                                  erode=erode_mask, erosion_radius=erosion_radius)

    n_frames = stack.shape[0]
    if masks_raw.shape[0] == 1 and n_frames > 1:
        masks_raw = np.repeat(masks_raw, n_frames, axis=0)

    all_band_means = []
    summary_rows   = []
    band_edges_ref = None

    for t in tqdm(range(n_frames), desc=f"Delamination [{cell_label}]", leave=False):
        image = stack[t]
        mask  = _clean_mask(masks_raw[t] if t < masks_raw.shape[0] else masks_raw[-1])

        band_means, band_edges, dist_map = _band_intensities(image, mask, n_bands)

        if band_edges_ref is None and not np.all(np.isnan(band_means)):
            band_edges_ref = band_edges

        all_band_means.append(band_means)

        # ---- v1 metrics ----
        total_int = float(image[mask].sum()) if mask.any() else 0.0

        if mask.any() and total_int > 0:
            centroid_px = float(np.sum(dist_map[mask] * image[mask]) / total_int)
        else:
            centroid_px = np.nan

        outer_mean = band_means[0]  if not np.isnan(band_means[0])  else np.nan
        inner_mean = band_means[-1] if not np.isnan(band_means[-1]) else np.nan
        outer_inner = (outer_mean / inner_mean
                       if (not np.isnan(outer_mean) and
                           not np.isnan(inner_mean) and
                           inner_mean != 0)
                       else np.nan)

        peak_band_idx = (int(np.nanargmax(band_means))
                         if not np.all(np.isnan(band_means)) else np.nan)

        norm_bands = band_means / total_int if total_int > 0 else band_means * np.nan

        half = n_bands // 2
        outer_frac = (np.nansum(norm_bands[:half])
                      if not np.all(np.isnan(norm_bands[:half])) else np.nan)
        inner_frac = (np.nansum(norm_bands[half:])
                      if not np.all(np.isnan(norm_bands[half:])) else np.nan)

        # ---- v2 new metrics ----
        cortex_peak_dist  = _cortex_peak_dist(band_means, band_edges)
        outer_2band_ratio = _outer_2band_ratio(band_means, n_bands)
        grad_outer        = _profile_gradient_outer(band_means)
        mem_enrichment    = _membrane_enrichment(band_means)

        summary_rows.append({
            # ---- shared metadata ----
            "frame":               t,
            "sample":              meta["sample"],
            "channel":             meta["channel"],
            "block":               meta["block"],
            "timepoint":           meta["timepoint"],
            "Cell_ID":             meta["Cell_ID"],
            # ---- v1 metrics ----
            "centroid_dist_px":    centroid_px,
            "outer_inner_ratio":   outer_inner,
            "peak_band":           peak_band_idx,
            "outer_half_frac":     outer_frac,
            "inner_half_frac":     inner_frac,
            "inner_enrichment":    inner_frac - outer_frac,
            "total_intensity":     total_int,
            # ---- v2 new metrics ----
            "cortex_peak_dist_px": cortex_peak_dist,
            "outer_2band_ratio":   outer_2band_ratio,
            "profile_gradient_outer": grad_outer,
            "membrane_enrichment": mem_enrichment,
        })

    df_summary = pd.DataFrame(summary_rows)

    band_means_arr = np.array(all_band_means)

    if band_edges_ref is None:
        band_edges_ref = np.linspace(0, 1, n_bands + 1)

    col_names = [
        f"band_{b:02d}_dist{band_edges_ref[b]:.1f}-{band_edges_ref[b+1]:.1f}px"
        for b in range(n_bands)
    ]
    df_bands = pd.DataFrame(band_means_arr, columns=col_names)
    df_bands.insert(0, "frame", np.arange(n_frames))
    for col, val in [("sample",    meta["sample"]),
                     ("channel",   meta["channel"]),
                     ("block",     meta["block"]),
                     ("timepoint", meta["timepoint"]),
                     ("Cell_ID",   meta["Cell_ID"])]:
        df_bands[col] = val

    if output_dir is not None:
        df_summary.to_csv(
            os.path.join(output_dir, f"{cell_label}_delamination_summary.csv"),
            index=False,
        )
        df_bands.to_csv(
            os.path.join(output_dir, f"{cell_label}_band_intensities.csv"),
            index=False,
        )
        _plot_delamination_kymograph(band_means_arr, band_edges_ref,
                                     cell_label, output_dir)
        _plot_delamination_metrics(df_summary, cell_label, output_dir, n_bands)
        _plot_band_profiles_overlay(band_means_arr, n_bands, cell_label, output_dir)

    return df_summary, df_bands


# =============================================================================
# Visualisation
# =============================================================================

def _plot_delamination_kymograph(band_means_arr, band_edges_ref,
                                  cell_label, output_dir):
    """Heatmap: rows = bands (0=outermost), columns = frames."""
    n_frames, n_bands = band_means_arr.shape

    norm_arr = band_means_arr.copy()
    frame_max = np.nanmax(norm_arr, axis=1, keepdims=True)
    frame_max[frame_max == 0] = 1
    norm_arr = norm_arr / frame_max

    fig, ax = plt.subplots(figsize=(max(6, n_frames * 0.15 + 2), 4))
    im = ax.imshow(norm_arr.T, aspect="auto", origin="lower",
                   cmap="inferno", vmin=0, vmax=1, interpolation="nearest")

    mid_dists = [
        f"{0.5*(band_edges_ref[b]+band_edges_ref[b+1]):.0f}"
        for b in range(n_bands)
    ]
    ax.set_yticks(range(n_bands))
    ax.set_yticklabels(mid_dists, fontsize=8)
    ax.set_ylabel("Distance from boundary (px)\noutermost → innermost")
    ax.set_xlabel("Frame")
    ax.set_title(f"Delamination kymograph — {cell_label}")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Norm. intensity")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{cell_label}_delamination_kymograph.png"),
        dpi=150,
    )
    plt.close(fig)


def _plot_delamination_metrics(df_summary, cell_label, output_dir, n_bands):
    """
    Eight-panel time-series showing both v1 and v2 metrics.
    v1 metrics (top 4): centroid_dist_px, outer_inner_ratio,
                        inner_enrichment, peak_band
    v2 metrics (bottom 4): cortex_peak_dist_px, outer_2band_ratio,
                            profile_gradient_outer, membrane_enrichment
    """
    frames = df_summary["frame"].values

    fig, axes = plt.subplots(8, 1, figsize=(9, 18), sharex=True)

    # ---------- v1 ----------
    axes[0].plot(frames, df_summary["centroid_dist_px"], color="steelblue", lw=1.5)
    axes[0].set_ylabel("px")
    axes[0].set_title("v1 | Intensity centroid distance from boundary\n(↑ = cortex moves inward)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, df_summary["outer_inner_ratio"], color="darkorange", lw=1.5)
    axes[1].axhline(1, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("ratio")
    axes[1].set_title("v1 | Outermost / innermost band intensity\n(↓ = delamination / inward shift)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, df_summary["inner_enrichment"], color="firebrick", lw=1.5)
    axes[2].axhline(0, color="gray", ls="--", lw=0.8)
    axes[2].set_ylabel("inner_frac − outer_frac")
    axes[2].set_title("v1 | Inner half enrichment (↑ = delamination / inward)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(frames, df_summary["peak_band"], color="purple", lw=1.5,
                 marker="o", ms=2)
    axes[3].set_ylabel("band index")
    axes[3].set_title("v1 | Peak band index (↑ = peak shifts inward)")
    axes[3].grid(True, alpha=0.3)

    # ---------- v2 ----------
    axes[4].plot(frames, df_summary["cortex_peak_dist_px"], color="teal", lw=1.5)
    axes[4].set_ylabel("px")
    axes[4].set_title(
        "v2 | Cortex peak distance from boundary\n"
        "(↑ = peak moves inward; ↓ = peak moves to membrane)"
    )
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(frames, df_summary["outer_2band_ratio"], color="goldenrod", lw=1.5)
    axes[5].axhline(1, color="gray", ls="--", lw=0.8)
    axes[5].set_ylabel("ratio")
    axes[5].set_title(
        "v2 | Outer-2-band ratio  [mean(b0,b1) / mean(b2,b3,b4)]\n"
        "(↑ = actin enriched at membrane; ↓ = cortex retreating)"
    )
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(frames, df_summary["profile_gradient_outer"], color="darkgreen", lw=1.5)
    axes[6].axhline(0, color="gray", ls="--", lw=0.8)
    axes[6].set_ylabel("norm. slope")
    axes[6].set_title(
        "v2 | Outer profile gradient  [(b2−b0)/(2·max)]\n"
        "(negative = cortex at membrane; rises toward 0 / + = delamination)"
    )
    axes[6].grid(True, alpha=0.3)

    axes[7].plot(frames, df_summary["membrane_enrichment"], color="saddlebrown", lw=1.5)
    axes[7].axhline(0, color="gray", ls="--", lw=0.8)
    axes[7].set_ylabel("robust z-score")
    axes[7].set_xlabel("Frame")
    axes[7].set_title(
        "v2 | Membrane enrichment  [(b0−median)/MAD]\n"
        "(↓ = b0 losing its prominence = cortex leaving membrane)"
    )
    axes[7].grid(True, alpha=0.3)

    fig.suptitle(f"Delamination metrics (v2) — {cell_label}\n(n_bands={n_bands})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{cell_label}_delamination_metrics.png"),
        dpi=150,
    )
    plt.close(fig)


def _plot_band_profiles_overlay(band_means_arr, n_bands, cell_label, output_dir):
    """Overlay per-frame band intensity profiles, coloured early→late."""
    n_frames = band_means_arr.shape[0]
    cmap = plt.cm.viridis
    band_indices = np.arange(n_bands)

    fig, ax = plt.subplots(figsize=(7, 4))

    for t in range(n_frames):
        profile = band_means_arr[t]
        color = cmap(t / max(n_frames - 1, 1))
        mx = np.nanmax(profile)
        if mx > 0:
            ax.plot(band_indices, profile / mx, color=color, alpha=0.5, lw=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=mcolors.Normalize(vmin=0, vmax=n_frames))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Frame")

    ax.set_xticks(band_indices)
    ax.set_xticklabels([str(b) for b in band_indices], fontsize=8)
    ax.set_xlabel("Band index  (0 = outermost)")
    ax.set_ylabel("Normalized mean intensity")
    ax.set_title(f"Band intensity profiles over time — {cell_label}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{cell_label}_band_profiles.png"),
        dpi=150,
    )
    plt.close(fig)


# =============================================================================
# Delamination onset detection (unchanged API, works with any metric)
# =============================================================================

def detect_delamination_onset(
    df_summary,
    metric="membrane_enrichment",
    baseline_frames=10,
    threshold_std_factor=2.0,
    direction="decrease",
):
    """
    Detect first frame where a metric crosses a threshold.

    Parameters
    ----------
    direction : "increase" or "decrease"
        "decrease" — onset when value falls below baseline_mean − k*std
                     (suitable for membrane_enrichment, outer_2band_ratio)
        "increase" — onset when value rises above baseline_mean + k*std
                     (suitable for centroid_dist_px, cortex_peak_dist_px)
    """
    values = df_summary[metric].values
    baseline = values[:baseline_frames]
    baseline_mean = np.nanmean(baseline)
    baseline_std  = np.nanstd(baseline)

    if direction == "decrease":
        threshold = baseline_mean - threshold_std_factor * baseline_std
        check = lambda v: v < threshold
    else:
        threshold = baseline_mean + threshold_std_factor * baseline_std
        check = lambda v: v > threshold

    onset_frame = None
    for t, v in enumerate(values):
        if t < baseline_frames:
            continue
        if not np.isnan(v) and check(v):
            onset_frame = t
            break

    return onset_frame, threshold, baseline_mean


# =============================================================================
# Aggregate across cells
# =============================================================================

def make_master_delamination_csv(results_root, condition_name):
    """
    Collect all *_delamination_summary.csv files in results_root,
    merge blocks per (sample, channel) into a continuous timecourse,
    and save master_delamination_{condition_name}.csv.
    """
    dfs = []
    for root, _dirs, files in os.walk(results_root):
        for fname in sorted(files):
            if not fname.endswith("_delamination_summary.csv"):
                continue
            df = pd.read_csv(os.path.join(root, fname))
            if "sample" not in df.columns:
                print(f"  SKIP (no metadata columns): {fname}")
                continue
            dfs.append(df)

    if not dfs:
        print("No delamination summary CSVs found.")
        return pd.DataFrame()

    all_data = pd.concat(dfs, ignore_index=True)

    for col in ("sample", "channel", "block"):
        n_missing = all_data[col].isna().sum() if col in all_data.columns else len(all_data)
        if n_missing > 0:
            print(f"  WARNING: {n_missing} rows with no '{col}' value.")

    def _block_num(b):
        m = re.search(r"(\d+)$", str(b))
        return int(m.group(1)) if m else 0

    all_data["_block_num"] = all_data["block"].apply(_block_num)

    merged_parts = []
    for (sample, channel), grp in all_data.groupby(["sample", "channel"], dropna=False):
        grp = grp.sort_values(["_block_num", "frame"]).copy()
        frame_offset = 0
        new_frames = []
        for blk in sorted(grp["_block_num"].unique()):
            blk_rows = grp[grp["_block_num"] == blk]
            n = len(blk_rows)
            new_frames.extend(range(frame_offset, frame_offset + n))
            frame_offset += n
        grp["frame"] = new_frames
        merged_parts.append(grp)

    if not merged_parts:
        return pd.DataFrame()

    master = pd.concat(merged_parts, ignore_index=True)
    master = master.drop(columns=["_block_num"])

    out_path = os.path.join(results_root, f"master_delamination_{condition_name}.csv")
    master.to_csv(out_path, index=False)
    n_cells = master["sample"].nunique() if "sample" in master.columns else "?"
    print(f"Master CSV → {out_path}  ({len(master)} rows, {n_cells} cells)")
    return master


# =============================================================================
# Batch processing
# =============================================================================

def batch_delamination(
    image_folder,
    mask_folder,
    results_root,
    condition_name,
    n_bands=10,
    mask_threshold=0.5,
    erode_mask=False,
    erosion_radius=5,
    image_suffix="_smooth.tif",
    mask_suffix="_Probabilities.h5",
):
    """
    Process all images in image_folder matching masks in mask_folder.
    Outputs per-cell subfolders under results_root.
    """
    os.makedirs(results_root, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(image_suffix)])
    mask_files  = sorted([f for f in os.listdir(mask_folder)  if f.endswith(mask_suffix)])

    if not image_files:
        print(f"No images found in {image_folder} with suffix '{image_suffix}'")
        return

    def _extract_cell_code(fname):
        m = re.search(r"(block\d+[_-]W\d+[_-]P\d+)", fname, re.IGNORECASE)
        if m:
            return m.group(1).replace("-", "_")
        return None

    mask_lookup = {}
    for mf in mask_files:
        code = _extract_cell_code(mf)
        if code:
            mask_lookup[code] = mf

    print(f"Found {len(image_files)} image(s), {len(mask_files)} mask(s).")
    matched, skipped = 0, 0

    for img_fname in image_files:
        cell_code = _extract_cell_code(img_fname)
        if cell_code is None:
            print(f"  SKIP (no cell code): {img_fname}")
            skipped += 1
            continue

        mask_fname = mask_lookup.get(cell_code)
        if mask_fname is None:
            alt_code = cell_code.replace("_C2_", "_C1_")
            mask_fname = mask_lookup.get(alt_code)
        if mask_fname is None:
            print(f"  SKIP (no mask for {cell_code}): {img_fname}")
            skipped += 1
            continue

        cell_label = os.path.splitext(img_fname)[0]
        out_dir = os.path.join(results_root, cell_label)

        print(f"  Processing: {img_fname}")
        try:
            analyze_delamination(
                image_path=os.path.join(image_folder, img_fname),
                mask_path=os.path.join(mask_folder, mask_fname),
                n_bands=n_bands,
                mask_threshold=mask_threshold,
                erode_mask=erode_mask,
                erosion_radius=erosion_radius,
                output_dir=out_dir,
                cell_label=cell_label,
                condition_name=condition_name,
            )
            matched += 1
        except Exception as e:
            print(f"  ERROR processing {img_fname}: {e}")
            skipped += 1

    print(f"\nBatch complete: {matched} processed, {skipped} skipped.")
    make_master_delamination_csv(results_root, condition_name)

#%%
# =============================================================================
# Entry point (example usage)
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configure paths here before running
    # -------------------------------------------------------------------------

    image_folder   = "/Volumes/diz/Filipe/2.Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/FNV061_260129_7XFN-NI2-RE-FACS-HIGH/Processed/S01-7XFN-NI2-RE-FACS-Tube2/Stacks/"
    mask_folder    = "/Volumes/diz/Filipe/2.Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/FNV061_260129_7XFN-NI2-RE-FACS-HIGH/Processed/S01-7XFN-NI2-RE-FACS-Tube2/Masks/"
    results_root   = "/Volumes/diz/Filipe/2.Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/FNV061_260129_7XFN-NI2-RE-FACS-HIGH/Processed/S01-7XFN-NI2-RE-FACS-Tube2/Delamination/"
    condition_name = "20260129-S01-7XFN-NI2RE-FACS-high"   # experiment label → used in Cell_ID and master CSV name


    batch_delamination(
        image_folder=image_folder,
        mask_folder=mask_folder,
        results_root=results_root,
        condition_name=condition_name,
        n_bands=15,           # recommended: 20 bands for finer cortex resolution
        mask_threshold=0.5,
        erode_mask=True,      # recommended: erode 2 px to avoid membrane boundary artefacts
        erosion_radius=3,
    )
