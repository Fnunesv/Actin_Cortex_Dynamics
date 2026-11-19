#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 2026

@author: nunes

Cortical accumulation & cross-channel correlation in fixed z-stacks.

Adapted from actin-polarization/2.Processing/Actin_Polarization_FullProcessing copy.py
(timelapse unwrap → kymograph → R / FWHM / ROI / PI). Here the "frame" axis of the
kymograph is the z-plane, and profiles of several channels are sampled on the SAME
cortical band (defined from the 488 actin ilastik segmentation) so they can be
correlated angle-by-angle.

Run (image_analysis env):
    /Users/nunes/yes/envs/image_analysis/bin/python Cortex_Accumulation_Zstack.py
    ... --z 9 21 --cells 1 2 9 --pairs all --experiment FNV082_D5
"""
#%% --- Imports ---

import os
import re
import argparse
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import tifffile
from scipy.ndimage import (binary_fill_holes, map_coordinates, uniform_filter1d,
                           median_filter, binary_dilation as nd_binary_dilation)
from scipy.stats import pearsonr, spearmanr
from skimage.morphology import binary_closing, binary_opening, disk, convex_hull_image
from skimage.measure import label, regionprops
from skimage.transform import resize


#%% --- Segmentation helpers (from Actin_Polarization_FullProcessing) ---

def robust_fill(mask, gap_close_radius=8, min_fill_ratio=0.5):
    """
    Fill a binary ring mask: closing → fill_holes → convex-hull fallback if the
    ring was still open (same logic as the live-cell pipeline).
    """
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    raw_area = int(mask.sum())
    closed   = binary_closing(mask, disk(gap_close_radius))
    filled   = binary_fill_holes(closed)
    if int(filled.sum()) < raw_area * (1 + min_fill_ratio):
        filled = convex_hull_image(mask)
    return filled


def clean_cell_mask(raw, px_um, gap_close_um=1.0, open_um=1.0, ds=4):
    """
    Round 2 replacement for robust_fill at 0.028 µm/px. Morphology runs on a ds× downsampled grid.
      1. closing (gap_close_um) bridges gaps in the ilastik ring, so the convex-hull
         fallback (which draws straight chords across the cell) is rarely needed
      2. fill_holes, with a convex-hull fallback if the ring is still open
      3. opening (open_um) removes protrusions narrower than ~2·open_um
    """
    small = raw[::ds, ::ds].astype(bool)
    if not small.any():
        return np.zeros_like(raw, bool)
    g = max(1, int(round(gap_close_um / (px_um * ds))))
    filled = binary_fill_holes(binary_closing(small, disk(g)))
    if filled.sum() < small.sum() * 1.5:
        filled = convex_hull_image(small)
    if open_um > 0:
        filled = binary_opening(filled, disk(max(1, int(round(open_um / (px_um * ds))))))
    filled = largest_component(filled)
    return resize(filled.astype(np.float32), raw.shape, order=1) > 0.5


def largest_component(mask):
    lbl = label(mask)
    if lbl.max() == 0:
        return mask.astype(bool)
    return lbl == (np.argmax(np.bincount(lbl.flat)[1:]) + 1)


#%% --- Polarization metrics (from Actin_Polarization_FullProcessing) ---

def weighted_circmean(weights):
    N = len(weights)
    angles = np.deg2rad(np.arange(N) * 360.0 / N)
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    return np.arctan2(y, x)  # rad [-pi,pi]

def resultant_length(weights):
    N = len(weights)
    angles = np.deg2rad(np.arange(N) * 360.0 / N)
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    return np.hypot(x, y) / (np.sum(weights) + 1e-12)

def fwhm_width(weights, frac=0.5):
    N = len(weights)
    peak = weights.max()
    mask = weights >= peak * frac
    if mask.all():
        return 360.0
    mask2 = np.concatenate([mask, mask])
    diffs = np.diff(np.concatenate([[0], mask2.view(np.int8), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths = ends - starts
    max_len = min(lengths.max(), N) if lengths.size else 0
    return (max_len / N) * 360

def roi_fraction(weights, center_rad, window_size=60):
    N = len(weights)
    deg_per = 360 / N
    center_deg = (np.rad2deg(center_rad) % 360)
    half = int((window_size/2)/deg_per)
    center_idx = int(center_deg/deg_per)
    idxs = np.arange(center_idx-half, center_idx+half+1) % N
    return weights[idxs].sum() / (weights.sum() + 1e-12)


#%% --- I/O ---

def find_cell_files(data_root, cell_id, channels):
    """Return {channel_name: tif_path} + mask path for one cell id (int)."""
    tag = f"Cell_{int(cell_id):03d}"
    files = {}
    for ch, sub in channels.items():
        folder = os.path.join(data_root, sub)
        hits = [f for f in os.listdir(folder)
                if tag in f and f.endswith(".tif") and not f.startswith("._")]
        if len(hits) != 1:
            raise FileNotFoundError(f"{ch}: expected 1 tif for {tag} in {folder}, got {hits}")
        files[ch] = os.path.join(folder, hits[0])
    return files


def find_mask_file(data_root, cell_id, mask_subfolder):
    tag = f"Cell_{int(cell_id):03d}_Probabilities.h5"
    folder = os.path.join(data_root, mask_subfolder)
    hits = [f for f in os.listdir(folder) if f.endswith(tag) and not f.startswith("._")]
    if len(hits) != 1:
        raise FileNotFoundError(f"mask: expected 1 h5 for {tag} in {folder}, got {hits}")
    return os.path.join(folder, hits[0])


def load_planes(tif_path, z_idx):
    """Load only the requested z planes (0-based) of a single-channel (Z,Y,X) tif."""
    arr = tifffile.imread(tif_path, key=list(z_idx))
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    return arr.astype(np.float32)


def load_mask_planes(h5_path, z_idx, threshold=0.5):
    """ilastik export (Z,Y,X,C) → binary (z,Y,X) for the requested planes."""
    with h5py.File(h5_path, "r") as f:
        d = f["exported_data"]
        data = np.stack([d[z] for z in z_idx])
    if data.ndim == 4:
        ch = 1 if data.shape[-1] == 2 else 0
        data = data[..., ch]
    return data > threshold


def pixel_size_from_tif(tif_path):
    """µm/px from the ImageJ XResolution tag (px per µm)."""
    with tifffile.TiffFile(tif_path) as t:
        num, den = t.pages[0].tags["XResolution"].value
    return den / num


#%% --- Radial unwrap ---

def outer_radius_profile(filled, center, n_rays, r_step=0.5, median_deg=9):
    """
    For each ray angle, the outermost radius (px) still inside the filled mask.
    Median-filtered circularly to suppress single-ray spikes from filopodia.
    Angle convention: 0° = +x (image right), increasing counter-clockwise on screen.
    """
    cy, cx = center
    rmax = np.hypot(*filled.shape)
    r = np.arange(0, rmax, r_step)
    th = np.deg2rad(np.arange(n_rays) * 360.0 / n_rays)
    yy = cy - np.outer(np.sin(th), r)
    xx = cx + np.outer(np.cos(th), r)
    inside = map_coordinates(filled.astype(np.float32), [yy, xx], order=0, cval=0) > 0.5
    # last True index along each ray
    idx = np.where(inside.any(axis=1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1), 0)
    r_out = r[idx]
    k = max(1, int(round(median_deg * n_rays / 360)))
    if k > 1:
        r_out = median_filter(r_out, size=k, mode="wrap")
    return r_out


def ridge_radius_profile(image, center, r_out, px_um, search_um=2.5, r_step=0.5, smooth_deg=15):
    """
    Per ray, the radius of maximal intensity (actin cortex) within [r_out − search_um, r_out].
    The ilastik ring is ~1 µm thick and protrusions push r_out outward, so the outer edge
    is not where the cortex is. The ridge follows the bright cortical line instead.
    Circular median filter over smooth_deg removes jumps to isolated bright spots.
    """
    cy, cx = center
    n = len(r_out)
    th = np.deg2rad(np.arange(n) * 360.0 / n)
    offs = np.arange(-search_um / px_um, r_step, r_step)
    rr = np.clip(r_out[:, None] + offs[None, :], 0, None)
    v = map_coordinates(image, [cy - np.sin(th)[:, None] * rr, cx + np.cos(th)[:, None] * rr],
                        order=1, mode="nearest")
    v = uniform_filter1d(v, size=5, axis=1)
    r = rr[np.arange(n), np.argmax(v, axis=1)]
    k = max(1, int(round(smooth_deg * n / 360)))
    return median_filter(r, size=k, mode="wrap") if k > 1 else r


def sample_band(image, center, r_out, n_bins, band_in_px, band_out_px,
                oversample=5, r_step=0.5, reduce="mean"):
    """
    Sample `image` in a cortical band [r_out - band_in, r_out + band_out] along
    n_bins*oversample rays, reduce radially (mean|max), then average sub-rays → n_bins.
    """
    cy, cx = center
    n_rays = n_bins * oversample
    th = np.deg2rad(np.arange(n_rays) * 360.0 / n_rays)
    offs = np.arange(-band_in_px, band_out_px + r_step, r_step)
    rr = r_out[:, None] + offs[None, :]
    rr = np.clip(rr, 0, None)
    yy = cy - np.sin(th)[:, None] * rr
    xx = cx + np.cos(th)[:, None] * rr
    vals = map_coordinates(image, [yy, xx], order=1, mode="nearest")
    v = vals.max(axis=1) if reduce == "max" else vals.mean(axis=1)
    return v.reshape(n_bins, oversample).mean(axis=1)


def arc_length_per_bin(r_out_bins, px_um):
    """µm of contour per angular bin: sqrt(r² + (dr/dθ)²)·dθ."""
    n = len(r_out_bins)
    dth = 2 * np.pi / n
    dr = (np.roll(r_out_bins, -1) - np.roll(r_out_bins, 1)) / (2 * dth)
    return np.sqrt(r_out_bins**2 + dr**2) * dth * px_um


#%% --- Accumulation metrics ---

def accumulation_region(E, method="halfmax", frac=0.5, threshold=1.5):
    """
    Contiguous (circular) region around the enrichment peak.
      halfmax  : E >= 1 + frac*(E_peak - 1)   (half-max above the cell baseline)
      threshold: E >= threshold               (fixed fold-enrichment)
    Returns boolean mask (len N) and peak index.
    """
    N = len(E)
    p = int(np.argmax(E))
    level = 1 + frac * (E[p] - 1) if method == "halfmax" else threshold
    region = np.zeros(N, bool)
    if E[p] < level or E[p] <= 1:
        return region, p
    region[p] = True
    for step in (1, -1):
        i = p
        for _ in range(N - 1):
            i = (i + step) % N
            if E[i] >= level and not region[i]:
                region[i] = True
            else:
                break
    return region, p


def profile_metrics(I, arc_um, roi_window_deg=60, region_method="halfmax",
                    region_frac=0.5, region_threshold=1.5, region_smooth_deg=20):
    """
    I: background-subtracted, smoothed angular profile (N bins).
    E = I / median(I) is the fold-enrichment over the cell's own cortical baseline.
    The peak/region are detected on E smoothed by region_smooth_deg (domain scale,
    not single speckles); enrichment values inside the region use the unsmoothed E.
    """
    N = len(I)
    Ip = np.clip(I, 0, None)
    base = np.median(Ip)
    E = Ip / (base + 1e-12)
    k = max(1, int(round(region_smooth_deg * N / 360)))
    E_dom = uniform_filter1d(E, size=k, mode="wrap") if k > 1 else E
    region, p = accumulation_region(E_dom, region_method, region_frac, region_threshold)
    deg_per = 360.0 / N
    c = weighted_circmean(Ip)
    R = resultant_length(Ip)
    roi = roi_fraction(Ip, c, roi_window_deg)
    excess = np.clip(E - 1, 0, None)
    return dict(
        mean_intensity       = float(I.mean()),
        baseline_intensity   = float(base),
        peak_enrichment      = float(E_dom[p]),          # domain-scale peak
        peak_enrichment_raw  = float(E.max()),           # sharpest speckle
        peak_deg             = p * deg_per,
        center_deg           = float(np.rad2deg(c) % 360),
        region_width_deg     = float(region.sum() * deg_per),
        region_width_um      = float(arc_um[region].sum()),
        region_mean_enrich   = float(E[region].mean()) if region.any() else np.nan,
        region_signal_frac   = float(Ip[region].sum() / (Ip.sum() + 1e-12)),
        # integrated excess over baseline inside the region, as a fraction of a full
        # baseline-cortex: combines degree (E-1) and size (width) of the accumulation
        accumulation_score   = float(excess[region].sum() * deg_per / 360.0),
        resultant_length     = float(R),
        FWHM_deg             = float(fwhm_width(Ip)),
        ROI_frac             = float(roi),
        polarity_index       = float(R * roi),
    ), E, region, p


#%% --- Correlation ---

def circ_xcorr(a, b):
    """Circular cross-correlation of z-scored profiles; lag>0 means b is shifted +lag vs a."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    xc = np.real(np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b))) / len(a)
    return xc  # index k ↔ lag k bins


def shift_null_p(a, b, n_perm=1000, min_shift_deg=30, rng=None):
    """
    p-value for Pearson r against random circular shifts of b (keeps autocorrelation).
    Shifts closer than min_shift_deg to 0 are excluded.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    N = len(a)
    lo = int(np.ceil(min_shift_deg * N / 360))
    shifts = rng.integers(lo, N - lo + 1, size=n_perm)
    r0 = pearsonr(a, b)[0]
    rs = np.array([pearsonr(a, np.roll(b, s))[0] for s in shifts])
    return float((np.sum(rs >= r0) + 1) / (n_perm + 1))


def partial_corr(a, b, c):
    """Pearson r of a and b after regressing out c from both (e.g. actin as cortex-amount control)."""
    def resid(y):
        X = np.column_stack([np.ones_like(c), c])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    return float(pearsonr(resid(a), resid(b))[0])


def circ_diff_deg(a, b):
    return float((b - a + 180) % 360 - 180)


def pair_correlation(Ia, Ib, Ea, Eb, reg_a, reg_b, pk_a, pk_b, Ic=None, n_perm=1000):
    N = len(Ia)
    xc = circ_xcorr(Ia, Ib)
    k = int(np.argmax(xc))
    lag = k if k <= N // 2 else k - N
    excess_b = np.clip(Eb - 1, 0, None)
    union = (reg_a | reg_b).sum()
    return dict(
        pearson_r            = float(pearsonr(Ia, Ib)[0]),
        spearman_rho         = float(spearmanr(Ia, Ib)[0]),
        shift_null_p         = shift_null_p(Ia, Ib, n_perm=n_perm),
        xcorr_zero_lag       = float(xc[0]),
        xcorr_max            = float(xc[k]),
        xcorr_lag_deg        = float(lag * 360.0 / N),
        # peak-to-peak angle B − A
        partial_r_ctrl       = partial_corr(Ia, Ib, Ic) if Ic is not None else np.nan,
        # domain-peak to domain-peak angle B − A
        peak_offset_deg      = circ_diff_deg(pk_a * 360.0 / N, pk_b * 360.0 / N),
        region_jaccard       = float((reg_a & reg_b).sum() / union) if union else np.nan,
        # does B accumulate where A accumulates?
        B_enrich_in_A_region = float(Eb[reg_a].mean()) if reg_a.any() else np.nan,
        B_excess_frac_in_A_region = float(excess_b[reg_a].sum() / (excess_b.sum() + 1e-12)),
        A_region_frac_of_cortex   = float(reg_a.mean()),
    )


#%% --- Per-cell analysis ---

def clean_experiment_name(name):
    """Make the experiment name safe for file names: keep letters, digits, '-', '.'; everything else → '_'."""
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", str(name or "").strip()).strip("_")


def cell_label(cell_id, experiment=""):
    """'<experiment>_Cell_###' (or 'Cell_###' without experiment) — unique across merged experiments."""
    tag = f"Cell_{int(cell_id):03d}"
    return f"{experiment}_{tag}" if experiment else tag


def analyze_cell(
    cell_id,
    data_root,
    channels,
    mask_subfolder,
    out_root,
    z_range_fiji=(9, 21),
    corr_pairs=(("561", "647"),),
    mask_threshold=0.5,
    gap_close_um=1.0,
    open_um=1.0,
    band_mode="ridge",
    ridge_channel="488",
    ridge_search_um=2.5,
    ridge_smooth_deg=15,
    num_points=360,
    oversample=5,
    band_in_um=0.3,
    band_out_um=0.3,
    reduce="mean",
    smooth_deg=5,
    bg_dilate_um=3.0,
    min_fill_area_px=5000,
    roi_window_deg=60,
    region_method="halfmax",
    region_frac=0.5,
    region_threshold=1.5,
    region_smooth_deg=20,
    control_channel="488",
    n_perm=1000,
    save_plots=True,
    experiment="",
):
    experiment = clean_experiment_name(experiment)
    cell = cell_label(cell_id, experiment)
    out_dir = os.path.join(out_root, cell)
    os.makedirs(out_dir, exist_ok=True)

    files = find_cell_files(data_root, cell_id, channels)
    mask_path = find_mask_file(data_root, cell_id, mask_subfolder)
    px_um = pixel_size_from_tif(next(iter(files.values())))

    z0, z1 = z_range_fiji
    z_idx = list(range(z0 - 1, z1))          # Fiji 1-based inclusive → 0-based
    stacks = {ch: load_planes(p, z_idx) for ch, p in files.items()}
    masks = load_mask_planes(mask_path, z_idx, mask_threshold)

    band_in_px, band_out_px = band_in_um / px_um, band_out_um / px_um
    smooth_bins = max(1, int(round(smooth_deg * num_points / 360)))
    bg_struct = disk(max(1, int(round(bg_dilate_um / px_um / 4))))

    prof_rows, metric_rows, corr_rows = [], [], []
    per_z = {ch: [] for ch in channels}      # list of (z_fiji, I) for kymographs
    arc_list, geom = [], {}

    for k, z in enumerate(z_idx):
        z_fiji = z + 1
        filled = clean_cell_mask(masks[k], px_um, gap_close_um=gap_close_um, open_um=open_um)
        if filled.sum() < min_fill_area_px:
            print(f"   ⚠️ {cell} z{z_fiji}: mask too small, skipped")
            continue
        props = regionprops(filled.astype(np.uint8))[0]
        center = props.centroid
        r_out_fine = outer_radius_profile(filled, center, num_points * oversample)
        if band_mode == "ridge":
            # band centred on the actin cortex line instead of the mask's outer edge
            r_out_fine = ridge_radius_profile(stacks[ridge_channel][k], center, r_out_fine, px_um,
                                              search_um=ridge_search_um, smooth_deg=ridge_smooth_deg)
        r_out = r_out_fine.reshape(num_points, oversample).mean(axis=1)
        arc = arc_length_per_bin(r_out, px_um)
        arc_list.append(arc)
        geom[z_fiji] = (center, r_out_fine)

        # background: median outside a dilated cell mask (dilate on 4× downsampled grid)
        small = filled[::4, ::4]
        outside = ~nd_binary_dilation(small, structure=bg_struct)

        for ch, st in stacks.items():
            img = st[k]
            bg = float(np.median(img[::4, ::4][outside])) if outside.any() else 0.0
            raw = sample_band(img, center, r_out_fine, num_points, band_in_px, band_out_px,
                              oversample=oversample, reduce=reduce)
            I = uniform_filter1d(raw - bg, size=smooth_bins, mode="wrap")
            per_z[ch].append((z_fiji, I))
            m, E, _, _ = profile_metrics(I, arc, roi_window_deg, region_method, region_frac, region_threshold, region_smooth_deg)
            metric_rows.append(dict(cell=cell, channel=ch, z=z_fiji, background=bg,
                                    radius_um=float(r_out.mean() * px_um),
                                    perimeter_um=float(arc.sum()), **m))
            for a, (Ival, Eval) in enumerate(zip(I, E)):
                prof_rows.append(dict(cell=cell, channel=ch, z=z_fiji, angle_deg=a * 360.0 / num_points,
                                      intensity=Ival, enrichment=Eval))

    if not arc_list:
        print(f"❌ {cell}: no valid planes in z range {z_range_fiji}")
        return None

    # --- z-aggregate: mean profile over valid planes ---
    arc_mean = np.mean(arc_list, axis=0)
    zmean, zE, zreg, zpk = {}, {}, {}, {}
    for ch in channels:
        I = np.mean([I for _, I in per_z[ch]], axis=0)
        m, E, reg, pk = profile_metrics(I, arc_mean, roi_window_deg, region_method, region_frac, region_threshold, region_smooth_deg)
        zmean[ch], zE[ch], zreg[ch], zpk[ch] = I, E, reg, pk
        metric_rows.append(dict(cell=cell, channel=ch, z="zmean", background=np.nan,
                                radius_um=float(arc_mean.sum() / (2 * np.pi)),
                                perimeter_um=float(arc_mean.sum()), **m))
        for a, (Ival, Eval) in enumerate(zip(I, E)):
            prof_rows.append(dict(cell=cell, channel=ch, z="zmean", angle_deg=a * 360.0 / num_points,
                                  intensity=Ival, enrichment=Eval))

    # --- correlations: per z and on the z-mean ---
    for A, B in corr_pairs:
        ctrl = control_channel if control_channel in channels and control_channel not in (A, B) else None
        zlist = [z for z, _ in per_z[A]]
        for i, zf in enumerate(zlist):
            Ia, Ib = per_z[A][i][1], per_z[B][i][1]
            Ic = per_z[ctrl][i][1] if ctrl else None
            _, Ea, ra, pa = profile_metrics(Ia, arc_list[i], roi_window_deg, region_method, region_frac, region_threshold, region_smooth_deg)
            _, Eb, rb, pb = profile_metrics(Ib, arc_list[i], roi_window_deg, region_method, region_frac, region_threshold, region_smooth_deg)
            corr_rows.append(dict(cell=cell, A=A, B=B, z=zf,
                                  **pair_correlation(Ia, Ib, Ea, Eb, ra, rb, pa, pb, Ic=Ic, n_perm=n_perm)))
        corr_rows.append(dict(cell=cell, A=A, B=B, z="zmean",
                              **pair_correlation(zmean[A], zmean[B], zE[A], zE[B], zreg[A], zreg[B], zpk[A], zpk[B],
                                                  Ic=zmean[ctrl] if ctrl else None, n_perm=n_perm)))

    df_prof = pd.DataFrame(prof_rows)
    df_met = pd.DataFrame(metric_rows)
    df_corr = pd.DataFrame(corr_rows)
    for df in (df_prof, df_met, df_corr):
        df.insert(0, "experiment", experiment)
    df_prof.to_csv(os.path.join(out_dir, f"{cell}_profiles.csv"), index=False)
    df_met.to_csv(os.path.join(out_dir, f"{cell}_metrics.csv"), index=False)
    df_corr.to_csv(os.path.join(out_dir, f"{cell}_correlations.csv"), index=False)

    if save_plots:
        mid_z = sorted(geom)[len(geom) // 2]
        k_mid = z_idx.index(mid_z - 1)
        plot_overlay(cell, stacks, k_mid, mid_z, geom[mid_z], band_in_px, band_out_px,
                     zE, zreg, out_dir)
        plot_qc_montage(cell, stacks[next(iter(channels))], z_idx, geom, band_in_px, band_out_px, out_dir)
        plot_profiles(cell, zE, zreg, out_dir)
        plot_zkymos(cell, per_z, out_dir)
        for A, B in corr_pairs:
            plot_pair(cell, A, B, zmean, zE, zreg, df_corr, out_dir)

    print(f"✅ {cell}: {len(arc_list)} planes, px={px_um:.4f} µm")
    return df_prof, df_met, df_corr


#%% --- Plotting ---

CH_COLORS = {"488": "tab:green", "561": "tab:red", "647": "tab:purple"}


def _ray_xy(center, r, deg_per):
    th = np.deg2rad(np.arange(len(r)) * deg_per)
    return center[1] + np.cos(th) * r, center[0] - np.sin(th) * r


def plot_overlay(cell, stacks, k, z_fiji, geom, bin_px, bout_px, zE, zreg, out_dir):
    center, r_fine = geom
    dp = 360.0 / len(r_fine)
    chs = list(stacks)
    fig, axs = plt.subplots(1, len(chs), figsize=(4.5 * len(chs), 4.8))
    for ax, ch in zip(np.atleast_1d(axs), chs):
        img = stacks[ch][k]
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
        for rr, ls in [(r_fine, "-"), (r_fine - bin_px, ":"), (r_fine + bout_px, ":")]:
            x, y = _ray_xy(center, rr, dp)
            ax.plot(np.append(x, x[0]), np.append(y, y[0]), ls, color="yellow", lw=0.6)
        # accumulation region of this channel (z-mean) drawn as an outer arc
        reg = np.repeat(zreg[ch], len(r_fine) // len(zreg[ch]))
        x, y = _ray_xy(center, r_fine + 3 * bout_px, dp)
        ax.scatter(x[reg], y[reg], s=1, color=CH_COLORS.get(ch, "cyan"))
        ax.plot(center[1], center[0], "+", color="yellow")
        ax.set_title(f"{ch} — z{z_fiji} (arc = z-mean accumulation region)", fontsize=9)
        ax.axis("off")
    fig.suptitle(cell)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cell}_overlay_z{z_fiji}.png"), dpi=150)
    plt.close()


def plot_qc_montage(cell, stack, z_idx, geom, bin_px, bout_px, out_dir):
    zs = sorted(geom)
    n = len(zs)
    cols = min(7, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.3 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, zf in zip(axs, zs):
        img = stack[z_idx.index(zf - 1)]
        center, r_fine = geom[zf]
        ax.imshow(img[::2, ::2], cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
        dp = 360.0 / len(r_fine)
        for rr in (r_fine - bin_px, r_fine + bout_px):
            x, y = _ray_xy(center, rr, dp)
            ax.plot(np.append(x, x[0]) / 2, np.append(y, y[0]) / 2, color="yellow", lw=0.5)
        ax.set_title(f"z{zf}", fontsize=8)
    fig.suptitle(f"{cell} — cortical band (488)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cell}_qc_band_montage.png"), dpi=110)
    plt.close()


def plot_profiles(cell, zE, zreg, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for ch, E in zE.items():
        ang = np.arange(len(E)) * 360.0 / len(E)
        ax.plot(ang, E, color=CH_COLORS.get(ch), label=ch)
        ax.fill_between(ang, 0, E.max() * 1.05, where=zreg[ch], color=CH_COLORS.get(ch), alpha=0.08, step="mid")
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Enrichment (I / median)")
    ax.set_title(f"{cell} — unwrapped cortex, z-mean (shaded = accumulation region)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cell}_profiles_zmean.png"), dpi=200)
    plt.close()


def plot_zkymos(cell, per_z, out_dir):
    chs = list(per_z)
    fig, axs = plt.subplots(len(chs), 1, figsize=(8, 2.0 * len(chs) + 0.6), sharex=True)
    for ax, ch in zip(np.atleast_1d(axs), chs):
        zs = [z for z, _ in per_z[ch]]
        K = np.vstack([I / (np.median(np.clip(I, 0, None)) + 1e-12) for _, I in per_z[ch]])
        im = ax.imshow(K, aspect="auto", cmap="magma", origin="lower",
                       extent=[0, 360, zs[0] - 0.5, zs[-1] + 0.5],
                       vmin=np.percentile(K, 1), vmax=np.percentile(K, 99))
        ax.set_ylabel(f"{ch}\nz (Fiji)")
        plt.colorbar(im, ax=ax, label="Enrich.")
    np.atleast_1d(axs)[-1].set_xlabel("Angle (°)")
    fig.suptitle(f"{cell} — z × angle kymographs")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cell}_zkymographs.png"), dpi=200)
    plt.close()


def plot_pair(cell, A, B, zmean, zE, zreg, df_corr, out_dir):
    sub = df_corr[(df_corr.A == A) & (df_corr.B == B)]
    zrow = sub[sub.z == "zmean"].iloc[0]
    per = sub[sub.z != "zmean"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axs[0]
    ax.scatter(zE[A], zE[B], s=6, c=np.arange(len(zE[A])), cmap="twilight")
    ax.set_xlabel(f"{A} enrichment")
    ax.set_ylabel(f"{B} enrichment")
    ax.set_title(f"r={zrow.pearson_r:.2f}, ρ={zrow.spearman_rho:.2f}, shift-p={zrow.shift_null_p:.3f}", fontsize=9)
    ax = axs[1]
    xc = circ_xcorr(zmean[A], zmean[B])
    N = len(xc)
    lags = (np.arange(N) + N // 2) % N - N // 2
    o = np.argsort(lags)
    ax.plot(lags[o] * 360.0 / N, xc[o], color="k")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"Lag of {B} vs {A} (°)")
    ax.set_ylabel("Circular xcorr")
    ax.set_title(f"max={zrow.xcorr_max:.2f} at {zrow.xcorr_lag_deg:.0f}°", fontsize=9)
    ax = axs[2]
    zz = per.z.astype(int)
    ax.plot(zz, per.pearson_r, "-o", ms=3, color="k", label="Pearson r")
    ax.plot(zz, per.B_enrich_in_A_region, "-o", ms=3, color=CH_COLORS.get(B), label=f"{B} enrich. in {A} region")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhline(1, color=CH_COLORS.get(B), lw=0.6, ls="--")
    ax.set_xlabel("z (Fiji)")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{cell} — {A} vs {B}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cell}_corr_{A}_vs_{B}.png"), dpi=200)
    plt.close()


def plot_summary(df_met, df_corr, df_prof, corr_pairs, out_root, align_channel="561", region_smooth_deg=20):
    zc = df_corr[df_corr.z == "zmean"]
    zm = df_met[df_met.z == "zmean"]

    # 1) per-cell correlation summary
    for A, B in corr_pairs:
        s = zc[(zc.A == A) & (zc.B == B)].sort_values("cell")
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        x = np.arange(len(s))
        cols = ["tab:blue" if p < 0.05 else "lightgray" for p in s.shift_null_p]
        axs[0].bar(x - 0.2, s.pearson_r, width=0.4, color=cols, label="Pearson r")
        if s.partial_r_ctrl.notna().any():
            axs[0].bar(x + 0.2, s.partial_r_ctrl, width=0.4, color="k", alpha=0.6, label="partial r | control")
            axs[0].legend(frameon=False, fontsize=7)
        axs[0].set_ylabel("r (z-mean)")
        axs[0].set_title("blue: circular-shift p < 0.05", fontsize=9)
        axs[1].bar(x, s.B_enrich_in_A_region, color=CH_COLORS.get(B))
        axs[1].axhline(1, color="k", ls="--", lw=0.7)
        axs[1].set_ylabel(f"{B} enrichment in {A} region")
        axs[2].bar(x, s.peak_offset_deg, color="grey")
        axs[2].set_ylabel(f"Peak offset {B}−{A} (°)")
        for ax in axs:
            ax.set_xticks(x)
            ax.set_xticklabels([c.split("Cell_")[-1] for c in s.cell], rotation=90)
            ax.axhline(0, color="k", lw=0.5)
        fig.suptitle(f"{A} vs {B} — all cells")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, f"summary_corr_{A}_vs_{B}.png"), dpi=200)
        plt.close()

    # 2) accumulation metrics per channel
    mets = ["peak_enrichment", "region_width_deg", "accumulation_score", "polarity_index"]
    fig, axs = plt.subplots(1, len(mets), figsize=(4 * len(mets), 3.6))
    chs = list(zm.channel.unique())
    for ax, m in zip(axs, mets):
        data = [zm[zm.channel == ch][m].values for ch in chs]
        ax.boxplot(data, tick_labels=chs, showfliers=False)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)),
                       d, s=14, color=CH_COLORS.get(chs[i]), zorder=3)
        ax.set_title(m, fontsize=10)
    fig.suptitle("Accumulation metrics (z-mean profiles, one dot per cell)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_root, "summary_accumulation_metrics.png"), dpi=200)
    plt.close()

    # 3) profiles aligned on the align_channel peak, mean ± sem over cells
    pz = df_prof[df_prof.z == "zmean"]
    if align_channel in pz.channel.unique():
        aligned = {ch: [] for ch in chs}
        for cell, g in pz.groupby("cell"):
            ref = g[g.channel == align_channel].sort_values("angle_deg").enrichment.values
            k = max(1, int(round(region_smooth_deg * len(ref) / 360)))
            sh = len(ref) // 2 - int(np.argmax(uniform_filter1d(ref, size=k, mode="wrap")))
            for ch in chs:
                e = g[g.channel == ch].sort_values("angle_deg").enrichment.values
                aligned[ch].append(np.roll(e, sh))
        N = len(ref)
        ang = (np.arange(N) - N // 2) * 360.0 / N
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for ch in chs:
            M = np.vstack(aligned[ch])
            mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros(N)
            ax.plot(ang, mu, color=CH_COLORS.get(ch), label=f"{ch} (n={len(M)})")
            ax.fill_between(ang, mu - se, mu + se, color=CH_COLORS.get(ch), alpha=0.25)
        ax.axhline(1, color="k", lw=0.6, ls="--")
        ax.set_xlabel(f"Angle relative to {align_channel} peak (°)")
        ax.set_ylabel("Enrichment (mean ± sem)")
        ax.set_xlim(-180, 180)
        ax.legend(frameon=False)
        ax.set_title(f"Cortical profiles aligned on {align_channel} accumulation")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, f"summary_aligned_on_{align_channel}.png"), dpi=200)
        plt.close()


#%% --- Batch ---

def run_batch(cell_ids, corr_pairs, out_root, align_channel="561", **kwargs):
    region_smooth_deg = kwargs.get("region_smooth_deg", 20)
    os.makedirs(out_root, exist_ok=True)
    P, M, C = [], [], []
    for cid in cell_ids:
        try:
            res = analyze_cell(cid, out_root=out_root, corr_pairs=corr_pairs, **kwargs)
        except Exception as e:
            print(f"❌ Cell {cid}: {e}")
            continue
        if res is None:
            continue
        P.append(res[0]); M.append(res[1]); C.append(res[2])
    if not M:
        print("❌ Nothing processed.")
        return None
    df_prof, df_met, df_corr = pd.concat(P), pd.concat(M), pd.concat(C)
    df_prof.to_csv(os.path.join(out_root, "profiles_all.csv"), index=False)
    df_met.to_csv(os.path.join(out_root, "metrics_all.csv"), index=False)
    df_corr.to_csv(os.path.join(out_root, "correlations_all.csv"), index=False)
    plot_summary(df_met, df_corr, df_prof, corr_pairs, out_root, align_channel=align_channel,
                 region_smooth_deg=region_smooth_deg)
    print(f"✅ Results in {out_root}")
    return df_prof, df_met, df_corr


def resolve_pairs(pairs, channels):
    """'all' → every channel combination; else list of 'A-B' strings or tuples."""
    if pairs in ("all", ["all"]):
        return list(combinations(channels, 2))
    out = []
    for p in pairs:
        a, b = p.split("-") if isinstance(p, str) else p
        if a not in channels or b not in channels:
            raise ValueError(f"Unknown channel in pair {p}; available {list(channels)}")
        out.append((a, b))
    return out


#%%Define variables

experiment = "FNV082_D5"   # prefixed to every cell ID (e.g. FNV082_D5_Cell_001) so experiments can be merged
data_root = "/Volumes/diz/Filipe/2.Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/FNV082_260923_1XA10-LatB-7XD2-activation/WellD5-7XFND2-Light/WellD5-7XD2/Processed"
channels = {"488": "Channel2_488", "561": "Channel3_561", "647": "Channel4_647"}   # name → subfolder
mask_subfolder = "Channel2_488"
results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

cell_ids = [1, 2, 26, 24, 23, 17, 16, 15, 9]   # visible 7XFN-CRY2 polarization
z_range_fiji = (9, 21)                          # inclusive, Fiji 1-based slice numbers
corr_pairs = ["561-647"]                        # or "all", or e.g. ["561-647", "488-647"]


#%%Run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=experiment,
                    help="experiment name prefixed to every cell ID (e.g. FNV082_D5 → FNV082_D5_Cell_001); '' = no prefix")
    ap.add_argument("--data-root", default=data_root)
    ap.add_argument("--out", default=results_root)
    ap.add_argument("--cells", type=int, nargs="+", default=cell_ids)
    ap.add_argument("--z", type=int, nargs=2, default=z_range_fiji, metavar=("ZSTART", "ZEND"),
                    help="inclusive Fiji (1-based) slice range")
    ap.add_argument("--pairs", nargs="+", default=corr_pairs, help="'all' or A-B pairs, e.g. 561-647 488-647")
    ap.add_argument("--band-mode", choices=["ridge", "outer"], default="ridge",
                    help="ridge: band centred on the actin cortex line; outer: band from the mask outer edge (round 1)")
    ap.add_argument("--band-in-um", type=float, default=0.3, help="band extent inside the reference line (outer mode: use 0.8)")
    ap.add_argument("--band-out-um", type=float, default=0.3, help="band extent outside the reference line (outer mode: use 0.2)")
    ap.add_argument("--gap-close-um", type=float, default=1.0)
    ap.add_argument("--open-um", type=float, default=1.0, help="opening radius to remove protrusions (0 = off)")
    ap.add_argument("--ridge-search-um", type=float, default=2.5)
    ap.add_argument("--ridge-smooth-deg", type=float, default=15)
    ap.add_argument("--reduce", choices=["mean", "max"], default="mean")
    ap.add_argument("--smooth-deg", type=float, default=5)
    ap.add_argument("--region-method", choices=["halfmax", "threshold"], default="halfmax")
    ap.add_argument("--region-frac", type=float, default=0.5)
    ap.add_argument("--region-threshold", type=float, default=1.5)
    ap.add_argument("--region-smooth-deg", type=float, default=20)
    ap.add_argument("--control-channel", default="488", help="channel regressed out for partial r ('' to disable)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--align-channel", default="561")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    pairs = resolve_pairs(a.pairs if a.pairs != ["all"] else "all", channels)
    run_batch(
        a.cells, pairs, a.out,
        align_channel=a.align_channel,
        data_root=a.data_root,
        channels=channels,
        mask_subfolder=mask_subfolder,
        z_range_fiji=tuple(a.z),
        band_mode=a.band_mode,
        band_in_um=a.band_in_um,
        band_out_um=a.band_out_um,
        gap_close_um=a.gap_close_um,
        open_um=a.open_um,
        ridge_search_um=a.ridge_search_um,
        ridge_smooth_deg=a.ridge_smooth_deg,
        reduce=a.reduce,
        smooth_deg=a.smooth_deg,
        region_method=a.region_method,
        region_frac=a.region_frac,
        region_threshold=a.region_threshold,
        region_smooth_deg=a.region_smooth_deg,
        control_channel=a.control_channel,
        n_perm=a.n_perm,
        save_plots=not a.no_plots,
        experiment=a.experiment,
    )
