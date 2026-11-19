#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:22:28 2025

@author: nunes
"""
#%% --- Imports ---

import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from tifffile import imread as tif_imread, imwrite as tif_imwrite
import imageio.v2 as imageio
from skimage import transform, measure
from scipy.ndimage import binary_erosion, distance_transform_edt,binary_fill_holes, center_of_mass
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, disk, remove_small_objects, convex_hull_image
import sys
import contextlib
from scipy.ndimage import uniform_filter1d
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours, label, regionprops
from skimage.draw import polygon_perimeter
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter1d
from scipy.stats import circmean, circstd, entropy, sem, pearsonr
from General_Functions import load_ilastik_mask, load_image_stack
from glob import glob
import cv2
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shlex
from matplotlib import cm



def robust_fill(mask, gap_close_radius=8, min_fill_ratio=0.5):
    """
    Fill a binary ring mask reliably, even when the ring has gaps.

    Steps:
      1. Morphological closing to bridge small gaps.
      2. binary_fill_holes to fill the enclosed interior.
      3. If the filled area is not substantially larger than the raw mask
         (i.e. fill_holes failed because the ring is still open), fall back
         to the convex hull of the raw mask pixels, which always produces a
         solid polygon regardless of gap size.

    Parameters
    ----------
    mask           : 2-D bool array — raw thresholded mask (ring)
    gap_close_radius: closing disk radius in pixels (default 8)
    min_fill_ratio : if filled_area < raw_area * (1 + min_fill_ratio),
                     treat fill as failed and use convex hull (default 0.5)
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


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()



#%%Polarization functions

# --- Weighted circular mean manually ---
def weighted_circmean(weights):
    N = len(weights)
    angles = np.deg2rad(np.arange(N))
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    return np.arctan2(y, x)  # rad [-pi,pi]

# --- Ensure unwrap_contour_profile returns fixed-length array ---
def unwrap_contour_profile(image, mask, num_points=360, smooth=True, averaging_width=3):
    contours = find_contours(mask, 0.5)
    if not contours:
        return np.zeros(num_points)
    path = contours[0]
    vals = [image[int(y), int(x)] for y, x in path
            if 0 <= int(y) < image.shape[0] and 0 <= int(x) < image.shape[1]]
    profile = np.array(vals, dtype=float)
    if smooth and profile.size > averaging_width:
        profile = uniform_filter1d(profile, size=averaging_width)
    if profile.size > 1 and profile.size != num_points:
        x_old = np.linspace(0, 1, profile.size)
        x_new = np.linspace(0, 1, num_points)
        profile = np.interp(x_new, x_old, profile)
    else:
        profile = np.full(num_points, profile.item() if profile.size==1 else 0.0)
    return profile

# --- Clustering metric functions ---
def resultant_length(weights):
    N = len(weights)
    angles = np.deg2rad(np.arange(N))
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    return np.hypot(x, y) / (np.sum(weights) + 1e-12)

def fwhm_width(weights, frac=0.5):
    N = len(weights)
    peak = weights.max()
    mask = weights >= peak * frac
    mask2 = np.concatenate([mask, mask])
    diffs = np.diff(np.concatenate([[0], mask2.view(np.int8), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths = ends - starts
    max_len = lengths.max() if lengths.size else 0
    return (max_len / N) * 360

def roi_fraction(weights, center_rad, window_size=60):
    N = len(weights)
    deg_per = 360 / N
    center_deg = (np.rad2deg(center_rad) % 360)
    half = int((window_size/2)/deg_per)
    center_idx = int(center_deg/deg_per)
    idxs = np.arange(center_idx-half, center_idx+half+1) % N
    return weights[idxs].sum() / (weights.sum() + 1e-12)
#%%Main analysis with extended metrics and plotting ---
def analyze_polarization_unwrap_v2(
    image_path,
    mask_path,
    mask_threshold=0.5,
    overlay_dir=None,
    erode_mask=True,
    erosion_radius=15,
    contour_dilation=3,
    averaging_width=3,
    num_points=360,
    normalize="none",
    percentile_clip=(1, 99),
    save_annotated=True,
    save_kymo_csv = True,
    roi_window_deg=60,
    R_jump = 0.1,
    gap_close_radius=8,    # morphological closing radius (px) applied before fill_holes
                           # bridges gaps in the ilastik ring mask so fill_holes can
                           # fill the cell interior even when the ring is not perfectly closed
):
    stack = load_image_stack(image_path)
    raw_masks = load_ilastik_mask(mask_path, threshold=mask_threshold)

    kymo = []
    frames = []
    for t, (frame, mask) in enumerate(zip(stack, raw_masks)):
        m = robust_fill(mask, gap_close_radius=gap_close_radius)
        if erode_mask:
            m = binary_erosion(m, disk(erosion_radius))
        lbl = label(m)
        if lbl.max()>0:
            m = (lbl == np.argmax(np.bincount(lbl.flat)[1:])+1)
        c = find_boundaries(m, mode="inner")
        d = binary_dilation(c, disk(contour_dilation))
        d = remove_small_objects(d, min_size=30)
        lbl2 = label(d)
        if lbl2.max()>0:
            d = (lbl2 == np.argmax(np.bincount(lbl2.flat)[1:])+1)

        profile = unwrap_contour_profile(
            frame, d,
            num_points=num_points,
            smooth=True,
            averaging_width=averaging_width
        )
        kymo.append(profile)
        frames.append(t)

        # overlay every N frames
        if overlay_dir and (t % 5 == 0):
            os.makedirs(overlay_dir, exist_ok=True)
            plt.imshow(frame, cmap='gray')
            plt.contour(d, colors='red')
            plt.title(f"Frame {t} overlay")
            plt.savefig(os.path.join(overlay_dir, f"overlay_{t:03d}.png"))
            plt.close()

    kymo_array = np.vstack(kymo)

    # normalization
    if normalize == "max":
        kymo_array = kymo_array / (kymo_array.max(axis=1, keepdims=True)+1e-6)
    elif normalize == "zscore":
        kymo_array = (kymo_array - kymo_array.mean(axis=1, keepdims=True)) / (kymo_array.std(axis=1, keepdims=True)+1e-6)
    elif normalize == "percentile":
        pmin, pmax = np.percentile(kymo_array, percentile_clip)
        kymo_array = np.clip((kymo_array-pmin)/(pmax-pmin+1e-6),0,1)

    # metrics
# ⬇️ Basic Intensity Metric
    mean_int = kymo_array.mean(axis=1)
    max_int  = kymo_array.max(axis=1)
    std_int  = kymo_array.std(axis=1)
    max_idx  = kymo_array.argmax(axis=1)
    max_ang  = (max_idx / num_points) * 360.0
    
    # —— Compute Weighted Center & Drift INLINE ——
    # 1) center per frame (deg)
    centers = np.array([weighted_circmean(row) for row in kymo_array])
    # 2) unwrap & difference → drift (deg/frame)
    rad_un = np.unwrap(np.deg2rad(centers))
    drifts = np.rad2deg(np.diff(rad_un))
    drifts = np.append(drifts, np.nan)   # pad last frame
    
    # —— Other Geometry Metrics ——
    R_vals    = [resultant_length(row)               for row in kymo_array]
    W_vals    = [fwhm_width(row)                     for row in kymo_array]
    ROI_vals  = [roi_fraction(row, c, roi_window_deg) for row, c in zip(kymo_array, centers)]
    
    # —— Build DataFrame with EVERYTHING ——
    df = pd.DataFrame({
        "frame":            frames,
        "mean_intensity":   mean_int,
        "max_intensity":    max_int,
        "std_intensity":    std_int,
        "max_angle_deg":    max_ang,
        "center_deg":       centers,
        "drift_deg":        drifts,
        "resultant_length": R_vals,
        "FWHM_deg":         W_vals,
        "ROI_frac":         ROI_vals
    })
    
    # —— Baseline & Thresholded R-fold ——
    baseline       = 24
    R0             = df.loc[df.frame < baseline, "resultant_length"].mean()
    df["delta_R"]  = df.resultant_length - R0
    df["R_fold"]   = df.resultant_length / (R0 + 1e-6)
    
    # impose a minimum absolute change threshold:
    min_abs_jump            = R_jump   # tune to your noise floor
    df["R_fold_thresh"]     = df.apply(
        lambda r: r.R_fold if r.delta_R >= min_abs_jump else 1.0,
        axis=1
    )
    window_size   = 15       # look‐ahead of 10–20 frames
    min_R_jump    = 0.05     # absolute rise you care about
    
    # 1) compute rolling future max over the next `window_size` frames
    #    R_forward_max[t] = max( R[t], R[t+1], …, R[t+window_size] )
    df["R_forward_max"] = (
        df["resultant_length"]
          .rolling(window=window_size+1, min_periods=1)
          .max()
          .shift(-window_size)   # align so that at frame t we compare to future
    )
    
    # 2) compute your windowed rise
    df["delta_R_window"] = df["R_forward_max"] - df["resultant_length"]
    
    # 3) apply your threshold over windows
    df["R_window_pass"] = df["delta_R_window"] >= min_R_jump
    
    # —— Composite Polarity Indices ——
    # 1) captures any sharpening or broadening
    df["polarity_index"]    = df.resultant_length * df.ROI_frac
    # 2) full: cluster × ROI × coherent motion
    df["full_polarity"]     = df.polarity_index * df.drift_deg.abs()


    # save kymograph image
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(kymo_array, cmap='magma', aspect='auto')
        ax.set_xlabel('Angle (°)')
        ax.set_ylabel('Frame')
        ax.set_title('Kymograph')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, 'kymograph.png'))
        plt.close()
    # — 2) numeric CSV if requested —
    if save_kymo_csv:
        pd.DataFrame(kymo_array).to_csv(
            os.path.join(overlay_dir, 'kymograph.csv'),
            index=False,
            header=False
        )
        # also write TIFF for perfect fidelity
        tif_imwrite(
            os.path.join(overlay_dir, 'kymograph.tif'),
            kymo_array.astype(np.float32)
        )

    # extended metrics plot
    # extended metrics + new 3 panels
    if overlay_dir and save_annotated:
        os.makedirs(overlay_dir, exist_ok=True)
        fig, axs = plt.subplots(4,1,figsize=(6,14), sharex=True)
    
        # existing 4
        axs[0].plot(frames, std_int,      marker='o', label='STD')
        axs[1].plot(frames, df["resultant_length"], marker='o', color='orange', label='Resultant')
        axs[2].plot(frames, df["FWHM_deg"],         marker='o', color='green',  label='FWHM (°)')
        axs[3].plot(frames, df["ROI_frac"],         marker='o', color='red',    label=f'ROI ±{roi_window_deg/2}°')
    
        # new 1: drift
# =============================================================================
#         axs[4].plot(frames, df["drift_deg"],        marker='o', color='purple', label='Drift (°/frame)')
#         # new 2: thresholded R-fold
#         axs[5].plot(frames, df["R_fold_thresh"],    marker='o', color='brown',  label='R_fold_thresh')
#         # new 3: full polarity
#         axs[6].plot(frames, df["full_polarity"],    marker='o', color='black',  label='FullPolarity')
# =============================================================================
    
        for ax in axs:
            ax.legend(loc='upper right')
        axs[-1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, 'extended_metrics_with_new.png'), dpi=200)
        plt.close()
    if overlay_dir:
        plt.figure(figsize=(6,3))
        plt.plot(df["frame"], df["drift_deg"], marker='o', color='purple')
        plt.axhline(0, color='k', lw=1)
        plt.xlabel("Frame")
        plt.ylabel("Drift (°/frame)")
        plt.title("Angular Drift Over Time")
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, "drift_deg_plot.png"), dpi=600)
        plt.close()
    
    # 2) FullPolarity-only plot
    if overlay_dir:
        plt.figure(figsize=(6,3))
        plt.plot(df["frame"], df["full_polarity"], marker='o', color='black')
        plt.xlabel("Frame")
        plt.ylabel("FullPolarity")
        plt.title("Full Polarity Index Over Time")
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, "full_polarity_plot.png"), dpi=600)
        plt.close()
        
    if overlay_dir:
        plt.figure(figsize=(6,3))
        plt.plot(df["frame"], df["R_fold_thresh"], marker='o', color='brown')
        plt.xlabel("Frame")
        plt.ylabel("R_fold_thresh")
        plt.title("R_fold_threshold over time")
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, "R_thresh_plot.png"), dpi=600)
        plt.close()
        
        plt.figure(figsize=(6,3))
        plt.plot(df.frame, df.delta_R_window, label=f"ΔR over next {window_size}f")
        plt.axhline(min_R_jump, color='r', ls='--', label=f"threshold {min_R_jump}")
        plt.xlabel("Frame")
        plt.ylabel("Windowed ΔR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, "delta_R_window.png"))
        plt.close()
        
        plt.figure(figsize=(6,3))
        plt.plot(df.frame, df.R_fold, label="R_fold")
        # highlight windows passing threshold
        pass_idx = df.index[df.R_window_pass]
        plt.scatter(df.frame[pass_idx], df.R_fold[pass_idx], c='red', label="sustained rise")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, "R_fold_with_windows.png"))
        plt.close()


    # ---- Save metrics to CSV ----
    if overlay_dir:
        df.to_csv(os.path.join(overlay_dir, 'polarization_extended_metrics.csv'), index=False)

    return df, kymo_array


#%%Selecting files


def extract_cell_code(filename):
    """
    Extract cell code from filename.
    Works with both OLD and NEW formats.
    
    OLD: C1_block2_W0000_P0005_T0001_smooth.tif → block2_W0000_P0005
    NEW: C1_20260129-153047_block2_W0000_P0001_smooth.tif → block2_W0000_P0001
    """
    match = re.search(r'(block\d+_W\d+_P\d+)', filename)
    if match:
        return match.group(1)
    return None

def extract_timepoint(filename):
    """
    Extract timepoint/timestamp from filename.
    
    NEW format: C1_20260129-153047_block2_... → "20260129-153047"
    OLD format: C1_block2_... → None
    
    Returns the timestamp string or None if not found.
    """
    # Match timestamp pattern: YYYYMMDD-HHMMSS
    match = re.search(r'(\d{8}-\d{6})', filename)
    if match:
        return match.group(1)
    return None


def extract_sample_with_timepoint(filename):
    """
    Extract a unique sample identifier that includes timepoint.
    
    Returns: "20260129-153047_P0001" or just "P0001" if no timestamp
    """
    timepoint = extract_timepoint(filename)
    sample_match = re.search(r'(P\d{4})', filename)
    
    if sample_match:
        sample = sample_match.group(1)
        if timepoint:
            return f"{timepoint}_{sample}"
        return sample
    return None


def find_best_mask_match(image_filename, mask_filenames):
    if "_C2" in image_filename:
        image_filename = image_filename.replace("_C2", "_C1")

    cell_code = extract_cell_code(image_filename)
    if cell_code is None:
        print(f"❌ Cannot extract cell code from {image_filename}")
        return None

    # Also extract the timestamp so we can disambiguate when multiple cells
    # share the same block/well/position code (e.g. different timepoints in
    # the same batch folder).
    timepoint = extract_timepoint(image_filename)

    matches = [f for f in mask_filenames if cell_code in f]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Narrow by timestamp first — avoids cross-timepoint mask swaps
        if timepoint:
            ts_matches = [m for m in matches if timepoint in m]
            if ts_matches:
                matches = ts_matches   # restrict to same timepoint
        preferred = [m for m in matches if m.endswith("_Probabilities.h5")]
        if preferred:
            return preferred[0]
        return matches[0]
    else:
        print(f"❌ No mask found for cell {cell_code}")
        return None

def extract_sample_id_from_merged_csv(filename):
    """
    Extract the full sample identifier from a merged CSV filename.
    
    Handles both formats:
      OLD: P0001_C1_merged.csv → "P0001"
      NEW: 20260129-153047_P0001_C1_merged.csv → "20260129-153047_P0001"
    """
    basename = os.path.basename(filename)
    
    # Try NEW format first: timestamp_sample_channel_merged.csv
    new_match = re.match(r'(\d{8}-\d{6})_(P\d{4})_(C[12])_merged\.csv', basename)
    if new_match:
        timepoint, sample, channel = new_match.groups()
        return f"{timepoint}_{sample}"
    
    # Try OLD format: sample_channel_merged.csv
    old_match = re.match(r'(P\d{4})_(C[12])_merged\.csv', basename)
    if old_match:
        sample, channel = old_match.groups()
        return sample
    
    return None


#%%Conversion to video


from General_Functions import load_image_stack

def stack_to_video(
    image_path,
    output_video_path,
    pixel_size_um,
    scale_bar_length_um,
    time_interval_s,
    fps=10,
    colormap=cv2.COLORMAP_INFERNO,
    # new customization parameters:
    scale_bar_thickness_px=None,     # thickness of bar in pixels (auto if None)
    scale_bar_offset_px=(10,10),     # (x_offset_from_right, y_offset_from_bottom)
    timestamp_offset_px=(10,30),     # (x_offset_from_left, y_offset_from_bottom)
    timestamp_font_scale=0.5,
    timestamp_thickness=1,
    channel_label=None,              # e.g. "ACTIN" or "CRY2"
    label_offset_px=(10,10),         # (x_offset_from_left, y_offset_from_top)
    label_font_scale=0.7,
    label_thickness=2,
):
    """
    Convert stack → video, with configurable scale‐bar, timestamp, and channel label.
    """
    stack = load_image_stack(image_path)    # (T, H, W)
    T, H, W = stack.shape

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H), isColor=True)

    # compute bar in pixels
    bar_px = int(round(scale_bar_length_um / pixel_size_um))
    thickness = scale_bar_thickness_px or max(2, H//200)
    x_off_bar, y_off_bar = scale_bar_offset_px

    x_off_stamp, y_off_stamp = timestamp_offset_px
    font = cv2.FONT_HERSHEY_SIMPLEX

    # label
    if channel_label:
        x_off_lbl, y_off_lbl = label_offset_px

    for i in range(T):
        frame = stack[i]
        # normalize → 8-bit
        img8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color = cv2.applyColorMap(img8, colormap)

        # timestamp (bottom‐left)
        t_sec = i * time_interval_s
        text = f"{t_sec:.1f} s"
        pos_stamp = (x_off_stamp, H - y_off_stamp)
        cv2.putText(color, text, pos_stamp, font,
                    timestamp_font_scale, (255,255,255), timestamp_thickness, cv2.LINE_AA)

        # scale‐bar (bottom‐right)
        x0 = W - x_off_bar - bar_px
        y0 = H - y_off_bar
        x1 = W - x_off_bar
        y1 = y0 - thickness
        cv2.rectangle(color, (x0,y1), (x1,y0), (255,255,255), cv2.FILLED)
        # label the bar
        bar_label = f"{scale_bar_length_um:g} um"
        cv2.putText(color, bar_label, (x0, y1 - 5),
                    font, timestamp_font_scale, (255,255,255), timestamp_thickness, cv2.LINE_AA)

        # channel label (top‐left)
        if channel_label:
            pos_label = (x_off_lbl, y_off_lbl)
            cv2.putText(color, channel_label, pos_label, font,
                        label_font_scale, (255,255,255), label_thickness, cv2.LINE_AA)

        writer.write(color)

    writer.release()
    print(f"✅ Movie saved to {output_video_path}")


def smart_batch_process_and_movie(
    image_folder, mask_folder, output_root,
    pixel_size_um, scale_bar_length_um, time_interval_s,
    **kwargs
):
    os.makedirs(output_root, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder)
                   if f.endswith("_smooth.tif")]
    mask_files = os.listdir(mask_folder)

    for img_file in sorted(image_files):
        if not img_file.startswith(("C1_","C2_")):
            continue

        channel = img_file[:2]         # "C1" or "C2"
        label = "ACTIN" if channel=="C1" else "CRY2"
        base = re.sub(r"C[12]_","", img_file).replace(".tif","")
        out_dir = os.path.join(output_root, f"{base}_{channel}")
        os.makedirs(out_dir, exist_ok=True)

        img_path  = os.path.join(image_folder, img_file)
        mask_name = find_best_mask_match(img_file, mask_files)
        if mask_name is None:
            continue
        mask_path = os.path.join(mask_folder, mask_name)

        # 1) polarization analysis + kymo
        try:
            df, kymo = analyze_polarization_unwrap_v2(
                image_path=img_path,
                mask_path=mask_path,
                overlay_dir=out_dir,
                **kwargs
            )
        except Exception as e:
            print(f"❌ Analysis failed on {base}: {e}")
            continue

        # 2) make annotated movie
        vid_name = f"{base}_{channel}_movie.mp4"
        vid_path = os.path.join(out_dir, vid_name)
        try:
            stack_to_video(
                image_path=img_path,
                output_video_path=vid_path,
                pixel_size_um=pixel_size_um,
                scale_bar_length_um=scale_bar_length_um,
                time_interval_s=time_interval_s,
                fps=10,
                # custom positions/sizes:
                scale_bar_offset_px=(20,20),
                timestamp_offset_px=(20,30),
                scale_bar_thickness_px=4,
                timestamp_font_scale=0.6,
                timestamp_thickness=2,
                channel_label=label,
                label_offset_px=(20,40),
                label_font_scale=0.8,
                label_thickness=2
            )
        except Exception as e:
            print(f"❌ Video failed on {base}: {e}")

#%%Polarity Index calculation

import re
import pandas as pd
from General_Functions import aggregate_polarization_results

#Merge baseline and activation
def stitch_blocks(master_df):
    stitched = []
    # helper to extract the block number or 0
    def blk_key(b):
        m = re.search(r"block(\d+)", b, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    for (samp, ch), grp in master_df.groupby(["sample","channel"], sort=False):
        # pick only non-empty block names
        blocks = [b for b in grp["block"].unique() if b and isinstance(b, str)]
        # sort by the numeric part (block2 before block4, etc.)
        blocks = sorted(blocks, key=blk_key)

        offset = 0
        for blk in blocks:
            sub = grp[grp["block"] == blk].copy()
            # apply the offset so block-4 frames follow block-2
            sub["frame"] = sub["frame"] + offset
            stitched.append(sub)
            # next offset is one frame past this sub-block’s max
            offset = int(sub["frame"].max()) + 1

    # finally, concat everything back together
    if stitched:
        return pd.concat(stitched, ignore_index=True)
    else:
        # no valid blocks found
        return pd.DataFrame(columns=master_df.columns)

#Plot polarization indexes and R variations
def full_pipeline(
    results_root,
    metrics_csv="polarization_extended_metrics.csv",
    window_size=15,
    min_R_jump=0.05,
    delta_ylim=(0, 0.2),       # global y‐limits for all ΔR plots
    pi_ylim=(0, 1.0),          # global y‐limits for all PI plots
    ma_window=5,               # moving‐average window for PI
    baseline_frame=24,         # frame at which activation begins
    pi_fold_window=15,         # window size for PI‐fold Δ
    min_PI_jump=0.1            # threshold on PI‐fold Δ
):
    # 1) aggregate & stitch
    master_df = aggregate_polarization_results(results_root, output_csv="master_all.csv")
    stitched  = stitch_blocks(master_df)

    for (sample, channel), grp in stitched.groupby(["sample","channel"], sort=False):
        grp = grp.sort_values("frame").reset_index(drop=True)
        frames = grp.frame.values

        # compute windowed ΔR
        R = grp.resultant_length
        grp["R_forward_max"]   = R.rolling(window=window_size+1, min_periods=1).max().shift(-window_size)
        grp["delta_R_window"]  = (grp["R_forward_max"] - R).clip(lower=0)

        # compute PI and its moving average
        grp["polarity_index"] = R * grp.ROI_frac
        grp["PI_smooth"]     = grp.polarity_index.rolling(
            window=ma_window, center=True, min_periods=1
        ).mean()

        # compute PI-windowed Δ
        PI = grp.polarity_index
        grp["PI_forward_max"]   = PI.rolling(window=pi_fold_window+1, min_periods=1).max().shift(-pi_fold_window)
        grp["delta_PI_window"]  = (grp["PI_forward_max"] - PI).clip(lower=0)

        # save merged CSV
        out_csv = os.path.join(results_root, f"{sample}_{channel}_merged.csv")
        grp.to_csv(out_csv, index=False)

        # light‐blue activation shading span
        span_args = dict(facecolor='lightblue', alpha=0.3)

        # Plot 1: ΔR windowed
        plt.figure(figsize=(6,3))
        plt.plot(frames, grp.delta_R_window, 'o-', label=f"ΔR next {window_size}f")
        plt.axhline(min_R_jump, color='r', ls='--', label=f"thresh {min_R_jump}")
        plt.axvspan(baseline_frame, frames.max(), **span_args)
        plt.ylim(*delta_ylim)
        plt.xlabel("Frame"); plt.ylabel("ΔR window")
        plt.title(f"{sample} {channel}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(results_root, f"{sample}_{channel}_delta_R_window.png"), dpi=600)
        plt.close()

        # Plot 2: Polarity Index raw & smooth
        plt.figure(figsize=(6,3))
        plt.plot(frames, grp.polarity_index, 'o-', alpha=0.4, label="raw PI")
        plt.plot(frames, grp.PI_smooth, '-k', lw=2, label=f"{ma_window}-frame MA")
        plt.axvspan(baseline_frame, frames.max(), **span_args)
        plt.ylim(*pi_ylim)
        plt.xlabel("Frame"); plt.ylabel("Polarity Index")
        plt.title(f"{sample} {channel}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(results_root, f"{sample}_{channel}_polarity_index.png"), dpi=600)
        plt.close()

        # Plot 3: PI fold-change vs baseline
        baseline_PI = grp.polarity_index.iloc[:baseline_frame].mean() + 1e-6
        grp["PI_fold"] = grp.polarity_index / baseline_PI
        grp ["PI_fold_smooth"]= grp.PI_fold.rolling(
            window=ma_window, center=True, min_periods=1
        ).mean()
        plt.figure(figsize=(6,3))
        plt.plot(frames, grp.PI_fold, 'o-', color='purple')
        plt.axhline(1 + min_PI_jump / baseline_PI, color='r', ls='--', label=f"thresh {min_PI_jump}")
        plt.axvspan(baseline_frame, frames.max(), **span_args)
        plt.ylim(1, None)  # fold-change always ≥1
        plt.xlabel("Frame"); plt.ylabel("PI fold")
        plt.title(f"{sample} {channel}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(results_root, f"{sample}_{channel}_PI_fold.png"), dpi=600)
        plt.close()

        # Plot 4: ΔPI windowed
        plt.figure(figsize=(6,3))
        plt.plot(frames, grp.delta_PI_window, 'o-', color='brown')
        plt.axhline(min_PI_jump, color='r', ls='--', label=f"thresh {min_PI_jump}")
        plt.axvspan(baseline_frame, frames.max(), **span_args)
        plt.ylim(0, pi_ylim[1])
        plt.xlabel("Frame"); plt.ylabel(f"ΔPI next {pi_fold_window}f")
        plt.title(f"{sample} {channel}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(results_root, f"{sample}_{channel}_delta_PI_window.png"), dpi=600)
        plt.close()

    print("✅ Full pipeline complete with global axis limits.")

#Generate PDF file
def make_two_reports(
    parent_folder,
    condition_name,
    rwindow_pattern="*_delta_R_window.png",
    pi_pattern     ="*_polarity_index.png",
    r_pdf_name     ="R_window_report.pdf",
    pi_pdf_name    ="polarity_index_report.pdf"
):
    """
    Creates two PDFs under parent_folder:
      1) R-window report   (one page per sample, C1|C2 side by side)
      2) Polarity-index report
    Each page is titled with `condition_name` + sample code.
    """
    def group_by_sample(pattern):
        files = glob(os.path.join(parent_folder, pattern))
        d = {}
        for fp in files:
            fn = os.path.basename(fp)
            m = re.match(r"(P\d{4})_C([12])_", fn)
            if not m: continue
            samp, ch = m.group(1), m.group(2)
            d.setdefault(samp, {})[ch] = fp
        return d

    r_by_sample  = group_by_sample(rwindow_pattern)
    pi_by_sample = group_by_sample(pi_pattern)

    # R-window PDF
    out1 = os.path.join(parent_folder, r_pdf_name)
    with PdfPages(out1) as pdf:
        for samp, chdict in sorted(r_by_sample.items()):
            if {"1","2"} <= set(chdict):
                fig, axs = plt.subplots(1,2,figsize=(8,4))
                for idx, ch in enumerate(["1","2"]):
                    img = plt.imread(chdict[ch])
                    axs[idx].imshow(img); axs[idx].axis("off")
                    axs[idx].set_title(f"{samp} C{ch}", pad=8)
                fig.suptitle(f"{condition_name} — ΔR Window Report", fontsize=16, y=0.98)
                pdf.savefig(fig, bbox_inches="tight",dpi=500); plt.close(fig)
    print(f"✅ Saved R-window report: {out1}")

    # Polarity-index PDF
    out2 = os.path.join(parent_folder, pi_pdf_name)
    with PdfPages(out2) as pdf:
        for samp, chdict in sorted(pi_by_sample.items()):
            if {"1","2"} <= set(chdict):
                fig, axs = plt.subplots(1,2,figsize=(8,4))
                for idx, ch in enumerate(["1","2"]):
                    img = plt.imread(chdict[ch])
                    axs[idx].imshow(img); axs[idx].axis("off")
                    axs[idx].set_title(f"{samp} C{ch}", pad=8)
                fig.suptitle(f"{condition_name} — Polarity Index Report", fontsize=16, y=0.98)
                pdf.savefig(fig, bbox_inches="tight",dpi=500); plt.close(fig)
    print(f"✅ Saved Polarity Index report: {out2}")


#%%Define variables     


results_root   = "/Users/nunes/Desktop/Postdoc_EMBL_Diz_Munoz/Git-Repositories/actin-polarization/4.Development/Delamination_videos/Polarization/"
image_folder = "/Volumes/diz/Filipe/2.Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/FNV032_250527_Fibcons_0X-1X-CRY2_HighExp_Polarization/Processed/S01-1XFNRE2-A10/Stacks/"
mask_folder = "/Volumes/diz/Filipe/2.Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/FNV032_250527_Fibcons_0X-1X-CRY2_HighExp_Polarization/Processed/S01-1XFNRE2-A10/Masks/"
condition_name = "20260129_S01-7XFN-NI2-RE-FACS-Tube2"


#%%Polarization analysis for all cells


image_files = [f for f in os.listdir(image_folder) if f.endswith("smooth_.tif")]

# 🔍 Show file types
for img in sorted(image_files):
    if "C1_" in img:
        print(f"✅ C1 detected: {img}")
    elif "C2_" in img:
        print(f"✅ C2 detected: {img}")
    else:
        print(f"⚠️ Unknown file: {img}")


smart_batch_process_and_movie(
    image_folder, mask_folder, results_root,
    pixel_size_um=0.049,
    scale_bar_length_um=5,
    time_interval_s=10.0,
    mask_threshold=0.5,
    erosion_radius=15,
    erode_mask=True,
    contour_dilation=3,
    averaging_width=3,
    num_points=360,
    normalize="max",
    percentile_clip=(1,99),
    save_annotated=True,
    save_kymo_csv=True,
    R_jump=0.1,
    gap_close_radius=8,    # bridges gaps in the ilastik ring before fill_holes
)



#%%Post-processing and plotting



# --- I. Core Helpers ---



def aggregate_polarization_results(parent_folder, output_csv="master_all.csv"):
    """
    Aggregate polarization results from all subfolders.
    
    Updated to handle new naming with timestamps as timepoint identifiers.
    Folder format: 20260129-153047_block2_W0000_P0001_smooth_C1
    """
    rows = []
    for path in glob(os.path.join(parent_folder, "**", "polarization_extended_metrics.csv"), recursive=True):
        df = pd.read_csv(path)
        folder = os.path.basename(os.path.dirname(path))
        
        # Extract sample (P0001)
        m_samp = re.search(r"(P\d{4})", folder)
        sample = m_samp.group(1) if m_samp else ""
        
        # Extract channel (C1 or C2) - at the end of folder name
        m_chan = re.search(r"(C[12])$", folder)
        channel = m_chan.group(1) if m_chan else ""
        
        # Extract block (block2, block4, etc.)
        m_blk = re.search(r"(block\d+)", folder, re.IGNORECASE)
        block = m_blk.group(1) if m_blk else ""
        
        # Extract timepoint/timestamp (NEW)
        m_tp = re.search(r"(\d{8}-\d{6})", folder)
        timepoint = m_tp.group(1) if m_tp else ""
        
        # Create unique sample ID that includes timepoint
        if timepoint:
            unique_sample = f"{timepoint}_{sample}"
        else:
            unique_sample = sample
        
        df = df.copy()
        df["sample"] = unique_sample  # Use unique sample ID
        df["sample_base"] = sample     # Keep original P0001 for reference
        df["channel"] = channel
        df["block"] = block
        df["timepoint"] = timepoint
        rows.append(df)
    
    if not rows:
        print("⚠️ No polarization data found!")
        return pd.DataFrame()
    
    master = pd.concat(rows, ignore_index=True)
    master.to_csv(os.path.join(parent_folder, output_csv), index=False)
    print(f"✅ Aggregated master CSV → {output_csv}")
    return master



def merge_blocks_per_cell(
    results_root,
    channel_list=["C1", "C2"],
    metrics_fname="polarization_extended_metrics.csv",
    kymo_csv="kymograph.csv",
    kymo_tif="kymograph.tif"
):
    """
    For each unique cell (timepoint + sample) and channel, find block folders,
    merge their metrics & kymographs with continuous frames.
    
    Updated to handle timestamp-based timepoint identification.
    """
    all_dirs = [d for d in os.listdir(results_root)
                if os.path.isdir(os.path.join(results_root, d))]
    
    # Find all unique (timepoint, sample) combinations
    sample_timepoints = set()
    for d in all_dirs:
        # Extract timepoint
        tp_match = re.search(r"(\d{8}-\d{6})", d)
        timepoint = tp_match.group(1) if tp_match else ""
        
        # Extract sample
        samp_match = re.search(r"(P\d{4})", d)
        sample = samp_match.group(1) if samp_match else None
        
        if sample:
            sample_timepoints.add((timepoint, sample))
    
    print(f"📊 Found {len(sample_timepoints)} unique (timepoint, sample) combinations")
    
    for timepoint, samp in sorted(sample_timepoints):
        for ch in channel_list:
            # Find block folders for this specific timepoint+sample+channel
            matching_dirs = []
            for d in all_dirs:
                # Check if folder matches this timepoint, sample, and channel
                has_timepoint = (timepoint == "") or (timepoint in d)
                has_sample = samp in d
                has_channel = d.endswith(f"_{ch}")
                
                if has_timepoint and has_sample and has_channel:
                    matching_dirs.append(d)
            
            # Sort by block number
            def block_sort_key(dirname):
                m = re.search(r"block(\d+)", dirname, re.IGNORECASE)
                return int(m.group(1)) if m else 0
            
            matching_dirs = sorted(matching_dirs, key=block_sort_key)
            
            if len(matching_dirs) < 2:
                if len(matching_dirs) == 1:
                    print(f"⚠️ Only 1 block for {timepoint}_{samp} {ch}: {matching_dirs}")
                continue
            
            print(f"  🔗 Merging {timepoint}_{samp} {ch}: {[os.path.basename(d) for d in matching_dirs]}")
            
            # 1) Merge metrics CSV
            merged_dfs = []
            frame_offset = 0
            for sub in matching_dirs:
                csv_path = os.path.join(results_root, sub, metrics_fname)
                if not os.path.exists(csv_path):
                    print(f"    ⚠️ Missing: {csv_path}")
                    continue
                
                df = pd.read_csv(csv_path)
                df = df.sort_values("frame").reset_index(drop=True)
                df["frame"] = df["frame"] + frame_offset
                merged_dfs.append(df)
                frame_offset = int(df["frame"].max()) + 1
            
            if not merged_dfs:
                continue
            
            merged_metrics = pd.concat(merged_dfs, ignore_index=True)
            
            # Create output name with timepoint if present
            if timepoint:
                out_base = f"{timepoint}_{samp}_{ch}"
            else:
                out_base = f"{samp}_{ch}"
            
            out_metrics = os.path.join(results_root, f"{out_base}_merged.csv")
            merged_metrics.to_csv(out_metrics, index=False)
            print(f"    ✅ Merged metrics: {out_base}_merged.csv")
            
            # 2) Merge kymograph if present
            kymo_parts = []
            for sub in matching_dirs:
                kcsv = os.path.join(results_root, sub, kymo_csv)
                if os.path.exists(kcsv):
                    arr = pd.read_csv(kcsv, header=None).values
                    kymo_parts.append(arr)
            
            if kymo_parts:
                merged_kymo = np.vstack(kymo_parts)
                out_kcsv = os.path.join(results_root, f"{out_base}_kymograph_merged.csv")
                pd.DataFrame(merged_kymo).to_csv(out_kcsv, index=False, header=False)
                out_ktif = os.path.join(results_root, f"{out_base}_kymograph_merged.tif")
                tif_imwrite(out_ktif, merged_kymo.astype(np.float32))
                print(f"    ✅ Merged kymograph: {out_base}_kymograph_merged.csv")
    
    print("✅ All cells merged.")
def stitch_blocks(master_df):
    def blk_key(b):
        m = re.search(r"block(\d+)", b, re.IGNORECASE)
        return int(m.group(1)) if m else 0
    stitched = []
    for (samp,ch), grp in master_df.groupby(["sample","channel"], sort=False):
        blocks = sorted([b for b in grp["block"].unique() if b], key=blk_key)
        offset = 0
        for blk in blocks:
            sub = grp[grp["block"]==blk].copy()
            sub["frame"] += offset
            stitched.append(sub)
            offset = sub["frame"].max()+1
    return pd.concat(stitched, ignore_index=True)

def circular_correlation(a_deg,b_deg):
    a = np.deg2rad(a_deg); b = np.deg2rad(b_deg)
    a_bar = np.angle(np.mean(np.exp(1j*a)))
    b_bar = np.angle(np.mean(np.exp(1j*b)))
    sa = np.sin(a-a_bar); sb = np.sin(b-b_bar)
    return np.sum(sa*sb)/np.sqrt(np.sum(sa**2)*np.sum(sb**2))

def shade_activation_region(ax, baseline_end, max_frame=None, color="lightblue", alpha=0.3):
    """
    Shades the region x>=baseline_end on the given Axes.
    
    ax           : a matplotlib Axes instance
    baseline_end : frame index where activation starts
    max_frame    : if None, uses ax.get_xlim()[1]; otherwise the right x‐limit
    color        : fill color
    alpha        : opacity
    """
    x0 = baseline_end
    x1 = max_frame if max_frame is not None else ax.get_xlim()[1]
    ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0)

# --- II. ΔR Analysis ---

def compute_delta_R_from_perframe(
    df_per_frame,
    condition_name,
    baseline_end=24
):
    """
    Given a DataFrame with columns [sample, channel, frame, resultant_length],
    compute R_baseline, R_activation, and delta_R per sample+channel.
    """
    records = []
    # group by sample & channel
    for (samp, ch), grp in df_per_frame.groupby(["sample","channel"]):
        R = grp["resultant_length"]
        R0 = R[grp.frame < baseline_end].mean()
        R1 = R[grp.frame >= baseline_end].mean()
        records.append({
            "condition": condition_name,
            "sample": samp,
            "channel": ch,
            "R_baseline": R0,
            "R_activation": R1,
            "delta_R": R1 - R0
        })
    summary = pd.DataFrame(records)
    out = os.path.join(results_root, f"delta_R_summary_{condition_name}.csv")
    summary.to_csv(out, index=False)
    print(f"✅ ΔPI summary → {out}")
    return summary

def plot_delta_R_summary(dR, results_root, condition_name, plot_fontsize=20):
    df = dR.copy()
    #df['channel'] = df['channel'].map(channel_map)
    # pivot: rows = sample, cols = channel
    piv = df.pivot_table(index='sample', columns='channel', values='delta_R', aggfunc='mean')
    if piv.empty:
        print(f"⚠️ no ΔR data for {condition_name}")
        return

    # grouped bar‐chart
    fig, ax = plt.subplots(figsize=(max(6, len(piv)*0.6), 4))
    piv.plot(kind='bar', ax=ax)
    ax.set_ylabel('ΔR', fontsize=plot_fontsize)
    ax.set_xlabel('Sample', fontsize=plot_fontsize)
    ax.tick_params(axis='x', labelrotation=45, labelsize=plot_fontsize)
    ax.tick_params(axis='y', labelsize=plot_fontsize)
    ax.set_title(f'ΔR per Cell & Channel — {condition_name}', fontsize=plot_fontsize+2)
    ax.legend(fontsize=plot_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(results_root, f'delta_R_percell_bar_{condition_name}.png'), dpi=600)
    plt.close(fig)

def plot_R_mean_timeseries(results_root, condition_name, baseline_end=24):
    allR = {'C1':[], 'C2':[]}
    for fn in glob(os.path.join(results_root, '*_C[12]_merged.csv')):
        code = re.search(r'_(C[12])_merged', fn).group(1)
        arr  = pd.read_csv(fn).sort_values('frame')['resultant_length'].values
        allR[code].append(arr)
    maxL = max((len(a) for arrs in allR.values() for a in arrs), default=0)
    times = np.arange(maxL) * time_interval_min
    fig, ax = plt.subplots(figsize=(8,4))
    for code,color in [('C1','tab:blue'),('C2','tab:orange')]:
        mats = np.vstack([np.pad(a,(0,maxL-len(a)),constant_values=np.nan)
                          for a in allR[code]])
        mean = np.nanmean(mats,axis=0)
        err  = sem(mats,axis=0,nan_policy='omit')
        ax.plot(times, mean, color=color, label=channel_map[code])
        ax.fill_between(times, mean-err, mean+err, color=color, alpha=0.3)
    ax.axvline(baseline_end*time_interval_min, color='gray', ls='--')
    ax.set_xlabel('Time (min)', fontsize=plot_fontsize)
    ax.set_ylabel('R', fontsize=plot_fontsize)
    ax.set_title(f'Mean ± SEM R — {condition_name}', fontsize=plot_fontsize+2)
    ax.legend(fontsize=plot_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(results_root, f'R_mean_{condition_name}.png'), dpi=600)
    plt.close(fig)

# --- III. ΔPI Analysis ---





def compute_delta_PI(
    results_root,
    condition_name,
    baseline_end=24
):
    """
    Computes per-cell ΔPI & fold_PI from each merged CSV.
    
    Updated to handle both naming formats:
      OLD: P0001_C1_merged.csv
      NEW: 20260129-153047_P0001_C1_merged.csv
    """
    records = []

    # Use flexible glob pattern that matches both formats
    for fn in glob(os.path.join(results_root, "*_C[12]_merged.csv")):
        basename = os.path.basename(fn)
        
        # Extract channel
        ch_match = re.search(r'(C[12])_merged\.csv$', basename)
        if not ch_match:
            continue
        ch = ch_match.group(1)
        
        # Extract full sample identifier (handles both formats)
        samp = extract_sample_id_from_merged_csv(fn)
        if not samp:
            continue
        
        df = pd.read_csv(fn)

        # compute the polarity index for every frame
        df["polarity_index"] = df["resultant_length"] * df["ROI_frac"]

        # baseline vs activation
        pi0 = df.loc[df.frame < baseline_end, "polarity_index"].mean()
        pi1 = df.loc[df.frame >= baseline_end, "polarity_index"].mean()

        records.append({
            "condition":     condition_name,
            "sample":        samp,
            "channel":       ch,
            "PI_baseline":   pi0,
            "PI_activation": pi1,
            "delta_PI":      pi1 - pi0,
            "fold_PI":       pi1 / (pi0 + 1e-12)
        })

    summary = pd.DataFrame(records)
    out = os.path.join(results_root, f"delta_PI_summary_{condition_name}.csv")
    summary.to_csv(out, index=False)
    print(f"✅ ΔPI summary → {out}")
    return summary

def plot_delta_PI_summary(dPI, results_root, condition_name, plot_fontsize=20):
    df = dPI.copy()
    df['channel'] = df['channel'].map(channel_map)
    # pivot: rows = sample, cols = channel
    piv = df.pivot_table(index='sample', columns='channel', values='delta_PI', aggfunc='mean')
    if piv.empty:
        print(f"⚠️ no ΔPI data for {condition_name}")
        return

    # grouped bar‐chart
    fig, ax = plt.subplots(figsize=(max(6, len(piv)*0.6), 4))
    piv.plot(kind='bar', ax=ax)
    ax.set_ylabel('ΔPI', fontsize=plot_fontsize)
    ax.set_xlabel('Sample', fontsize=plot_fontsize)
    ax.tick_params(axis='x', labelrotation=45, labelsize=plot_fontsize)
    ax.tick_params(axis='y', labelsize=plot_fontsize)
    ax.set_title(f'ΔPI per Cell & Channel — {condition_name}', fontsize=plot_fontsize+2)
    ax.legend(fontsize=plot_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(results_root, f'delta_PI_percell_bar_{condition_name}.png'), dpi=600)
    plt.close(fig)


def plot_PI_by_condition(df_pf, condition_name, results_root,
                         baseline_end=24, dpi=600):
    fig, axs = plt.subplots(2,1,figsize=(8,6), sharex=True)
    for ax, metric in zip(axs,('delta_PI','fold_PI')):
        for code,color in [('C1','tab:blue'),('C2','tab:orange')]:
            sub = df_pf[df_pf['channel']==channel_map[code]].groupby('frame')[metric]
            m   = sub.mean(); e = sub.apply(lambda x: sem(x, nan_policy='omit'))
            ax.plot(m.index*time_interval_min, m, color=color, label=channel_map[code])
            ax.fill_between(m.index*time_interval_min, m-e, m+e, color=color, alpha=0.3)
        ax.axvline(baseline_end*time_interval_min, color='gray', ls='--')
        ax.set_ylabel(metric, fontsize=plot_fontsize)
        ax.legend(fontsize=plot_fontsize); ax.grid(True)
    axs[-1].set_xlabel('Time (min)', fontsize=plot_fontsize)
    fig.suptitle(f'PI time‐series — {condition_name}', y=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(results_root, f'PI_timeseries_{condition_name}.png'), dpi=dpi)
    plt.close(fig)
    
    
def compute_sliding_corr(x, y, window=10):
    """
    rolling Pearson correlation between x & y
    returns array of length len(x) with np.nan for positions < window
    """
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window-1, len(x)):
        xi = x[i-window+1 : i+1]
        yi = y[i-window+1 : i+1]
        if np.all(np.isfinite(xi)) and np.all(np.isfinite(yi)):
            out[i] = pearsonr(xi, yi)[0]
    return out

def plot_cell_R_PI_corr(results_root, baseline_end=24, corr_window=10, dpi=600):
    """
    Plot R and PI correlation for each cell.
    
    Updated to handle both naming formats.
    """
    traces = {}
    
    # Find all C1 merged files (both formats)
    c1_files = glob(os.path.join(results_root, '*_C1_merged.csv'))
    
    for fn1 in c1_files:
        # Extract sample identifier
        samp = extract_sample_id_from_merged_csv(fn1)
        if not samp:
            continue
        
        # Find matching C2 file
        fn2 = fn1.replace('_C1_merged.csv', '_C2_merged.csv')
        if not os.path.exists(fn2):
            print(f"⚠️ No C2 match for {samp}")
            continue
        
        df1 = pd.read_csv(fn1).sort_values('frame')
        df2 = pd.read_csv(fn2).sort_values('frame')
        m = pd.merge(df1, df2, on='frame', suffixes=('_C1', '_C2'))
        
        t = m['frame'].values * time_interval_min
        R1 = m['resultant_length_C1'].values
        R2 = m['resultant_length_C2'].values
        PI1 = R1 * m['ROI_frac_C1'].values
        PI2 = R2 * m['ROI_frac_C2'].values
        
        corr = np.full_like(t, np.nan)
        for i in range(corr_window - 1, len(t)):
            xi, yi = R1[i - corr_window + 1:i + 1], R2[i - corr_window + 1:i + 1]
            corr[i] = np.corrcoef(xi, yi)[0, 1]
        
        traces[samp] = corr
        
        fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
        
        # R
        axs[0].plot(t, R1, '-o', label='Actin')
        axs[0].plot(t, R2, '-o', label='CRY2')
        axs[0].axvline(baseline_end * time_interval_min, color='gray', ls='--')
        axs[0].set_ylabel('R', fontsize=plot_fontsize)
        axs[0].legend(fontsize=plot_fontsize)
        
        # PI
        axs[1].plot(t, PI1, '-s', label='Actin')
        axs[1].plot(t, PI2, '-s', label='CRY2')
        axs[1].axvline(baseline_end * time_interval_min, color='gray', ls='--')
        axs[1].set_ylabel('PI', fontsize=plot_fontsize)
        axs[1].legend(fontsize=plot_fontsize)
        
        # corr
        axs[2].plot(t, corr, '-o', color='tab:purple', label='R corr')
        axs[2].axhline(0, color='k')
        axs[2].axvline(baseline_end * time_interval_min, color='gray', ls='--')
        axs[2].set_ylabel('Pearson R', fontsize=plot_fontsize)
        axs[2].set_xlabel('Time (min)', fontsize=plot_fontsize)
        axs[2].legend(fontsize=plot_fontsize)
        
        for ax in axs:
            ax.tick_params(axis='both', labelsize=plot_fontsize)
        
        fig.tight_layout()
        # Use full sample identifier in filename
        fig.savefig(os.path.join(results_root, f'{samp}_R_PI_corr_3panel.png'), dpi=dpi)
        plt.close(fig)
    
    return traces

def plot_allcells_corr(traces, results_root, corr_window=10, dpi=600):
    if not traces:
        print("⚠️ plot_allcells_corr: no traces to plot.")
        return
    fig, ax = plt.subplots(figsize=(8,4))
    for samp,corr in traces.items():
        t = np.arange(len(corr)) * time_interval_min
        ax.plot(t, corr, label=samp, alpha=0.7)
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('Time (min)', fontsize=plot_fontsize)
    ax.set_ylabel('Pearson R', fontsize=plot_fontsize)
    ax.legend(fontsize='small', bbox_to_anchor=(1.05,1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(results_root, 'allcells_R_corr.png'), dpi=dpi)
    plt.close(fig)


# --- IV. Orientation Analysis ---

def batch_orientation_correlation_from_master(
    results_root,
    master_csv="master_all.csv",
    output_csv="orientation_summary.csv"
):
    path = os.path.join(results_root, master_csv)
    df = pd.read_csv(path)

    records = []
    for samp, grp_s in df.groupby("sample"):
        # use pivot_table with mean aggregator
        pivot = grp_s.pivot_table(
            index="frame",
            columns="channel",
            values="max_angle_deg",
            aggfunc="mean"    # collapse any duplicates by taking the mean
        )

        if {"C1","C2"}.issubset(pivot.columns):
            a1 = pivot["C1"].values
            a2 = pivot["C2"].values

            # compute circular diff & correlation as before
            diff_rad = np.angle(np.exp(1j*np.deg2rad(a1)) / np.exp(1j*np.deg2rad(a2)))
            diff_deg = np.rad2deg(diff_rad)
            mean_abs_diff = np.mean(np.abs(diff_deg))
            circ_corr     = circular_correlation(a1, a2)

            records.append({
                "sample":                samp,
                "mean_abs_circ_diff_deg": mean_abs_diff,
                "circular_corr":          circ_corr
            })

    summary = pd.DataFrame(records)
    summary.to_csv(os.path.join(results_root, output_csv), index=False)
    print(f"✅ Orientation summary → {output_csv}")
    return summary

def plot_orientation_metrics(
    summary_df,
    results_root,
    abs_diff_thresh=30,
    corr_thresh=0.8,
    out_prefix="orientation"
):
    """
    summary_df: DataFrame with either
      • a 'sample' column, or
      • its index equal to the sample names,
    and columns 'mean_abs_circ_diff_deg' and 'circular_corr'.
    """
    # Ensure sample names
    if "sample" in summary_df.columns:
        samples = summary_df["sample"].astype(str).tolist()
    else:
        # fall back to index
        samples = [str(s) for s in summary_df.index]
        summary_df = summary_df.copy()
        summary_df["sample"] = samples

    # Required metrics
    if summary_df.empty or \
       not {"mean_abs_circ_diff_deg","circular_corr"}.issubset(summary_df.columns):
        print("⚠️ No orientation data to plot.")
        return

    mad = summary_df["mean_abs_circ_diff_deg"].values
    cc  = summary_df["circular_corr"].values
    x   = np.arange(len(samples))
    w   = 0.35

    # — Bar plot —
    fig, ax1 = plt.subplots(figsize=(max(6,len(samples)*0.4), 4))
    ax1.bar(x - w/2, mad, w,
            color=np.where(mad<=abs_diff_thresh, "tab:green", "lightgray"),
            label="mean |Δθ| (°)")
    ax1.axhline(abs_diff_thresh, color="tab:green", ls="--",
                label=f"thresh={abs_diff_thresh}°")
    ax1.set_xticks(x)
    ax1.set_xticklabels(samples, rotation=45, ha="right")
    ax1.set_ylabel("mean |Δθ| (°)")

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, cc, w,
            color=np.where(cc>=corr_thresh, "tab:blue", "lightgray"),
            label="circular_corr")
    ax2.axhline(corr_thresh, color="tab:blue", ls="--",
                label=f"thresh={corr_thresh}")
    ax2.set_ylabel("circular_corr")

    # combine legends
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")
    ax1.set_title("Orientation Metrics — Bar Plot")
    plt.tight_layout()
    bar_out = os.path.join(results_root, f"{out_prefix}_bar.png")
    fig.savefig(bar_out, dpi=600)
    plt.close(fig)
    print(f"✅ Saved {bar_out}")

    # — Box plot —
    fig, (b1,b2) = plt.subplots(1,2,figsize=(6,4))
    b1.boxplot(mad, labels=["|Δθ|"])
    b1.axhline(abs_diff_thresh, color="tab:green", ls="--")
    b1.set_title("mean |Δθ| dist")

    b2.boxplot(cc, labels=["corr"])
    b2.axhline(corr_thresh, color="tab:blue", ls="--")
    b2.set_title("circular_corr dist")

    plt.suptitle("Orientation Metrics — Distribution", y=0.94)
    plt.tight_layout()
    box_out = os.path.join(results_root, f"{out_prefix}_box.png")
    fig.savefig(box_out, dpi=600)
    plt.close(fig)
    print(f"✅ Saved {box_out}")


# reuse your earlier sliding circular‐corr helper:
def sliding_circular_correlation(a_deg, b_deg, window=10):
    """
    a_deg, b_deg: angle arrays (degrees), same length T
    window: # of frames in each window
    returns corr[t] = circ‐corr over frames [t-window+1 .. t]
    """
    T = len(a_deg)
    corr = np.full(T, np.nan)
    a_rad = np.deg2rad(a_deg)
    b_rad = np.deg2rad(b_deg)

    for t in range(window-1, T):
        ar = a_rad[t-window+1:t+1]
        br = b_rad[t-window+1:t+1]
        # circular means
        a_bar = np.angle(np.mean(np.exp(1j*ar)))
        b_bar = np.angle(np.mean(np.exp(1j*br)))
        sa = np.sin(ar - a_bar)
        sb = np.sin(br - b_bar)
        num = np.sum(sa*sb)
        den = np.sqrt(np.sum(sa**2)*np.sum(sb**2))
        corr[t] = num/(den + 1e-12)
    return corr

def plot_cell_orientation_corr(
    results_root,
    window=10,
    dpi=600
):
    """
    For each Pxxxx, load Pxxxx_C1_merged.csv and Pxxxx_C2_merged.csv,
    extract center_deg_C1 & center_deg_C2, compute sliding circ‐corr,
    and save a per‐cell time‐series plot.
    Returns dict of sample→corr array.
    """
    corr_traces = {}
    for fn1 in glob(os.path.join(results_root, "P????_C1_merged.csv")):
        samp = os.path.basename(fn1).split("_")[0]
        fn2  = os.path.join(results_root, f"{samp}_C2_merged.csv")
        if not os.path.exists(fn2):
            continue

        df1 = pd.read_csv(fn1).sort_values("frame")
        df2 = pd.read_csv(fn2).sort_values("frame")
        # align frames
        df = pd.merge(df1, df2, on="frame", suffixes=("_C1","_C2"))

        t      = df["frame"].values
        cen1   = df["center_deg_C1"].values
        cen2   = df["center_deg_C2"].values
        corr   = sliding_circular_correlation(cen1, cen2, window=window)
        corr_traces[samp] = corr

        # plot
        plt.figure(figsize=(6,3))
        plt.plot(t, corr, '-o', color='tab:purple')
        plt.axhline(0, color='k', lw=1)
        plt.axvline(window, color='gray', ls='--', label="window length")
        plt.xlabel("Frame")
        plt.ylabel(f"Circ‐corr (w={window})")
        plt.title(f"{samp} Orientation Corr over time")
        plt.tight_layout()
        out = os.path.join(results_root, f"{samp}_orientation_corr_ts.png")
        plt.savefig(out, dpi=dpi)
        plt.close()
        print(f"✅ Saved {out}")

    return corr_traces

def plot_allcells_orientation_corr(
    corr_traces,
    results_root,
    window=10,
    dpi=600
):
    """
    Superimpose all per‐cell orientation corr traces.
    """
    plt.figure(figsize=(8,4))
    for samp, corr in sorted(corr_traces.items()):
        plt.plot(corr, label=samp, alpha=0.7)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel("Frame")
    plt.ylabel(f"Circual correlation (w={window})")
    plt.title("Sliding orientation corr — all cells")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small")
    plt.tight_layout()
    out = os.path.join(results_root, f"allcells_orientation_corr_w{window}.png")
    plt.savefig(out, dpi=dpi)
    plt.close()
    print(f"✅ Saved {out}")

def remap_legend(ax=None):
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    new_labels = [channel_map.get(lbl, lbl) for lbl in labels]
    ax.legend(handles, new_labels, fontsize=plot_fontsize)

# load single-block kymo
def load_kymo_from_folder(base_dir, block_prefix, channel_suffix, sample=None, timepoint=None):
    """
    Find folder matching the given criteria and load its kymograph.
    
    Updated to optionally filter by sample and timepoint.
    
    Parameters:
    -----------
    base_dir : str
        Parent directory containing result folders
    block_prefix : str
        Block identifier (e.g., 'block2')
    channel_suffix : str
        Channel identifier (e.g., 'C1')
    sample : str, optional
        Sample identifier (e.g., 'P0001') to filter by
    timepoint : str, optional
        Timepoint/timestamp (e.g., '20260129-153047') to filter by
    """
    # Build pattern
    if timepoint and sample:
        patt = os.path.join(base_dir, f"*{timepoint}*{block_prefix}*{sample}*_{channel_suffix}")
    elif sample:
        patt = os.path.join(base_dir, f"*{block_prefix}*{sample}*_{channel_suffix}")
    else:
        patt = os.path.join(base_dir, f"{block_prefix}*_{channel_suffix}")
    
    folders = glob(patt)
    
    if not folders:
        raise FileNotFoundError(
            f"No folder for block={block_prefix}, channel={channel_suffix}, "
            f"sample={sample}, timepoint={timepoint}"
        )
    
    if len(folders) > 1:
        print(f"⚠️ Multiple matches for {block_prefix}/{channel_suffix}, using first: {folders[0]}")
    
    fld = folders[0]
    
    # Try to load kymograph (tif, png, or csv)
    for ext in ("tif", "png", "csv"):
        files = glob(os.path.join(fld, f"*.{ext}"))
        if not files:
            continue
        
        for f in files:
            if ext == "csv":
                if "kymograph" not in os.path.basename(f).lower():
                    continue
                arr = pd.read_csv(f, header=None, dtype=float).values
                return arr.astype(np.float32)
            
            # For tif/png
            img = tif_imread(f)
            if img.ndim == 3:
                img = img.mean(axis=-1)
            return np.array(img, dtype=np.float32)
    
    raise FileNotFoundError(f"No kymo file in {fld}")



def batch_postprocess_kymos(results_root, channel_map, plots_dir, 
                            time_interval_min, kymo_fontsize, baseline_end):
    """
    Updated per-cell merged kymograph generation.
    
    Properly handles multiple timepoints by grouping folders correctly.
    """
    from glob import glob
    
    all_dirs = [d for d in os.listdir(results_root)
                if os.path.isdir(os.path.join(results_root, d))]
    
    # Find all unique (timepoint, sample) combinations
    cell_identities = set()
    for d in all_dirs:
        tp_match = re.search(r"(\d{8}-\d{6})", d)
        timepoint = tp_match.group(1) if tp_match else ""
        
        samp_match = re.search(r"(P\d{4})", d)
        sample = samp_match.group(1) if samp_match else None
        
        if sample:
            cell_identities.add((timepoint, sample))
    
    print(f"📊 Processing kymographs for {len(cell_identities)} cells")
    
    for chan_key, chan_lbl in channel_map.items():
        for timepoint, samp in sorted(cell_identities):
            try:
                k2 = load_kymo_from_folder(
                    results_root, 'block2', chan_key,
                    sample=samp, timepoint=timepoint
                )
                k4 = load_kymo_from_folder(
                    results_root, 'block4', chan_key,
                    sample=samp, timepoint=timepoint
                )
                
                # Create output filename with timepoint if present
                if timepoint:
                    out_name = f"{timepoint}_{samp}_merged_kymo_{chan_key}.png"
                else:
                    out_name = f"{samp}_merged_kymo_{chan_key}.png"
                
                out = os.path.join(plots_dir, out_name)
                
                plot_merged_kymograph(
                    k2, k4,
                    time_interval_min, kymo_fontsize,
                    baseline_end, out,
                    chan_lbl
                )
                print(f"  ✅ {out_name}")
                
            except FileNotFoundError as e:
                print(f"  ⚠️ Skipping {timepoint}_{samp} {chan_key}: {e}")

# merged kymo swapped axes

def plot_merged_kymograph(kymo2, kymo4,
                          time_interval_min, fontsize,
                          activation_frame, output_path,
                          channel_label):
    merged = np.vstack([kymo2, kymo4]).astype(np.float32)
    n_frames, n_angles = merged.shape
    times = np.arange(n_frames) * time_interval_min
    angles = np.linspace(0, 360, n_angles)

    fig, ax = plt.subplots(figsize=(8,6))
    # now extent = [xmin, xmax, ymin, ymax] = [angle, angle, time, time]
    ax.imshow(merged, cmap='magma', aspect='auto', origin='upper',
              extent=[angles[0], angles[-1], times[-1], times[0]])
    #origin has to be upper to avoid flipping the kymo!
    # X = Angle, Y = Time
    ax.set_xlabel('Angle (°)', fontsize=fontsize)
    ax.set_ylabel('Time (min)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(f'Merged Kymograph ({channel_label})', fontsize=fontsize+2)

    # Activation horizontal line at t0
    t0 = activation_frame * time_interval_min
    ax.axhline(t0, color='white', linestyle='--', linewidth=2)
    ax.text(
        angles[0] + (angles[-1] - angles[0]) * 0.02,
        t0 + (times[0] - times[-1]) * 0.02,
        'Activation',
        color='white',
        fontsize=fontsize,
        verticalalignment='bottom'
    )

    cbar = plt.colorbar(ax.images[0], ax=ax, label='Intensity')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label('Intensity', fontsize=fontsize)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

# --- V. PDF Reports ---

def make_pdf_report(
    results_root,
    report_name,
    patterns,
    titles,
    search_subfolders=True
):
    """
    Assemble a PDF by stacking for each cell+channel the images matching `patterns`.

    If search_subfolders=True (default), looks inside each folder ending in _C1/_C2.
    If False, looks at top-level results_root for files matching each pattern.
    """
    if search_subfolders:
        # 1) subfolder mode
        subs = [
            d for d in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, d))
            and (d.endswith("_C1") or d.endswith("_C2"))
        ]
        # sort by sample then channel
        def sort_key(d):
            m = re.search(r"(P\d{4})", d)
            samp = m.group(1) if m else ""
            ch   = "1" if d.endswith("_C1") else "2"
            return (samp, ch)
        subs = sorted(subs, key=sort_key)

        out_path = os.path.join(results_root, report_name)
        with PdfPages(out_path) as pdf:
            for sub in subs:
                fig, axs = plt.subplots(len(patterns), 1,
                                         figsize=(6, 4*len(patterns)))
                if len(patterns)==1:
                    axs = [axs]
                for i, (pat, title) in enumerate(zip(patterns, titles)):
                    full_pattern = os.path.join(results_root, sub, pat)
                    imgs = glob(full_pattern)
                    if not imgs:
                        print(f"⚠️ No files matching {pat} in {sub}")
                        continue
                    img = plt.imread(imgs[0])
                    axs[i].imshow(img)
                    axs[i].axis("off")
                    axs[i].set_title(f"{sub} — {title}", pad=8)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"✅ PDF report → {out_path}")

    else:
        # 2) top-level mode
        # find all sample codes present in the filenames
        all_files = []
        for pat in patterns:
            all_files += glob(os.path.join(results_root, pat))
        samples = sorted({ re.search(r"P\d{4}", os.path.basename(f))[0]
                           for f in all_files if re.search(r"P\d{4}", f) })

        out_path = os.path.join(results_root, report_name)
        with PdfPages(out_path) as pdf:
            for samp in samples:
                fig, axs = plt.subplots(len(patterns), 1,
                                         figsize=(6, 4*len(patterns)))
                if len(patterns)==1:
                    axs = [axs]
                for i, (pat, title) in enumerate(zip(patterns, titles)):
                    # look top‐level for samp + pattern
                    full_pattern = os.path.join(results_root, pat)
                    imgs = [f for f in glob(full_pattern)
                            if samp in os.path.basename(f)]
                    if not imgs:
                        print(f"⚠️ No files matching {pat} for sample {samp}")
                        continue
                    img = plt.imread(imgs[0])
                    axs[i].imshow(img)
                    axs[i].axis("off")
                    axs[i].set_title(f"{samp} — {title}", pad=8)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"✅ PDF report → {out_path}")

# VI: Make videos


# If ffmpeg on PATH, use:
# ensure /opt/homebrew/bin (where brew-installed ffmpeg lives) is on PATH
os.environ["PATH"] = "/opt/homebrew/bin" + os.pathsep + os.environ.get("PATH", "")

# point to the exact ffmpeg binary
FFMPEG = "/opt/homebrew/bin/ffmpeg"


# ─── Replace your annotate_and_concat_videos with this ───


# ─── USER CONFIG ──────────────────────────────────────────────────────────────
# ensure Homebrew’s ffmpeg is on PATH
os.environ["PATH"] = "/opt/homebrew/bin" + os.pathsep + os.environ.get("PATH", "")
FFMPEG = "/opt/homebrew/bin/ffmpeg"

# Map channel codes to labels
channel_map = {"C1": "Actin", "C2": "CRY2"}

# ─── CORE FUNCTION ────────────────────────────────────────────────────────────

def merge_and_annotate_videos(
    stacks_dir: str,
    time_interval_s: float = 10.0,
    baseline_frames: int = 24,
    fps: int = 6,
    fontsize: int = 30,
    scale_length_px: int = 100,
    scale_thickness_px: int = 5,
    scale_label: str = "10 um",
    activation_box_color: tuple = (255, 216, 173),   # BGR light blue
    activation_box_thickness: int = 4,
    activation_box_padding: int = 20,        # extra px added around the detected cell footprint
    activation_box_threshold_frac: float = 0.15,  # fraction of the frame's 99th-percentile intensity used to detect cell extent
    activation_circle_color: tuple = (255, 0, 0),    # BGR blue
    activation_circle_radius: int = 10,
    activation_circle_gap: int = 8
):
    """
    Reads every C1/C2 block2+block4 TIFF pair in `stacks_dir`,
    concatenates them (block2 then block4), applies the magma colormap,
    and writes an annotated MP4 for each sample+channel.

    Supports both naming formats:
      OLD: C1_block2_W0000_P0005_T0001_smooth.tif
      NEW: C1_20260129-153047_block2_W0000_P0001_smooth.tif

    Annotations per frame:
      • Channel name (top-left)
      • "Baseline" / "Activation" (top-right)
      • Timestamp MM:SS (bottom-left)
      • Scale bar + label (bottom-right)
      • During Activation only: box tightly surrounding the cell (auto-detected from
        the max-intensity projection of the stack, thresholded at
        activation_box_threshold_frac of the 99th percentile, then padded by
        activation_box_padding px — controls activation_box_color/activation_box_thickness)
        and a filled circle under the "Activation" label
        (activation_circle_color/activation_circle_radius)
    """
    
    # Helper functions for parsing (inline to keep function self-contained)
    def extract_timepoint(filename):
        """Extract timestamp (YYYYMMDD-HHMMSS) from filename, or None if not present."""
        match = re.search(r'(\d{8}-\d{6})', filename)
        return match.group(1) if match else None
    
    def extract_sample(filename):
        """Extract sample ID (P0001) from filename."""
        match = re.search(r'(P\d{4})', filename)
        return match.group(1) if match else None
    
    def extract_channel(filename):
        """Extract channel (C1 or C2) from filename."""
        if filename.startswith("C1"):
            return "C1"
        elif filename.startswith("C2"):
            return "C2"
        return None
    
    def extract_block(filename):
        """Extract block identifier from filename."""
        match = re.search(r'(block\d+)', filename, re.IGNORECASE)
        return match.group(1) if match else None
    
    # Group files by (timepoint, sample, channel) to find block2/block4 pairs
    # Structure: {(timepoint, sample, channel): {'block2': path, 'block4': path}}
    file_groups = {}
    
    for fname in os.listdir(stacks_dir):
        if not fname.endswith("smooth.tif"):
            continue
        
        channel = extract_channel(fname)
        sample = extract_sample(fname)
        block = extract_block(fname)
        timepoint = extract_timepoint(fname) or ""  # Empty string for old format
        
        if not all([channel, sample, block]):
            continue
        
        key = (timepoint, sample, channel)
        if key not in file_groups:
            file_groups[key] = {}
        
        file_groups[key][block] = os.path.join(stacks_dir, fname)
    
    # Process each group that has both block2 and block4
    for (timepoint, sample, channel_key), blocks in sorted(file_groups.items()):
        if 'block2' not in blocks or 'block4' not in blocks:
            missing = 'block4' if 'block2' in blocks else 'block2'
            tp_label = f"{timepoint}_" if timepoint else ""
            print(f"⚠️ No {missing} for {tp_label}{sample} {channel_key}")
            continue
        
        path2 = blocks['block2']
        path4 = blocks['block4']
        
        # Load and concatenate
        stack2 = tif_imread(path2)  # shape (n2, h, w)
        stack4 = tif_imread(path4)  # shape (n4, h, w)
        merged = np.concatenate([stack2, stack4], axis=0)  # (n_frames, h, w)
        
        # Normalize to 0–255
        norm = ((merged - merged.min()) / np.ptp(merged) * 255).astype(np.uint8)
        
        # Prepare output filename (include timepoint if present to avoid overwrites)
        if timepoint:
            out_fn = os.path.join(stacks_dir, f"{timepoint}_{sample}_{channel_key}_magma.mp4")
        else:
            out_fn = os.path.join(stacks_dir, f"{sample}_{channel_key}_magma.mp4")
        
        # Prepare video writer
        n_frames, h, w = norm.shape
        writer = cv2.VideoWriter(
            out_fn,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )
        ch_label = channel_map[channel_key]

        # Auto-detect the cell's on-screen footprint (once per video) for the
        # activation box: threshold the max-intensity projection across all
        # frames, then pad. Falls back to the full frame if nothing is found.
        # Only the largest connected component is used — the raw thresholded
        # mask alone also catches scattered single/few-pixel noise blobs
        # elsewhere in the frame, which blew the naive min/max bbox out to
        # nearly 3x the actual cell's height.
        proj = merged.max(axis=0)
        cell_thresh = np.percentile(proj, 99) * activation_box_threshold_frac
        mask = proj > cell_thresh
        labeled_mask = label(mask)
        if labeled_mask.max() > 0:
            props = regionprops(labeled_mask)
            largest = max(props, key=lambda p: p.area)
            y0, x0, y1, x1 = largest.bbox  # bbox is (min_row, min_col, max_row, max_col), max exclusive
            box_x0 = max(x0 - activation_box_padding, 0)
            box_x1 = min(x1 - 1 + activation_box_padding, w - 1)
            box_y0 = max(y0 - activation_box_padding, 0)
            box_y1 = min(y1 - 1 + activation_box_padding, h - 1)
        else:
            box_x0, box_y0, box_x1, box_y1 = 0, 0, w - 1, h - 1
        
        for i in range(n_frames):
            gray = norm[i]
            # Apply magma colormap
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            
            # 1) Channel label (top-left)
            cv2.putText(
                frame,
                ch_label,
                (10, fontsize + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize / 30,
                (255, 255, 255),
                2
            )
            
            # 2) Baseline/Activation (top-right)
            mode = "Baseline" if i < baseline_frames else "Activation"
            (tw, th), _ = cv2.getTextSize(
                mode, cv2.FONT_HERSHEY_SIMPLEX, fontsize/30, 2
            )
            cv2.putText(
                frame,
                mode,
                (w - tw - 10, fontsize + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize / 30,
                (255, 255, 255),
                2
            )

            # 2b) Activation-only visual cues: box around the cell + circle under label
            if mode == "Activation":
                cv2.rectangle(
                    frame,
                    (box_x0, box_y0),
                    (box_x1, box_y1),
                    activation_box_color,
                    thickness=activation_box_thickness
                )

                circle_center = (
                    (w - tw - 10) + tw // 2,
                    fontsize + 5 + th + activation_circle_gap + activation_circle_radius
                )
                cv2.circle(
                    frame,
                    circle_center,
                    activation_circle_radius,
                    activation_circle_color,
                    thickness=-1
                )

            # 3) Timestamp MM:SS (bottom-left)
            total_sec = int(i * time_interval_s)
            ts = f"{total_sec//60:02d}:{total_sec%60:02d}"
            cv2.putText(
                frame,
                ts,
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize / 30,
                (255, 255, 255),
                2
            )
            
            # 4) Scale bar (bottom-right)
            x0 = w - scale_length_px - 10
            y0 = h - scale_thickness_px - 10
            x1 = w - 10
            y1 = h - 10
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
            
            # 5) Scale label just above bar
            (ltw, lth), _ = cv2.getTextSize(
                scale_label, cv2.FONT_HERSHEY_SIMPLEX, fontsize/30, 2
            )
            bar_center_x = x0 + scale_length_px // 2
            label_x = int(bar_center_x - ltw/2) - 20
            label_y = y0 - 8
            
            cv2.putText(
                frame,
                scale_label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize / 30,
                (255, 255, 255),
                2
            )
            
            writer.write(frame)
        
        writer.release()
        print(f"✅ Wrote {out_fn}")

#%%Define batch variables

channel_map = {"C1":"Actin","C2":"CRY2"}
baseline_end = 24
time_interval_s = 10.0
time_interval_min = time_interval_s / 60.0
kymo_fontsize   = 20
plot_fontsize  = 15
video_fontsize = 60
padding = 10
plots_dir = os.path.join(results_root, "plots")
os.makedirs(plots_dir, exist_ok=True)

# merge_and_annotate_videos: activation-only visual cues
activation_box_color          = (255, 0, 0)  # BGR light blue — box around the cell
activation_box_thickness      = 4
activation_box_padding        = 20    # px added around the auto-detected cell footprint (bigger = looser box)
activation_box_threshold_frac = 0.15  # detection sensitivity (lower = includes dimmer/wider signal)
activation_circle_color       = (255, 0, 0)      # BGR blue — filled circle under "Activation"
activation_circle_radius      = 30
activation_circle_gap         = 4
#%%Batch plotting and video generation 

if __name__ == '__main__':
    # 1) Merge and stitch
    merge_blocks_per_cell(
        results_root,
        channel_list=list(channel_map.keys()),
        metrics_fname="polarization_extended_metrics.csv",
        kymo_csv="kymograph.csv",
        kymo_tif="kymograph.tif"
    )

    master   = aggregate_polarization_results(results_root)
    stitched = stitch_blocks(master)
    stitched['channel'] = stitched['channel'].map(channel_map)
    df_pf = stitched.copy()

    # 2) Compute polarization metrics
    df_pf['polarity_index'] = df_pf['resultant_length'] * df_pf['ROI_frac']
    df_pf['PI0'] = (
        df_pf.groupby(['sample','channel'])['polarity_index']
            .transform(lambda x: x.iloc[:baseline_end].mean())
    )
    df_pf['delta_PI'] = df_pf['polarity_index'] - df_pf['PI0']
    df_pf['fold_PI']  = df_pf['polarity_index'] / (df_pf['PI0'] + 1e-6)
    df_pf.to_csv(os.path.join(results_root,'master_per_frame_PI.csv'), index=False)
    
    #3 new plots
    dR = compute_delta_R_from_perframe(df_pf, condition_name, baseline_end)
    plot_delta_R_summary(dR, results_root, condition_name)
    
    dPI = compute_delta_PI(results_root, condition_name, baseline_end)
    plot_delta_PI_summary(dPI, results_root, condition_name)
    
    plot_R_mean_timeseries(results_root, condition_name, baseline_end)
    plot_PI_by_condition(df_pf, condition_name, results_root, baseline_end)
    
    corr_traces = plot_cell_R_PI_corr(results_root, baseline_end, corr_window=10)
    plot_allcells_corr(corr_traces, results_root, corr_window=10)


    # 8) Per-cell merged kymos


    batch_postprocess_kymos(
       results_root, channel_map, plots_dir,
       time_interval_min, kymo_fontsize, baseline_end
       )

    merge_and_annotate_videos(
        stacks_dir=image_folder,
        time_interval_s=10.0,
        baseline_frames=24,
        fps=6,
        fontsize=55,
        scale_length_px=102,
        scale_thickness_px=5,
        scale_label="10um",
        activation_box_color=activation_box_color,
        activation_box_thickness=activation_box_thickness,
        activation_box_padding=activation_box_padding,
        activation_box_threshold_frac=activation_box_threshold_frac,
        activation_circle_color=activation_circle_color,
        activation_circle_radius=activation_circle_radius,
        activation_circle_gap=activation_circle_gap
    )
    print("✅ All done!")


  

#%%Make videos without analysis


if __name__ == '__main__':
    # 1) Merge and stitch
 

    merge_and_annotate_videos(
        stacks_dir=image_folder,
        time_interval_s=10.0,
        baseline_frames=24,
        fps=12,
        fontsize=55,
        scale_length_px=102,
        scale_thickness_px=5,
        scale_label="10um",
        activation_box_color=activation_box_color,
        activation_box_thickness=activation_box_thickness,
        activation_box_padding=activation_box_padding,
        activation_box_threshold_frac=activation_box_threshold_frac,
        activation_circle_color=activation_circle_color,
        activation_circle_radius=activation_circle_radius,
        activation_circle_gap=activation_circle_gap
    )
    print("✅ All done!")


  


#%%
batch_process_drifts(
    parent_folder="/Volumes/diz/Filipe/Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/20250325_Fibcons_CRY2_1X-7X_Polarization/Analysis/S01-7XFN-D2/Polarization_erosion15/",
    method="xcorr"
)

# 2) Or, optical‐flow instead:
# =============================================================================
# batch_process_drifts(
#     parent_folder="/…/Polarization_erosion15/",
#     method="optiflow"
# )
# =============================================================================


#%% DEBUG blocks

image_folder = "/Volumes/diz/Filipe/Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/20250325_Fibcons_CRY2_1X-7X_Polarization/Analysis/S01-7XFN-D2/Stacks_Mask/"
mask_folder = "/Volumes/diz/Filipe/Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/20250325_Fibcons_CRY2_1X-7X_Polarization/Analysis/S01-7XFN-D2/Masks/"
output_root_c1 = "/Volumes/diz/Filipe/Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/20250325_Fibcons_CRY2_1X-7X_Polarization/Analysis/S01-7XFN-D2/C1_Polarization/"
output_root_c2 = "/Volumes/diz/Filipe/Actin_Optogenetics/Cortex_Morphology_Intensity_CRY2/20250325_Fibcons_CRY2_1X-7X_Polarization/Analysis/S01-7XFN-D2/C2_Polarization/"

debug_sample = "block4_W0000_P0005_T0001"
image_path = os.path.join(image_folder, f"C1_{debug_sample}.tif_smooth.tif")
mask_path = os.path.join(mask_folder, f"C1_{debug_sample}.tif_smooth_Probabilities.h5")

print(f"🔬 DEBUGGING {debug_sample}")

df, kymo = analyze_polarization_unwrap_v2(    
    image_path=image_path,
    mask_path=mask_path,
    contour_dilation=3,
    averaging_width=3,
    num_points=360,
    erode_mask=True, erosion_radius=15,         # 🔧 New: normalize each row
    normalize= "max",           # "max" or "zscore"
    percentile_clip=(1, 99), 
    overlay_dir="DEBUG_OVERLAY",
    save_annotated=True
)
