# %%
#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:00:36 2025

@author: nunes
"""
#%% --- Imports ---

import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread as tif_imread, imwrite as tif_imwrite
import imageio.v2 as imageio
from skimage import transform
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm
import h5py
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_erosion, disk

#%%


import os, re, pandas as pd

def aggregate_polarization_results(
    parent_folder,
    output_csv="master_polarization_metrics.csv"
):
    all_data = []

    for root, _, files in os.walk(parent_folder):
        for file in files:
            name = file.lower()
            # pick up both old and extended CSV files
            if not (name.endswith(".csv") and "polarization" in name):
                continue

            full = os.path.join(root, file)
            df = pd.read_csv(full)

            folder = os.path.basename(root)
            parts = folder.split("_")

            # channel is always the last part
            channel = parts[-1]

            # block is always the first part
            block = parts[0]

            # sample is the part matching P\d{4}
            sample = next((p for p in parts if re.fullmatch(r"P\d{4}", p)), "")

            # add them
            df["sample"]  = sample
            df["channel"] = channel
            df["block"]   = block

            all_data.append(df)

    if not all_data:
        print(f"⚠️ No polarization CSVs found in {parent_folder}.")
        return pd.DataFrame()

    master_df = pd.concat(all_data, ignore_index=True)
    outpath   = os.path.join(parent_folder, output_csv)
    master_df.to_csv(outpath, index=False)
    print(f"✅ Master CSV saved to: {outpath}")
    return master_df



#%%

def plot_mean_angle_timeseries(master_df, group_by="sample"):
    plt.figure(figsize=(8, 5))

    for name, group in master_df.groupby(group_by):
        plt.plot(group["frame"], group["angle_center_deg"], label=name, alpha=0.8)

    plt.xlabel("Time (frame)")
    plt.ylabel("Mean Angle (°)")
    plt.title(f"Polarization Angle Over Time by {group_by}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

#%%

def load_image_stack(image_path):
    image_stack = tif_imread(image_path)
    if image_stack.ndim == 3:
        return image_stack  # (time, y, x)
    else:
        raise ValueError(f"Unexpected image shape: {image_stack.shape}")

#%%
def load_ilastik_mask(mask_path, threshold=0.8, erode=True, erosion_radius=15):
    with h5py.File(mask_path, 'r') as h5_file:
        if 'exported_data' in h5_file:
            mask_stack = np.array(h5_file['exported_data'])
        else:
            raise KeyError(f"'exported_data' not found in {mask_path}")

    if mask_stack.ndim == 4:
        mask_stack = mask_stack[:, :, :, 0]

    binary_masks = (mask_stack > threshold).astype(np.uint8)

    return binary_masks


