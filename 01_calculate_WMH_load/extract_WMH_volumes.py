#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Extract the WMH volumes -- Version 1.0
Last edit:  2023/03/20
Authors:    Leone, Riccardo (RL)
Notes:      - Initial release 
            - Release notes:
                * None
To do:      - None
Comments:   

Sources:  
"""
# %%
# Imports and directories
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.imagestats as nibstats
from bids import BIDSLayout
from pathlib import Path

SPINE = Path.cwd().parents[2]
DATA_DIR = SPINE / "data"
PREP_DIR = DATA_DIR / "preprocessed"
FPRP_DIR = PREP_DIR / "fmriprep"
UTL_DIR = DATA_DIR / "utils"
WMH_DIR = PREP_DIR / "WMH_segmentation"

RES_DIR = SPINE / "results"
if not Path.is_dir(RES_DIR):
    Path.mkdir(RES_DIR)

WMH_RES_DIR = RES_DIR / "WMH_lesion_load"
if not Path.is_dir(WMH_RES_DIR):
    Path.mkdir(WMH_RES_DIR)

# Define useful functions
def calculate_WMH_load_subj_space(subj):
    wmh_mask = layout.get(subject=subj, label="WMHMask", return_type="file")
    wmh_mask = [f for f in wmh_mask if "space" not in f]
    try:
        wmh = nib.load(wmh_mask[0])
        wmh_volume = np.round(nibstats.mask_volume(wmh), 1)
    except:
        wmh_volume = 0
    return wmh_volume

#%%
def calculate_WMH_load_mni_space(subj):
    wmh_mask_mni = WMH_DIR / f"sub-{subj}" / f"sub-{subj}_ses-M00_space-MNI152NLin6Asym_desc-fromSubjSpace.nii.gz"
    try:
        wmh_mni = nib.load(wmh_mask_mni)
        wmh_volume_mni = np.round(nibstats.mask_volume(wmh_mni), 1)
    except:
        print(f"MNI mask not found for subj-{subj}")
        wmh_volume_mni = 0    
    return wmh_volume_mni

#%%
# Get the BIDS layout and subject list
print("Loading the folder layout...")
layout = BIDSLayout(WMH_DIR, validate=False, config=["bids", "derivatives"])
subjs = layout.get_subjects()
#%%
# Calculate the WMH lesion load for each subj and store in pandas df
global_WMH_df = pd.DataFrame()
global_WMH_df["PTID"] = subjs
#%%
print("Calculating WMH lesion load in subject space...")
global_WMH_df["WMH_load_subj_space"] = [
    calculate_WMH_load_subj_space(subj) for subj in global_WMH_df["PTID"]
]
# %%
print("Calculating WMH lesion load in mni space...")
global_WMH_df["WMH_load_mni_space"] = [
    calculate_WMH_load_mni_space(subj) for subj in global_WMH_df["PTID"]
]
print("Done with WMH lesion load")
#%%
# Normalize the whole group in [0,1]
print("Normalizing the WMH burden across all subjects in subject space...")
global_WMH_df["WMH_load_norm_subj_space"] = np.round(
    (global_WMH_df["WMH_load_subj_space"] - global_WMH_df["WMH_load_subj_space"].min())
    / np.ptp(global_WMH_df["WMH_load_subj_space"]),
    5,
)

print("Normalizing the WMH burden across all subjects in MNI space...")
global_WMH_df["WMH_load_norm_mni_space"] = np.round(
    (
        global_WMH_df["WMH_load_mni_space"]
        - global_WMH_df["WMH_load_mni_space"].min()
    )
    / np.ptp(global_WMH_df["WMH_load_mni_space"]),
    5,
)

# %%
