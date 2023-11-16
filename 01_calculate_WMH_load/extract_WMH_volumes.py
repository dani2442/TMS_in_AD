#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Extract the WMH volumes -- Version 1.1
Last edit:  2023/08/
Authors:    Leone, Riccardo (RL)
Notes:      - ... 
            - Release notes:
                * Removed MNI space since we never use it, so the df is more clean
                * Change names to new standard df_name
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


# Get the BIDS layout and subject list
print("Loading the folder layout...")
layout = BIDSLayout(WMH_DIR, validate=False, config=["bids", "derivatives"])
subjs = layout.get_subjects()
# Calculate the WMH lesion load for each subj and store in pandas df
df_wmh = pd.DataFrame()
df_wmh["PTID"] = subjs
print("Calculating WMH lesion load in subject space...")
df_wmh["WMH_load_subj_space"] = [
    calculate_WMH_load_subj_space(subj) for subj in df_wmh["PTID"]
]
print("Done with WMH lesion load")
# %%
