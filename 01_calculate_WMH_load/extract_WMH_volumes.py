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
# %% Imports and directories
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.imagestats as nibstats
from bids import BIDSLayout
from pathlib import Path

# Directories
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
def calculate_WMH_load(subj):
    wmh_mask = layout.get(
        subject=subj, label="WMHMask", return_type="file"
    )
    wmh_mask = [f for f in wmh_mask if "space" not in f]
    if wmh_mask != list():
        wmh = nib.load(wmh_mask[0])
        wmh_volume = np.round(nibstats.mask_volume(wmh), 1)
    else:
        wmh_volume = 0
    return wmh_volume

# Get the BIDS layout and subject list
print('Loading the folder layout')
layout = BIDSLayout(WMH_DIR, validate=False, config=["bids", "derivatives"])
subjs = layout.get_subjects()

# iterate through subjects, since not all subjects have WMH masks, perform WMH volume extraction only on those who have the WMH segmentation masks
global_WMH_dict = {}

for i, subj in enumerate(subjs):
    print(f'Processing sub-{subj}... ({i+1}/{len(subjs)})')
    global_WMH_dict[subj] = calculate_WMH_load(subj)

WMH_subjs_list = []
WMH_burden_list = []

for k, v in global_WMH_dict.items():
    WMH_subjs_list.append(k)
    WMH_burden_list.append(v)

WMH_burden_arr = np.array(WMH_burden_list)

# Normalize the whole group in [0,1]
print('Normalizing the WMH burden across all subjects...')
normalized_WMH_burden_series = np.round(
    (WMH_burden_arr - np.min(WMH_burden_arr)) / np.ptp(WMH_burden_arr), 5
)  

global_WMH_dict_normalized = {
    k: normalized_WMH_burden_series[n] for n, k in enumerate(WMH_subjs_list)
}

# Save global WMH load dict, first open file for writing, "w"
print(f'Saving dictionaries in {WMH_RES_DIR}...')
g = open(WMH_RES_DIR / "global_WMH_burden_all_subjs.pkl", "wb")
# write json object to file
pickle.dump(global_WMH_dict, g)
# close file
g.close()
# Save normalized dict
f = open(WMH_RES_DIR / "global_WMH_burden_all_subjs_normalized.pkl", "wb")
pickle.dump(global_WMH_dict_normalized, f)
f.close()
