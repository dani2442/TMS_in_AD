#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Evaluate different preprocessing strategies on modeling -- Version 1
Last edit:  2023/03/03
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file to evaluate the impact of preprocessing on modeling
            - Release notes:
                * Initial release
To do:      - 
Comments:   

Sources:    
"""

#%% ~~ Imports and directories ~~ %%#
# Import needed packages
import numpy as np
import pandas as pd
from pathlib import Path
from bids import BIDSLayout

# Directories
SPINE = Path.cwd().parents[2]
DATA_DIR = SPINE / "data"
PREP_DIR = DATA_DIR / "preprocessed"
FPRP_DIR = PREP_DIR / "fmriprep"
UTL_DIR = DATA_DIR / "utils"
WMH_DIR = PREP_DIR / "WMH_segmentation"
XCP_DIR = PREP_DIR / "xcp_d"
P24_DIR = XCP_DIR / "24P"
P24_NI_DIR = XCP_DIR / "24P_ni"
P36_DIR = XCP_DIR / "36P"
P36_NI_DIR = XCP_DIR / "36P_ni"
AROMA_DIR = XCP_DIR / "aroma"
AROMA_NI_DIR = XCP_DIR / "aroma_ni"

REL_SES = "M00"

RES_DIR = SPINE / "results"
if not Path.is_dir(RES_DIR):
    Path.mkdir(RES_DIR)

OUT_XCP_DIR = RES_DIR / "evaluate_xcp_methods"
if not Path.is_dir(OUT_XCP_DIR):
    Path.mkdir(OUT_XCP_DIR)

XCP_SUBDIR_LIST = [P24_DIR, P24_NI_DIR, P36_DIR, P36_NI_DIR, AROMA_DIR, AROMA_NI_DIR]

#%% ~~ Load data ~~ %%#
def get_layout_subjs():
    layout = BIDSLayout(P24_DIR, validate = False, config = ['bids', 'derivatives'])
    subjs = layout.get_subjects()
    return layout, subjs

def get_method_ts(xcp_subdir, subj):
    try:
        sub_ts = np.genfromtxt(xcp_subdir / f"sub-{subj}" / f"ses-{REL_SES}" / "func" / f"sub-{subj}_ses-{REL_SES}_task-rest_run-1_space-MNI152NLin2009cAsym_atlas-Schaefer217_timeseries.tsv", delimiter='\t')
    except:
        sub_ts = np.genfromtxt(xcp_subdir / f"sub-{subj}" / f"ses-{REL_SES}" / "func" / f"sub-{subj}_ses-{REL_SES}_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer217_timeseries.tsv", delimiter='\t')
    sub_ts = sub_ts[4:]
    sub_ts_t = sub_ts.T 
    return sub_ts_t


def get_all_ts(subjs):
    ts_dict = {}
    for subdir in XCP_SUBDIR_LIST:
        meth_name = str(subdir).split('/')[7]
        ts_dict[meth_name] = {}
        for subj in subjs:
            ts_dict[meth_name][subj] = get_method_ts(subdir, subj)
    return ts_dict

def get_sc():
    sc = pd.read_csv(UTL_DIR / "sc_enigma.csv", header=None)
    sc_numpy = sc.to_numpy()
    return sc_numpy