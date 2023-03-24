#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Find the best G coupling parameter based on healthy controls -- Version 1.1
Last edit:  2023/03/15
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file to evaluate the impact of preprocessing on modeling
            - Release notes:
                * Current xcp_d implementation removes the first 4 timepoints already.
To do:      - 
Comments:   Current version is for AROMA

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""

#%% ~~ Imports and directories ~~ %%#
# Import needed packages
import numpy as np
import pandas as pd
from nilearn.signal import clean
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
AROMA_DIR = XCP_DIR / "aroma"

REL_SES = "M00"

RES_DIR = SPINE / "results"
if not Path.is_dir(RES_DIR):
    Path.mkdir(RES_DIR)

LQT_DIR = RES_DIR / "LQT"

if not Path.is_dir(AROMA_DIR):
    Path.mkdir(AROMA_DIR)

# Set the output directory
OUT_DIR = RES_DIR / "Finding_best_G"
#%% ~~ Load data ~~ %%#
def get_layout_subjs():
    print("Getting the layout...")
    layout = BIDSLayout(AROMA_DIR, validate=False, config=["bids", "derivatives"])
    subjs = layout.get_subjects()
    print("Done with the layout...")
    return layout, subjs


def get_method_ts(subj):
    try:
        sub_ts = np.genfromtxt(
            AROMA_DIR
            / f"sub-{subj}"
            / f"ses-{REL_SES}"
            / "func"
            / f"sub-{subj}_ses-{REL_SES}_task-rest_run-1_space-MNI152NLin2009cAsym_atlas-Schaefer217_timeseries.tsv",
            delimiter="\t",
        )
    except:
        sub_ts = np.genfromtxt(
            AROMA_DIR
            / f"sub-{subj}"
            / f"ses-{REL_SES}"
            / "func"
            / f"sub-{subj}_ses-{REL_SES}_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer217_timeseries.tsv",
            delimiter="\t",
        )
    # The first 4 timepoints are already dropped in my implementation of xcp_d AROMA, transpose as Nxt timeseries
    sub_ts = sub_ts.T
    # Z-score the signal
    sub_ts_post = clean(sub_ts, detrend=False, standardize="zscore", filter=None)
    return sub_ts_post


def get_sc():
    sc = pd.read_csv(UTL_DIR / "Schaefer200_sc.csv", header=None)
    # filter out the names and the subcortical structures
    sc_mat = sc.iloc[1:-35, 1:-35]
    sc_mat_fl = sc_mat.astype("float")
    return sc_mat_fl.to_numpy()


def get_sc_enigma():
    sc = pd.read_csv(UTL_DIR / "sc_enigma.csv", header=None)
    return sc.to_numpy()


def get_classification(subjs):
    adnimerge = pd.read_csv(UTL_DIR / "ADNIMERGE.csv")
    adnimerge["PTID"] = adnimerge["PTID"].str.replace("_", "")
    adnimerge["PTID"] = "ADNI" + adnimerge["PTID"]
    HC = adnimerge[
        (adnimerge["PTID"].isin(subjs))
        & ((adnimerge["DX_bl"] == "CN") | (adnimerge["DX_bl"] == "SMC"))
    ]["PTID"].unique()
    MCI = adnimerge[
        (adnimerge["PTID"].isin(subjs))
        & ((adnimerge["DX_bl"] == "EMCI") | (adnimerge["DX_bl"] == "LMCI"))
    ]["PTID"].unique()
    AD = adnimerge[(adnimerge["PTID"].isin(subjs)) & (adnimerge["DX_bl"] == "AD")][
        "PTID"
    ].unique()
    return HC, MCI, AD


# %%
def get_sc_wmh_weighted(subj, sc_norm):
    wmh_df = pd.read_csv(LQT_DIR / f"sub-{subj}" / "pct_spared_sc_matrix.csv")
    wmh_df = wmh_df.iloc[:200, 1:201]
    sc_wmh_weighted = sc_norm * wmh_df / 100.
    return sc_wmh_weighted

def get_node_damage(subj):
    wmh_df = pd.read_csv(LQT_DIR / f"sub-{subj}" / "pct_sdc_matrix.csv")
    # Computes the percent damage sustained by each node
    wmh_damage = wmh_df.iloc[:200, 1:201].mean(axis=0).to_numpy()
    return wmh_damage