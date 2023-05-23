#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Find the best G coupling parameter based on healthy controls -- Version 2.1
Last edit:  2023/03/27
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file 
            - Release notes:
                * Added disconnectomics and nodal damage
To do:      - 
Comments:   Current implementation is for the  AAL atlas

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain)
          
"""

# %% ~~ Imports and directories ~~ %%#
# Import needed packages
import json
import numpy as np
import pandas as pd
import glob
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

REL_SES = "M00"

RES_DIR = SPINE / "results"
if not Path.is_dir(RES_DIR):
    Path.mkdir(RES_DIR)

LQT_DIR = RES_DIR / "LQT"


# %% ~~ Load data ~~ %%#
def get_layout_subjs():
    print("Getting the layout...")
    layout = BIDSLayout(XCP_DIR, validate=False, config=["bids", "derivatives"])
    subjs = layout.get_subjects()
    print("Done with the layout...")
    return layout, subjs


def check_ts(all_fMRI):
    """This function checks that the timeseries doesn't have rows with all zeros."""
    zeros_ts = []
    for subj, ts in all_fMRI.items():
        zeros_row = np.where(np.all(np.isclose(ts, 0), axis=1))[0]
        if zeros_row.size > 0:
            print(f"Subj-{subj} has some ROI with only 0s...")
            zeros_ts.append(subj)
    dropping_dict = {"excluded subjs": zeros_ts}
    print(
        f"The following patients were discarded for having ROIs with all zeros: {zeros_ts}"
    )
    with open(RES_DIR / "dropped_patients_with_ROIS_with_zeros.json", "w") as f:
        json.dump(dropping_dict, f)
    all_fMRI_cleaned = {
        subj: ts for subj, ts in all_fMRI.items() if subj not in zeros_ts
    }

    return all_fMRI_cleaned


def load_ts_aal(subj):
    sub_ts = np.genfromtxt(
        XCP_DIR
        / f"sub-{subj}"
        / f"ses-{REL_SES}"
        / "func"
        / f"sub-{subj}_ses-M00_task-rest_space-MNI152NLin2009cAsym_atlas-AAL_cort_timeseries.csv",
        delimiter=",",
    )
    return sub_ts


def load_norm_aal_sc():
    sc_list = []
    # Loop through all files that match the pattern "S*_rawcounts.tsv"
    for filename in glob.glob(str(UTL_DIR / "AAL_not_norm" / "S*_rawcounts.csv")):
        # Load the data from the file using numpy
        arr = np.genfromtxt(filename, delimiter=",")
        # Append the loaded data to the list
        sc_list.append(arr)
    sc_mean = np.array(sc_list).mean(axis=0)
    sc_norm = sc_mean * 0.3 / sc_mean.max()
    return sc_norm


def get_classification(subjs):
    global HC_no_WMH, HC_WMH, MCI_no_WMH, MCI_WMH
    adnimerge = pd.read_csv(RES_DIR / "petTOAD_dataframe.csv")
    adnimerge["PTID"] = adnimerge["PTID"].str.replace("_", "")

    HC_no_WMH = adnimerge[
        (adnimerge["PTID"].isin(subjs))
        & ((adnimerge["Group_bin_Fazekas"] == "CN_no_WMH"))
    ]["PTID"].unique()

    HC_WMH = adnimerge[
        (adnimerge["PTID"].isin(subjs)) & ((adnimerge["Group_bin_Fazekas"] == "CN_WMH"))
    ]["PTID"].unique()

    MCI_no_WMH = adnimerge[
        (adnimerge["PTID"].isin(subjs))
        & ((adnimerge["Group_bin_Fazekas"] == "MCI_no_WMH"))
    ]["PTID"].unique()

    MCI_WMH = adnimerge[
        (adnimerge["PTID"].isin(subjs))
        & ((adnimerge["Group_bin_Fazekas"] == "MCI_WMH"))
    ]["PTID"].unique()

    return HC_no_WMH, HC_WMH, MCI_no_WMH, MCI_WMH


def get_group_ts_for_freqs(group_name, all_fMRI_clean):
    if group_name == "HC_noWMH":
        HC_noWMH_fMRI_clean = {
            k: v for k, v in all_fMRI_clean.items() if k in HC_no_WMH
        }
        timeseries = np.array([ts for ts in HC_noWMH_fMRI_clean.values()])
        group = HC_noWMH_fMRI_clean

    elif group_name == "HC_WMH":
        HC_WMH_fMRI_clean = {k: v for k, v in all_fMRI_clean.items() if k in HC_WMH}
        timeseries = np.array([ts for ts in HC_WMH_fMRI_clean.values()])
        group = HC_WMH_fMRI_clean

    elif group_name == "MCI_noWMH":
        MCI_noWMH_fMRI_clean = {
            k: v for k, v in all_fMRI_clean.items() if k in MCI_no_WMH
        }
        timeseries = np.array([ts for ts in MCI_noWMH_fMRI_clean.values()])
        group = MCI_noWMH_fMRI_clean

    elif group_name == "MCI_WMH":
        MCI_WMH_fMRI_clean = {k: v for k, v in all_fMRI_clean.items() if k in MCI_WMH}
        timeseries = np.array([ts for ts in MCI_WMH_fMRI_clean.values()])
        group = MCI_WMH_fMRI_clean

    return group, timeseries


# %%
def get_wmh_load_homogeneous(subjs):
    adnimerge = pd.read_csv(RES_DIR / "petTOAD_dataframe.csv")
    adnimerge["PTID"] = adnimerge["PTID"].str.replace("_", "")
    adnimerge = adnimerge[adnimerge["PTID"].isin(subjs)]
    adnimerge["WMH_load_subj_space_norm"] = (
        adnimerge["WMH_load_subj_space"] - adnimerge["WMH_load_subj_space"].min()
    ) / (
        adnimerge["WMH_load_subj_space"].max() - adnimerge["WMH_load_subj_space"].min()
    )
    homo_wmh_dict = dict(
        zip(adnimerge["PTID"], round(adnimerge["WMH_load_subj_space_norm"], 3))
    )
    return homo_wmh_dict


def get_sc_wmh_weighted(subj, sc_norm):
    wmh_df = pd.read_csv(LQT_DIR / f"sub-{subj}" / "pct_spared_sc_matrix.csv")
    wmh_df = wmh_df.iloc[:200, 1:201]
    sc_wmh_weighted = sc_norm * wmh_df / 100.0
    return sc_wmh_weighted


def get_node_damage(subj):
    wmh_df = pd.read_csv(LQT_DIR / f"sub-{subj}" / "pct_sdc_matrix.csv")
    # Computes the percent damage sustained by each node
    wmh_damage = wmh_df.iloc[:200, 1:201].mean(axis=0).to_numpy()
    return wmh_damage


# %%
