#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Load file for petTOAD -- Version 2.2
Last edit:  2023/08/05
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file 
            - Release notes:
                * Refactored df names to new standard
To do:      - 
Comments:   Current implementation is for the  AAL atlas

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain)
          Skoch et al., Nature Scientific Data, 2022 (for the SC matrix)  
          
"""

# %% ~~ Imports and directories ~~ %%#
# Import needed packages
import glob
import json
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

REL_SES = "M00"

RES_DIR = SPINE / "results"
if not Path.is_dir(RES_DIR):
    Path.mkdir(RES_DIR)

LQT_DIR = RES_DIR / "LQT"

SIM_DIR = RES_DIR / "final_simulations"
if not Path.exists(SIM_DIR):
    Path.mkdir(SIM_DIR)

# %% ~~ Load data ~~ %%#
def get_layout_subjs():
    print("Getting the layout...")
    layout = BIDSLayout(XCP_DIR, validate=False, config=["bids", "derivatives"])
    subjs = layout.get_subjects()
    print("Done with the layout...")
    return layout, subjs

def load_norm_aal_sc():
    sc_list = []
    # Loop through all files that match the pattern "S*_rawcounts.tsv"
    for filename in glob.glob(str(UTL_DIR / "AAL_not_norm" / "S*_rawcounts.csv")):
        # Load the data from the file using numpy
        arr = np.genfromtxt(filename, delimiter=",")
        # Append the loaded data to the list
        sc_list.append(arr)
    sc_mean = np.array(sc_list).mean(axis=0)
    # In the paper by Skoch et al., they state that since they "do not enforce symmetry in any direct
    # artificial way, the matrices are not perfectly symmetrical". In modeling we commonly 
    # symmetrise, so we do the same here.
    sc_mean_sym = (sc_mean + sc_mean.T) / 2
    sc_norm = sc_mean_sym / sc_mean_sym.max()
    return sc_norm

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

def check_ts(all_fMRI):
    """This function checks that the timeseries doesn't have empty rows."""
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

def get_classification(subjs):
    df_petTOAD = pd.read_csv(RES_DIR / "df_petTOAD.csv")
    df_petTOAD["PTID"] = df_petTOAD["PTID"].str.replace("_", "")

    CN_no_WMH = df_petTOAD[
        (df_petTOAD["PTID"].isin(subjs))
        & ((df_petTOAD["Group_bin_Fazekas"] == "CN_no_WMH"))
    ]["PTID"].unique()

    CN_WMH = df_petTOAD[
        (df_petTOAD["PTID"].isin(subjs)) & ((df_petTOAD["Group_bin_Fazekas"] == "CN_WMH"))
    ]["PTID"].unique()

    MCI_no_WMH = df_petTOAD[
        (df_petTOAD["PTID"].isin(subjs))
        & ((df_petTOAD["Group_bin_Fazekas"] == "MCI_no_WMH"))
    ]["PTID"].unique()

    MCI_WMH = df_petTOAD[
        (df_petTOAD["PTID"].isin(subjs))
        & ((df_petTOAD["Group_bin_Fazekas"] == "MCI_WMH"))
    ]["PTID"].unique()

    CN = np.hstack([CN_WMH, CN_no_WMH])
    MCI = np.hstack([MCI_WMH, MCI_no_WMH])
    return CN, MCI, CN_no_WMH, CN_WMH, MCI_no_WMH, MCI_WMH

def define_subjs_to_sim(CN_WMH, MCI_WMH):
    subjs_to_sim_pre = np.hstack([CN_WMH, MCI_WMH])
    df_petTOAD = pd.read_csv(RES_DIR / "df_petTOAD.csv")
    df_petTOAD["PTID"] = df_petTOAD["PTID"].str.replace("_", "")
    df_petTOAD = df_petTOAD[df_petTOAD["WMH_load_subj_space"] < 80000]
    df_petTOAD = df_petTOAD[df_petTOAD["PTID"].isin(subjs_to_sim_pre)]
    return df_petTOAD["PTID"].to_numpy()

def get_group_ts_for_freqs(group_list, all_fMRI_clean):

    group_fMRI_clean = {k: v for k, v in all_fMRI_clean.items() if k in group_list}
    timeseries = np.array([ts for ts in group_fMRI_clean.values()])

    return group_fMRI_clean, timeseries

def get_wmh_load_homogeneous(subjs):
    df_petTOAD = pd.read_csv(RES_DIR / "df_petTOAD.csv")
    df_petTOAD["PTID"] = df_petTOAD["PTID"].str.replace("_", "")
    df_petTOAD = df_petTOAD[df_petTOAD["WMH_load_subj_space"] < 80000]
    df_petTOAD = df_petTOAD[df_petTOAD["PTID"].isin(subjs)]
    df_petTOAD["WMH_load_subj_space_norm"] = (
        df_petTOAD["WMH_load_subj_space"] - df_petTOAD["WMH_load_subj_space"].min()
    ) / (
        df_petTOAD["WMH_load_subj_space"].max() - df_petTOAD["WMH_load_subj_space"].min()
    )
    homo_wmh_dict = dict(
        zip(df_petTOAD["PTID"], round(df_petTOAD["WMH_load_subj_space_norm"], 3))
    )

    return homo_wmh_dict

def get_wmh_load_random(subjs):
    np.random.seed(1991)
    mins = np.random.normal(0.1, 0.01, len(subjs))
    maxs = np.random.normal(0.9, 0.01, len(subjs))
    arr_wmh_rand = np.hstack([mins, maxs])
    np.random.shuffle(arr_wmh_rand)
    wmh_dict_rand = {subj: round(arr_wmh_rand[i],3) for i, subj in enumerate(subjs)}
    return wmh_dict_rand

def get_sc_wmh_weighted(subj):
    spared_sc = pd.read_csv(LQT_DIR / f"sub-{subj}" / "pct_spared_sc_matrix.csv", index_col = 0)
    spared_sc_perc = spared_sc / 100
    return spared_sc_perc

def get_node_spared(subj):
    wmh_df = pd.read_csv(LQT_DIR / f"sub-{subj}" / "pct_spared_sc_matrix.csv", index_col = 0)    
    wmh_spared = wmh_df[wmh_df != 0].mean(axis = 0) / 100
    # For how the code is written now, if a subject has all its connections for a specific row completely damaged,
    # the code outputs a nan. This has to be a 0, because there is no spared connection, so the damage is 100%!
    wmh_spared.iloc[:,] = np.nan_to_num(wmh_spared)
    return wmh_spared.to_numpy()