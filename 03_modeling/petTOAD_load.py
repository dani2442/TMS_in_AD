#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Find the best G coupling parameter based on healthy controls -- Version 1.2
Last edit:  2023/03/27
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file 
            - Release notes:
                * Added modification from Schaefer 7-networks to 17-networks
To do:      - 
Comments:   Current implementation is for the Schaefer200 parcellation

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain)
          https://github.com/MICA-MNI/ENIGMA/tree/master/enigmatoolbox/datasets/matrices/hcp_connectivity (for ENIGMA SC matrix)


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
    """
    This function loads and transforms the structural connectivity matrix downloaded from the ENIGMA-TOOLBOX
    (which is in the Schaefer200 7-Network atlas into the Schaefer 17-network atlas.

    Args:
        None

    Returns:
        sc_17(np.array): the ENIGMA struc. conn. matrix ordered as the Schaefer200 parcels 17-network atlas
    """
    net17 = pd.read_csv(UTL_DIR / "Schaefer17Net.txt", delimiter = ',')
    net7 = pd.read_csv(UTL_DIR / "Schaefer7Net.txt", delimiter = ',')
    sc = np.loadtxt(UTL_DIR / "sc_enigma.csv", delimiter = ',')
    #%%
    net7 = net7.rename(columns={'ROI Label': 'ROI_label_7'})
    net17 = net17.rename(columns={'ROI Label': 'ROI_label_17'})
    unite_df = pd.merge(net17, net7, on = ['R', 'A', 'S'])
    # %%
    unite_df.sort_values('ROI_label_7')
    unite_df['ROI_index_17'] = unite_df['ROI_label_17'] - 1
    unite_df['ROI_index_7'] = unite_df['ROI_label_7'] - 1 
    unite_df = unite_df.sort_values('ROI_label_7')
    seven_to_teen_dict = pd.Series(unite_df.ROI_index_17.values,index=unite_df.ROI_index_7).to_dict() 

    n_rows, n_cols = sc.shape
    sc_17 = np.empty_like(sc)
    for i, row_label in enumerate(seven_to_teen_dict.keys()):
            new_i = seven_to_teen_dict[i]
            sc_17[new_i, :] = sc[i, :]
            new_j = seven_to_teen_dict[i]
            sc_17[:, new_j] = sc[:, i]
    return sc_17


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
    sc_wmh_weighted = sc_norm * wmh_df / 100.0
    return sc_wmh_weighted


def get_node_damage(subj):
    wmh_df = pd.read_csv(LQT_DIR / f"sub-{subj}" / "pct_sdc_matrix.csv")
    # Computes the percent damage sustained by each node
    wmh_damage = wmh_df.iloc[:200, 1:201].mean(axis=0).to_numpy()
    return wmh_damage

#%%

