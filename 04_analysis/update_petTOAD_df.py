#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Modify ADNIMERGE.csv to update the WMH volumes -- Version 1.0
Last edit:  2023/03/20
Authors:    Leone, Riccardo (RL)
Notes:      - Initial release 
            - Release notes:
                * None
To do:      - None
Comments:   

Sources:  
"""
# %% Imports
from extract_WMH_volumes import *

# Load csv files
sel_pts = pd.read_csv(UTL_DIR / "ADNI_selected_pts.csv")
wmh_check = pd.read_csv(UTL_DIR / "WMH_checklist.csv")
# Simplify csv
sel_pts = sel_pts.drop(
    columns=[
        "Image Data ID",
        "Modality",
        "Description",
        "Type",
        "Acq Date",
        "Format",
        "Downloaded",
    ]
)
sel_pts = sel_pts.drop_duplicates()
sel_pts = sel_pts.rename(columns={"Subject": "PTID"})
sel_pts["PTID"] = sel_pts["PTID"].str.replace("_", "")
sel_pts["PTID"] = "ADNI" + sel_pts["PTID"]
# Merge subtypes of MCI into just MCI
sel_pts["Group_bin"] = np.where(sel_pts["Group"] == "CN", "CN", "MCI")
# Merge with WMH lesion load
sel_pts_new = pd.merge(sel_pts, global_WMH_df, on="PTID")

thresh_subj = np.quantile(
    sel_pts_new[sel_pts_new["Group"] == "CN"]["WMH_load_subj_space"], 0.25
)
sel_pts_new["Group_bin_subj"] = np.where(
    sel_pts_new["WMH_load_subj_space"] < thresh_subj,
    sel_pts_new["Group_bin"] + "_no_WMH",
    sel_pts_new["Group_bin"] + "_WMH",
)

thresh_mni = np.quantile(
    sel_pts_new[sel_pts_new["Group"] == "CN"]["WMH_load_mni_space"], 0.25
)
sel_pts_new["Group_bin_mni"] = np.where(
    sel_pts_new["WMH_load_subj_space"] < thresh_subj,
    sel_pts_new["Group_bin"] + "_no_WMH",
    sel_pts_new["Group_bin"] + "_WMH",
)

wmh_check = wmh_check.rename(columns={"SUB_ID": "PTID"})
wmh_check["PTID"] = wmh_check["PTID"].str.replace("sub-", "")
wmh_check = wmh_check.drop(
    columns=["quality_mni_space_dns_mask", "quality_subj_space_dns_mask", "Comments"]
)
petTOAD_df = pd.merge(sel_pts_new, wmh_check, on="PTID")
petTOAD_df["Group_bin_Fazekas"] = np.where(
    petTOAD_df["Fazekas_periventricular"] + petTOAD_df["Fazekas_lobar"] < 2,
    petTOAD_df["Group_bin"] + "_no_WMH",
    petTOAD_df["Group_bin"] + "_WMH",
)
#%%
petTOAD_df.to_csv(UTL_DIR / "petTOAD_dataframe.csv")

