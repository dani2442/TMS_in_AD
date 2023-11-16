#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Modify df_adnimerge.csv to update the WMH volumes -- Version 1.1
Last edit:  2023/08/05
Authors:    Leone, Riccardo (RL)
Notes:      - Initial release 
            - Release notes:
                * Change names to new standard df_name
To do:      - None
Comments:   

Sources:  
"""
# %% Imports
from extract_WMH_volumes import *

# Load csv files
df_sel_pts = pd.read_csv(UTL_DIR / "ADNI_selected_pts.csv")
df_wmh_check = pd.read_csv(UTL_DIR / "df_wmh_checklist.csv")
df_adnimerge = pd.read_csv(UTL_DIR / "df_adnimerge.csv")
# df_wmh_check contains the final list of subjects (those for whom we have Fazekas score):
df_wmh_check = df_wmh_check.rename(columns={"SUB_ID": "PTID"})
df_wmh_check["PTID"] = df_wmh_check["PTID"].str.replace("sub-", "")
df_wmh_check = df_wmh_check.drop(
    columns=["quality_mni_space_dns_mask", "quality_subj_space_dns_mask", "Comments"]
)
df_wmh_check_final = df_wmh_check.dropna(subset="Fazekas_lobar")

# Simplify selected patients.csv
df_sel_pts = df_sel_pts.drop(
    columns=[
        "Image Data ID",
        "Modality",
        "Description",
        "Type",
        "Format",
        "Downloaded",
    ]
)
df_sel_pts = df_sel_pts.drop_duplicates()
df_sel_pts = df_sel_pts.rename(
    columns={"Acq Date": "EXAMDATE", "Subject": "PTID", "Visit": "VISCODE"}
)
df_sel_pts["PTID"] = df_sel_pts["PTID"].str.replace("_", "")
df_sel_pts["PTID"] = "ADNI" + df_sel_pts["PTID"]
# Merge the selected patients df and the final list of subjects
df_sel_pts_dummy = pd.merge(df_sel_pts, df_wmh_check_final, on="PTID")
# Merge subtypes of MCI into just MCI
df_sel_pts_dummy["Group_bin"] = np.where(df_sel_pts_dummy["Group"] == "CN", "CN", "MCI")
# Merge with df containing wmh lesion load
df_sel_pts_new = pd.merge(df_sel_pts_dummy, df_wmh, on="PTID")
# Apply a threshold based on Fazekas score <= 2
# We calculated separate Fazekas for periventricular (0-3 points) and lobar (0-3 points)
# for all subjects. Here we want to divide patients into a non-wmh group and a wmh group
# and we use the sum of the 2 scores --> If the sum is <= 2 (== overall Fazekas of 1), then
# patients are considered to be non-wmh. If any of the two score is higher or equal than 2,
# then the patient is assigned to the WMH group.
df_sel_pts_new["Group_bin_Fazekas"] = np.where(
    (df_sel_pts_new["Fazekas_periventricular"] >= 2) | (df_sel_pts_new["Fazekas_lobar"] >=2),
    df_sel_pts_new["Group_bin"] + "_WMH",
    df_sel_pts_new["Group_bin"] + "_no_WMH",
)
# Make a shorter version for analyses
df_petTOAD = df_sel_pts_new.drop(
    columns=[
        "VISCODE",
        "EXAMDATE",
    ]
)
# Save the shorter version
df_petTOAD.to_csv(RES_DIR / "df_petTOAD.csv")


# # Make a merged version of df_adnimerge with WMH data
# df_sel_pts_new["VISCODE"] = df_sel_pts_new["VISCODE"].str.replace("v", "m")
# df_sel_pts_new["VISCODE"] = df_sel_pts_new["VISCODE"].str.replace("sc", "bl")
# df_sel_pts_new["VISCODE"] = df_sel_pts_new["VISCODE"].str.replace("init", "bl")
# df_sel_pts_new["VISCODE"] = df_sel_pts_new["VISCODE"].str.replace("y1", "m12")
# df_sel_pts_new["VISCODE"] = df_sel_pts_new["VISCODE"].str.replace("y2", "m24")

# df_adnimerge["PTID"] = df_adnimerge["PTID"].str.replace("_", "")
# df_adnimerge["PTID"] = "ADNI" + df_adnimerge["PTID"]
# adni_new = pd.merge(df_adnimerge, df_sel_pts_new, on = ['PTID', 'VISCODE'])
# adni_new.to_csv(RES_DIR / "adnimerge_petTOAD.csv")
# # Not all patients that we processed have info on the df_adnimerge dataset!
# # Nonetheless, the group to which they belong is reported on the ADNI website
# pt_adni = adni_new['PTID'].to_numpy()
# pt_sel = df_sel_pts_new['PTID'].to_numpy()
# a = [f for f in pt_sel if f not in pt_adni]
# help_df = df_sel_pts_new[df_sel_pts_new['PTID'].isin(a)][['PTID', 'EXAMDATE']]
# # Save the list of patients with no df_adnimerge data
# help_df.to_csv(RES_DIR / 'list_of_patients_with_no_adnimerge_data.csv')
# %%
