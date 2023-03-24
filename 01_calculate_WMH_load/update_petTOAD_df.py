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

#%%
# Load csv files
sel_pts = pd.read_csv(UTL_DIR / "ADNI_selected_pts.csv")
wmh_check = pd.read_csv(UTL_DIR / "WMH_checklist.csv")
adnimerge = pd.read_csv(UTL_DIR / "ADNIMERGE.csv")
# wmh_check contains the final list of subjects (those for whom we have Fazekas score):
wmh_check = wmh_check.rename(columns={"SUB_ID": "PTID"})
wmh_check["PTID"] = wmh_check["PTID"].str.replace("sub-", "")
wmh_check = wmh_check.drop(
    columns=["quality_mni_space_dns_mask", "quality_subj_space_dns_mask", "Comments"]
)
wmh_check_final = wmh_check.dropna(subset = 'Fazekas_lobar')

# Simplify selected patients.csv
sel_pts = sel_pts.drop(
    columns=[
        "Image Data ID",
        "Modality",
        "Description",
        "Type",
        "Format",
        "Downloaded",
    ]
)
sel_pts = sel_pts.drop_duplicates()
sel_pts = sel_pts.rename(columns = {'Acq Date': 'EXAMDATE', 'Subject': 'PTID', 'Visit': 'VISCODE'})
sel_pts["PTID"] = sel_pts["PTID"].str.replace("_", "")
sel_pts["PTID"] = "ADNI" + sel_pts["PTID"]
#Merge the selected patients df and the final list of subjects
sel_pts_dummy = pd.merge(sel_pts, wmh_check_final, on = 'PTID')
# Merge subtypes of MCI into just MCI
sel_pts_dummy["Group_bin"] = np.where(sel_pts_dummy["Group"] == "CN", "CN", "MCI")
# Merge with WMH lesion load df
sel_pts_new = pd.merge(sel_pts_dummy, global_WMH_df, on="PTID")

#Calculate the threshold to say that a patient is WMH/no WMH in subj space
thresh_subj = np.quantile(
    sel_pts_new[sel_pts_new["Group"] == "CN"]["WMH_load_subj_space"], 0.25
)
# Apply the threshold to binarize CN and MCI groups
sel_pts_new["Group_bin_subj"] = np.where(
    sel_pts_new["WMH_load_subj_space"] < thresh_subj,
    sel_pts_new["Group_bin"] + "_no_WMH",
    sel_pts_new["Group_bin"] + "_WMH",
)
# Do the same for the threshold in MNI space
thresh_mni = np.quantile(
    sel_pts_new[sel_pts_new["Group"] == "CN"]["WMH_load_mni_space"], 0.25
)
sel_pts_new["Group_bin_mni"] = np.where(
    sel_pts_new["WMH_load_subj_space"] < thresh_subj,
    sel_pts_new["Group_bin"] + "_no_WMH",
    sel_pts_new["Group_bin"] + "_WMH",
)
# Apply a threshold based on Fazekas score and not WMH volume
sel_pts_new["Group_bin_Fazekas"] = np.where(
    sel_pts_new["Fazekas_periventricular"] + sel_pts_new["Fazekas_lobar"] < 2,
    sel_pts_new["Group_bin"] + "_no_WMH",
    sel_pts_new["Group_bin"] + "_WMH",
)

# Make a shorter version for analyses
petTOAD_df = sel_pts_new.drop(
    columns=[
        "VISCODE",
        "EXAMDATE",
    ]
)
# Save the shorter version
petTOAD_df.to_csv(RES_DIR / "petTOAD_dataframe.csv")

# Make a merged version of ADNIMERGE with WMH data
sel_pts_new["VISCODE"] = sel_pts_new["VISCODE"].str.replace("v", "m")
sel_pts_new["VISCODE"] = sel_pts_new["VISCODE"].str.replace("sc", "bl")
sel_pts_new["VISCODE"] = sel_pts_new["VISCODE"].str.replace("init", "bl")
sel_pts_new["VISCODE"] = sel_pts_new["VISCODE"].str.replace("y1", "m12")
sel_pts_new["VISCODE"] = sel_pts_new["VISCODE"].str.replace("y2", "m24")

adnimerge["PTID"] = adnimerge["PTID"].str.replace("_", "")
adnimerge["PTID"] = "ADNI" + adnimerge["PTID"]
adni_new = pd.merge(adnimerge, sel_pts_new, on = ['PTID', 'VISCODE'])
adni_new.to_csv(RES_DIR / "ADNIMERGE_petTOAD.csv")
# Not all patients that we processed have info on the ADNIMERGE dataset!
# Nonetheless, the group to which they belong is reported on the ADNI website
pt_adni = adni_new['PTID'].to_numpy()
pt_sel = sel_pts_new['PTID'].to_numpy()
a = [f for f in pt_sel if f not in pt_adni]
help_df = sel_pts_new[sel_pts_new['PTID'].isin(a)][['PTID', 'EXAMDATE']]
# Save the list of patients with no ADNIMERGE data
help_df.to_csv(RES_DIR / 'list_of_patients_with_no_ADNIMERGE_data.csv')