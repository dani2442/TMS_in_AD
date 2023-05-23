#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   PetTOAD setup for neurolib modeling   -- Version 2.0
Last edit:  2023/05/09
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file for neurolib
            - Release notes:
                * Fixed for AAL atlas
To do:      - 
Comments:   

Sources: filtPowSpectr abd BOLDFilters taken from Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""
# %%
# Imports
import numpy as np
import BOLDFilters as BOLDFilters
from scipy import stats
from petTOAD_load import *

# Get subjs names
_, subjs = get_layout_subjs()

# Set the frequencies for the bandwidth filter for the ts
BOLDFilters.flp = 0.04
BOLDFilters.fhi = 0.07
TR = 3.0

# Load SCs, timelines and group classifications
sc = load_norm_aal_sc()
# Prevent full synchronization of the model
# Create dict to store the "raw" unfiltered ts
all_fMRI_raw = {subj: load_ts_aal(subj) for subj in subjs}
# Check that the timeseries do not have regions with all zeros and return only OK subjs
all_fMRI_raw = check_ts(all_fMRI_raw)
# Demean, detrend and filter the signal only for subjects with complete data
all_fMRI_clean = {
    subj: stats.zscore(BOLDFilters.BandPassFilter(sub_ts), axis=1)
    for subj, sub_ts in all_fMRI_raw.items()
}
# New subject list overwrites the old one
subjs = [k for k in all_fMRI_clean.keys()]
# Get subject list for condition
HC_no_WMH, HC_WMH, MCI_no_WMH, MCI_WMH = get_classification(subjs)
# Group HC and MCI
HC = np.array([j for i in [HC_WMH, HC_no_WMH] for j in i]).astype("object")
MCI = np.array([j for i in [MCI_WMH, MCI_no_WMH] for j in i]).astype("object")

print("petTOAD Setup done!")

# %%
