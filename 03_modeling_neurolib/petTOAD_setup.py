#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   PetTOAD setup for neurolib modeling   -- Version 2.1
Last edit:  2023/05/26
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file for neurolib
            - Release notes:
                * 
To do:      - 
Comments:   

Sources: filtPowSpectr abd BOLDFilters taken from Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""
# %%
# Imports
import numpy as np
from scipy import stats
from scipy.signal import detrend

import BOLDFilters as BOLDFilters
from petTOAD_load import *

# Get subjs names
_, subjs = get_layout_subjs()

# Set the frequencies for the bandwidth filter for the ts
BOLDFilters.flp = 0.04
BOLDFilters.fhi = 0.07
TR = 3.0

# Load SCs, timelines and group classifications
sc = load_norm_aal_sc()
n_nodes = sc.shape[0]
# Create dict to store the "raw" unfiltered ts
all_fMRI_raw = {subj: load_ts_aal(subj) for subj in subjs}
# Check that the timeseries do not have regions with all zeros and return only OK subjs
all_fMRI_raw = check_ts(all_fMRI_raw)
# Filter, detrend, z-score the signal only for subjects with complete data
all_fMRI_clean = {
    subj: stats.zscore(detrend(BOLDFilters.BandPassFilter(sub_ts)), axis=1)
    for subj, sub_ts in all_fMRI_raw.items()
}
# New subject list overwrites the old one
subjs = [k for k in all_fMRI_clean.keys()]
# Get subject list for each condition
CN, MCI, CN_no_WMH, CN_WMH, MCI_no_WMH, MCI_WMH = get_classification(subjs)

# We only want to simulate with WMH-weighted models those patients that do have wmh
subjs_to_sim = define_subjs_to_sim()


print("petTOAD Setup done!")
