#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Nipype support -- Version 1
Last edit:  2023/03/03
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file for petTOAD Hopf simulation
            - Release notes:
                * Initial release
To do:      - 
Comments:   

Sources:    
"""

#%% ~~ Imports and directories ~~ %%#
# Import needed packages
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

#%% ~~ Load data ~~ %%#
layout = BIDSLayout(P24_DIR, validate = False, config = ['bids', 'derivatives'])
subjs = layout.get_subjects()

def filter_adnimerge(subjs):
    adnimerge = pd.read_csv(UTL_DIR / 'ADNIMERGE.csv')
    adnimerge['PTID'] = adnimerge['PTID'].str.replace('_','')
    adnimerge['PTID'] = "ADNI" + adnimerge['PTID']
    adnimerge_filt = adnimerge[adnimerge["PTID"].isin(subjs)]
    return adnimerge_filt

def get_classification(adnimerge_filt):
    CN = []
    MCI = []
    AD = []