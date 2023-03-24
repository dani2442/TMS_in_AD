#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Evaluate different preprocessing strategies on modeling -- Version 1
Last edit:  2023/03/03
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file to evaluate the impact of preprocessing on modeling
            - Release notes:
                * Initial release
To do:      - 
Comments:   

Sources:    
"""

#%% ~~ Imports and directories ~~ %%#
# Import needed packages
import numpy as np
import scipy.io as sio
import WholeBrain.Utils.filteredPowerSpectralDensity as filtPowSpectr
import WholeBrain.BOLDFilters as BOLDfilters
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

XCP_SUBDIR_LIST = [P24_DIR, P24_NI_DIR, P36_DIR, P36_NI_DIR, AROMA_DIR, AROMA_NI_DIR]

#%% ~~ Load data ~~ %%#
def get_layout_subjs():
    layout = BIDSLayout(P24_DIR, validate = False, config = ['bids', 'derivatives'])
    subjs = layout.get_subjects()
    return layout, subjs

def get_method_ts(xcp_subdir, subj):
    try:
        sub_ts = np.genfromtxt(xcp_subdir / f"sub-{subj}" / f"ses-{REL_SES}" / "func" / f"sub-{subj}_ses-{REL_SES}_task-rest_run-1_space-MNI152NLin2009cAsym_atlas-Schaefer217_timeseries.tsv", delimiter='\t')
    except:
        sub_ts = np.genfromtxt(xcp_subdir / f"sub-{subj}" / f"ses-{REL_SES}" / "func" / f"sub-{subj}_ses-{REL_SES}_task-rest_space-MNI152NLin2009cAsym_atlas-Schaefer217_timeseries.tsv", delimiter='\t')
    sub_ts = sub_ts[4:]
    sub_ts_t = sub_ts.T 
    return sub_ts_t


def get_all_ts(subjs):
    ts_dict = {}
    for subdir in XCP_SUBDIR_LIST:
        meth_name = str(subdir).split('/')[7]
        ts_dict[meth_name] = {}
        for subj in subjs:
            ts_dict[meth_name][subj] = get_method_ts(subdir, subj)
    return ts_dict

def get_sc():
    # sc = pd.read_csv(UTL_DIR / "sc_enigma.csv", header=None)
    # sc_numpy = sc.to_numpy()
    sc = sio.loadmat(UTL_DIR / "Schaefer2018_200Parcels_17Networks" / "sc.mat")
    sc_numpy = sc['mat_zero']
    return sc_numpy


def get_frequencies(all_fMRI, flp, fhi, TR):
    BOLDfilters.flp = flp
    BOLDfilters.fhi = fhi
    BOLDfilters.TR = TR
    # Data is already filtered
    timeseries_4freq = np.array([v for k, v in all_fMRI.items()])
    # Get the phase difference using WholeBrain implementation
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(
        timeseries_4freq, TR=3.0
    )  # should be baseline_group_ts .. or baseling_group[0].reshape((1,52,193))
    f_diff[np.where(f_diff == 0)] = np.mean(
        f_diff[np.where(f_diff != 0)]
    )  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
    w_s = 2 * np.pi * f_diff
    return w_s

