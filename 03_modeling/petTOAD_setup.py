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

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""
#%%
# Import the needed packages
# ==========================================================================
# ==========================================================================
import numpy as np
from petTOAD_load import *

# --------------------------------------------------------------------------
# Begin setup...
# --------------------------------------------------------------------------

# Subject names and timeseries dictionary
_, subjs = get_layout_subjs()

import WholeBrain.Models.supHopf as Hopf
from WholeBrain.simulate_SimOnly import Tmaxneuronal

Hopf.initialValue = 0.1
neuronalModel = Hopf

import WholeBrain.Integrator_EulerMaruyama as myIntegrator

# import functions.Integrator_Euler as myIntegrator
integrator = myIntegrator
integrator.neuronalModel = neuronalModel
integrator.clamping = False
integrator.verbose = False
# Integration parms...
dt = 5e-5
tmax = 193
ds = 1e-4
Tmaxneuronal = int((tmax+dt))

import WholeBrain.simulate_SimOnly as simulateBOLD

simulateBOLD.warmUp = True
simulateBOLD.warmUpFactor = 606.0 / 2000.0
simulateBOLD.integrator = integrator

import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.phFCD as phFCD
import WholeBrain.Optimizers.ParmSeep as ParmSeep

ParmSeep.simulateBOLD = simulateBOLD
ParmSeep.integrator = integrator
ParmSeep.verbose = True

import WholeBrain.Utils.filteredPowerSpectralDensity as filtPowSpectr
import WholeBrain.BOLDFilters as BOLDFilters

# NARROW LOW BANDPASS just for filtering the intrinsic frequencies
BOLDFilters.flp = 0.01  # lowpass frequency of filter
BOLDFilters.fhi = 0.08 # highpass
BOLDFilters.TR = 3.0
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------

# Check that the timeseries doesn't have zero rows
def checking_timeseries(all_fMRI):
    import json

    zeros_ts = []
    for subj, ts in all_fMRI.items():
        zeros_row = np.where(np.all(np.isclose(ts, 0), axis=1))[0]
        if zeros_row.size > 0:
            print(f"Subj-{subj} has some ROI with only 0s...")
            zeros_ts.append(subj)
    dropping_dict = {
        "The following patients were discarded due to some ROIs with only zero values": zeros_ts
    }
    with open(RES_DIR / "dropped_patients_with_ROIS_with_zeros.json", "w") as f:
        json.dump(dropping_dict, f)
    all_fMRI_cleaned = {
        subj: ts for subj, ts in all_fMRI.items() if subj not in zeros_ts
    }

    return all_fMRI_cleaned

# Load SCs, timelines and group classifications
sc = get_sc_enigma()
# Prevent full synchronization of the model
sc_norm = sc * 0.2 / sc.max()


all_fMRI = {}
for subj in subjs:
    all_fMRI[subj] = get_method_ts(subj)
all_fMRI = checking_timeseries(all_fMRI)
# New subject list overwrites the old one
subjs = [k for k in all_fMRI.keys()]
HC_no_WMH, HC_WMH, MCI_no_WMH, MCI_WMH = get_classification(subjs)
MCI = np.array([j for i in [MCI_WMH, MCI_no_WMH] for j in i]).astype('object')
all_HC_fMRI = {k: v for k, v in all_fMRI.items() if k in HC_no_WMH}
baseline_group_ts = np.array([ts for id, ts in all_HC_fMRI.items() if id in HC_no_WMH])
#%%
nNodes, Tmax = list(all_HC_fMRI.values())[0].shape

# ------------------------------------------------
# Hopf.beta = 0.01
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(
    baseline_group_ts, TR=3.0
)  # baseline_group[0].reshape((1,52,193))
f_diff[np.where(f_diff == 0)] = np.mean(
    f_diff[np.where(f_diff != 0)]
)  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
# Hopf.omega = repmat(2*pi*f_diff',1,2);     # f_diff is the frequency power
Hopf.omega = 2 * np.pi * f_diff

#%%
# Configure and compute Simulation
# ------------------------------------------------
# distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}
# selectedObservable = 'phFCD'
distanceSettings = {"FC": (FC, True), "phFCD": (phFCD, True)}

warmUp = True
warmUpFactor = 10.
simulateBOLD.TR = 3.0  # Recording interval: 1 sample every 3 seconds
simulateBOLD.dt = 0.1 * simulateBOLD.TR / 2.0
simulateBOLD.Tmax = Tmax  # This is the length, in seconds
simulateBOLD.dtt = simulateBOLD.TR  # We are not using milliseconds
simulateBOLD.t_min = 10 * simulateBOLD.TR
# simulateBOLD.recomputeTmaxneuronal() <- do not update Tmaxneuronal this way!
# simulateBOLD.warmUpFactor = 6.
simulateBOLD.Tmaxneuronal = (Tmax - 1) * simulateBOLD.TR + 30
integrator.ds = simulateBOLD.TR  # record every TR millisecond

base_a_value = -0.02
Hopf.setParms({"a": base_a_value})
Hopf.setParms({"SC": sc_norm})

# Set the filters for the simulations
print("ADHopf Setup done!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF

# %%
