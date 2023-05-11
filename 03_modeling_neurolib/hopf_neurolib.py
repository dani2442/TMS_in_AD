#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 2.0
Last edit:  2023/05/11
Authors:    Leone, Riccardo (RL)
Notes:      - Model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Introducing phenomenological Hopf model from personal Github repo
To do:      - 
Comments:   

Sources: 
"""
# %% Initial imports
import matplotlib.pyplot as plt
import neurolib.utils.functions as func
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from petTOAD_setup import *

# Set the directory where to save results
paths.HDF_DIR = str(RES_DIR / "neurolib")

# Choose the group on which to perform analyses
group = all_HC_fMRI_clean
group_name = "HC"

# Set if the model has delay
delay = False
if not delay:
    Dmat_dummy = np.zeros_like(sc)
    Dmat = Dmat_dummy
else:
    pass


# %% Define functions
# Define the evaluation function
def evaluate(traj):
    model = search.getModelFromTraj(traj)
    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 180 secs
    ts = model.outputs.x[:, 18000::300]
    t = model.outputs.t[18000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    sFC = func.fc(ts_filt)
    result_dict = {}
    result_dict["BOLD"] = ts_filt
    result_dict["FC"] = sFC
    result_dict["t"] = t
    result_dict["fc_corr"] = func.matrix_correlation(sFC, avg_fc)

    search.saveToPypet(result_dict, traj)


def calc_and_save_fc(ts_dict):
    # Calculate the empirical FC matrix out of bandpass-filtered timeseries
    fcs = []
    for ts in ts_dict.values():
        fcs.append(func.fc(ts))
    avg_fc = np.array(fcs).mean(axis=0)
    np.savetxt(RES_DIR / "average_FC_HC.csv", avg_fc, delimiter=",")
    return avg_fc


# %% Initialize the model
model = PhenoHopfModel(Cmat=sc, Dmat=Dmat_dummy)
model.params["Dmat"] = None
# Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
model.params["duration"] = 12.65 * 60 * 1000
model.params["signalV"] = 0
model.params["w"] = 2 * np.pi * f_diff
model.params["dt"] = 0.1
model.params["sampling_dt"] = 10.0
model.params["sigma"] = 0.02
model.params["a"] = np.ones(90) * (-0.02)

parameters = ParameterSpace({"K_gl": np.round(np.linspace(1, 4, 2), 3)}, kind="grid")

search = BoxSearch(
    model=model,
    evalFunction=evaluate,
    parameterSpace=parameters,
    filename="initial_exploration_Gs.hdf",
)

search.run(chunkwise=True, chunksize=60000, append=True)
search.loadResults()
