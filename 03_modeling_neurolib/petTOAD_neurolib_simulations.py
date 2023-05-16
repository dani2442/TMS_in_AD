#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 2.2
Last edit:  2023/05/11
Authors:    Leone, Riccardo (RL)
Notes:      - Model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Changed script to work with following steps
To do:      - 
Comments:   

Sources: 
"""
# %% Initial imports
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from petTOAD_setup import *

# Choose the group on which to perform analyses
group = all_HC_fMRI_clean
group_name = "HC_noWMH"

SIM_DIR = RES_DIR / 'model_simulations'
if not Path.exists(SIM_DIR):
    Path.mkdir(SIM_DIR)

# Create the results dir for the group
SIM_DIR_GROUP = SIM_DIR / group_name
if not Path.exists(SIM_DIR_GROUP):
    Path.mkdir(SIM_DIR_GROUP)

# Set the directory where to save results
paths.HDF_DIR = str(SIM_DIR_GROUP)

# %% Define functions
# Define the evaluation function
def evaluate(traj):
    model = search.getModelFromTraj(traj)
    bold_list = []

    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 180 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 18000::300]
    t = model.outputs.t[18000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    bold_list.append(ts_filt)

    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list).squeeze()
    result_dict["t"] = t

    search.saveToPypet(result_dict, traj)


# Set if the model has delay
delay = False
if not delay:
    Dmat_dummy = np.zeros_like(sc)
    Dmat = Dmat_dummy
else:
    pass

# Initialize the model (neurolib wants a Dmat to initialize the mode,
# so we gave it an empty Dmat, which we also later cancel by setting it to None)
model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
model.params["Dmat"] = None if not delay else Dmat
# Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
model.params["duration"] = 0.65 * 60 * 1000
model.params["signalV"] = 0
model.params["w"] = 2 * np.pi * f_diff
model.params["dt"] = 0.1
model.params["sampling_dt"] = 10.0
model.params["sigma"] = 0.02
model.params["a"] = np.ones(90) * (-0.02)

# Define the parametere space to explore
parameters = ParameterSpace(
    {"K_gl": np.round(np.linspace(0.0, 6.0, 2), 3)}, kind="grid"
)

filename = "exploration_Gs.hdf" 

if __name__ == '__main__':
    for _ in range(50):
        # Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)

