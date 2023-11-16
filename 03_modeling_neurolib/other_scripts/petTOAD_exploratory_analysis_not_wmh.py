#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 1.1
Last edit:  2023/05/30
Authors:    Leone, Riccardo (RL)
Notes:      - Homogeneous wmh-weighted model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Refactored
To do:      - 
Comments:   

Sources: 
"""
# %% Initial imports
import filteredPowerSpectralDensity as filtPowSpectr
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from petTOAD_setup import *


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

# Set the simulation directory for the group
EXPL_DIR = RES_DIR / "exploratory_not_wmh_weighted"
if not Path.exists(EXPL_DIR):
    Path.mkdir(EXPL_DIR)
# Set the directory where to save results
paths.HDF_DIR = str(EXPL_DIR)


#%%
if __name__ == "__main__":
    # Get the timeseries for the HC group
    group_HC, timeseries_HC = get_group_ts_for_freqs(HC, all_fMRI_clean)
    f_diff_HC = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_HC, TR)
    f_diff_HC[np.where(f_diff_HC == 0)] = np.mean(f_diff_HC[np.where(f_diff_HC != 0)])
    # Get the timeseries for the MCI group
    group_MCI, timeseries_MCI = get_group_ts_for_freqs(MCI, all_fMRI_clean)
    f_diff_MCI = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_MCI, TR)
    f_diff_MCI[np.where(f_diff_MCI == 0)] = np.mean(f_diff_MCI[np.where(f_diff_MCI != 0)])

    parameters = ParameterSpace(
            {
                "a": [(np.ones(90) * a) for a in np.linspace(-0.08, 0.02, 101)],
                "K_gl": [float(G) for G in np.linspace(0.5, 2.5, 21)]
            },
            kind="grid",
        )
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
    model.params["duration"] = 12.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    n_sim = 10
    for i in range(n_sim):
        print(f"Starting simulations n°: {i+1}/{n_sim} for HC...")
        model.params["w"] = 2 * np.pi * f_diff_HC
        filename = "homogeneous_model_not_WMH-weight_HC.hdf"    
        #Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)
    for i in range(n_sim):
        print(f"Starting simulations n°: {i+1}/{n_sim} for MCI...")
        model.params["w"] = 2 * np.pi * f_diff_MCI
        filename = "homogeneous_model_not_WMH-weight_MCI.hdf"    
        #Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)


# %%
