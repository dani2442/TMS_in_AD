#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 2.1
Last edit:  2023/06/15
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

# Set the simulation directory for the group
MOD_DIR = SIM_DIR / "sc_disconn"
if not Path.exists(MOD_DIR):
    Path.mkdir(MOD_DIR)
# Set the directory where to save results
paths.HDF_DIR = str(MOD_DIR)

# %% Define functions
def prepare_subject_simulation(subj):
    sc = model.params.Cmat
    disconn_sc = get_sc_wmh_weighted(subj)
    disconn_sc = disconn_sc.to_numpy()
    disconn_cmat = np.multiply(sc, disconn_sc)
    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "Cmat": [disconn_cmat],
        },
        kind="grid",
    )
    filename = f"{subj}_sc_disconn_model.hdf"

    return parameters, filename

# Define the evaluation function
def evaluate(traj):
    model = search.getModelFromTraj(traj)
    bold_list = []

    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 120 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 12000::300]
    t = model.outputs.t[12000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    bold_list.append(ts_filt)

    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list).squeeze()
    result_dict["t"] = t

    search.saveToPypet(result_dict, traj)


#%%
if __name__ == "__main__":

    # Get the timeseries for the CN group
    group_CN, timeseries_CN = get_group_ts_for_freqs(CN, all_fMRI_clean)
    f_diff_CN = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_CN, TR)
    f_diff_CN[np.where(f_diff_CN == 0)] = np.mean(f_diff_CN[np.where(f_diff_CN != 0)])
    # Get the timeseries for the MCI group
    group_MCI, timeseries_MCI = get_group_ts_for_freqs(MCI, all_fMRI_clean)
    f_diff_MCI = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_MCI, TR)
    f_diff_MCI[np.where(f_diff_MCI == 0)] = np.mean(f_diff_MCI[np.where(f_diff_MCI != 0)])


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
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["K_gl"] = 1.9  # Set this to the best G previously found!!!!
    model.params["a"] = np.ones(90) * -0.02
    n_sim = 2

    for j, subj in enumerate(subjs_to_sim[:2]):
        print(f"Starting simulations for subject: {subj}, ({j + 1}/{len(subjs_to_sim)})")
        parameters, filename = prepare_subject_simulation(subj)
        if subj in CN:
            f_diff = f_diff_CN
        elif subj in MCI:
            f_diff = f_diff_MCI
        model.params["w"] = 2 * np.pi * f_diff
        for i in range(n_sim):
            print(f"Starting simulations n°: {i+1}/{n_sim}")
            #Initialize the search
            search = BoxSearch(
                model=model,
                evalFunction=evaluate,
                parameterSpace=parameters,
                filename=filename,
            )
            search.run(chunkwise=True, chunksize=60000, append=True)



# %%
