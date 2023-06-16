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
EXPL_DIR = RES_DIR / "exploratory"
if not Path.exists(EXPL_DIR):
    Path.mkdir(EXPL_DIR)
# Set the directory where to save results
paths.HDF_DIR = str(EXPL_DIR)

# %% Define functions
def prepare_subject_simulation(subj, ws, bs):
    WMH = wmh_dict[subj]
    timeseries = all_fMRI_clean[subj]
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR)
    f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])
    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "a": [(np.ones(90) * -0.02) * w * WMH + b for w in ws for b in bs],
        },
        kind="grid",
    )
    filename = f"{subj}_homogeneous_model.hdf"

    return f_diff, parameters, filename

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


short_subjs = HC_WMH[:30]
short_subjs = np.append(short_subjs, HC_no_WMH[:30])
short_subjs = np.append(short_subjs, MCI_no_WMH[:30])
short_subjs = np.append(short_subjs, MCI_WMH[:30])

ws = np.linspace(-0.5, 0.5, 31)
bs = np.linspace(0, 0.02, 5)

wmh_dict = get_wmh_load_homogeneous(subjs)

#%%
if __name__ == "__main__":
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
    model.params["K_gl"] = 1.9  # Set this to the best G previously found!!!!
    n_sim = 2
    for j, subj in enumerate(short_subjs):
        print(f"Starting simulations for subject: {subj}, ({j + 1}/{len(short_subjs)})")
        f_diff, parameters, filename = prepare_subject_simulation(subj, ws, bs)
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
