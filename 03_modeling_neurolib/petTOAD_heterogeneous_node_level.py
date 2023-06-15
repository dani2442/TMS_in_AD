#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Heterogeneous model at the node level   -- Version 1.0
Last edit:  2023/06/13
Authors:    Leone, Riccardo (RL)
Notes:      - Heterogeneous node-level disconnected model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Initial commit
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


def prepare_group_simulation(group_name):
    # Set the simulation directory for the group
    SIM_DIR_GROUP = SIM_DIR / group_name
    # Set the directory where to save results
    paths.HDF_DIR = str(SIM_DIR_GROUP)
    # Get the timeseries for the chosen group
    group, timeseries = get_group_ts_for_freqs(group_name, all_fMRI_clean)
    # Get the frequencies (narrow bandwidth)
    n_subjs, n_nodes, Tmax = timeseries.shape
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR)
    f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])
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
    model.params["w"] = 2 * np.pi * f_diff
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["K_gl"] = 2.5  # Set this to the best G previously found!!!!
    return n_subjs, n_nodes, model, group

def prepare_subject_simulation(subj, ws, bs, random=False):
    WMH = wmh_dict[subj]
    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "a": [(np.ones(90) * -0.02) * w * WMH + b for w in ws for b in bs],
        },
        kind="grid",
    )
    if not random:
        filename = f"{subj}_homogeneous_model.hdf"
    else:
        filename = f"{subj}_homogeneous_model_random.hdf"
    return parameters, filename

# Choose the group on which to perform analyses ("HC_noWMH", "HC_WMH", "MCI_noWMH", "MCI_WMH")
group_list = ["HC_WMH", "MCI_WMH"]


ws = np.linspace(-0.1, 0.1, 51)
bs = np.linspace(-0.05, 0.05, 11)

wmh_dict = get_wmh_load_homogeneous(subjs)
#%%
for group_name in group_list:
    print(f"Now processing group {group_name}.")
    n_subjs, n_nodes, model, group = prepare_group_simulation(group_name)
    n_sim = 1
    group_subjs = group.keys()
    for j, subj in enumerate(group_subjs):
        print(f"Starting simulations for subject: {subj}, ({j + 1}/{len(group_subjs)})")
        parameters, filename = prepare_subject_simulation(subj, ws, bs, random = False)
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
    for j, subj in enumerate(group_subjs):
            print(f"Starting simulations with shuffled weights for subject: {subj}, ({j + 1}/{len(group_subjs)})")

            parameters, filename = prepare_subject_simulation(subj, ws, bs, random = True)
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
