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
import sys
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
    # skip the first 120 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 12000::300]
    t = model.outputs.t[12000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    bold_list.append(ts_filt)

    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list).squeeze()
    result_dict["t"] = t

    search.saveToPypet(result_dict, traj)

def prepare_subject_simulation(subj, ws, bs, random):
    WMH = wmh_dict[subj]
    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "a": [(np.ones(90) * -0.02) + w * WMH + b for w in ws for b in bs],
        },
        kind="grid",
    )
    if not random:
        filename = f"{subj}_homogeneous_model.hdf"
    else:
        filename = f"{subj}_homogeneous_model_random.hdf"

    return parameters, filename


ws_min = -0.1
ws_max = 0.1
bs_min = -0.05
bs_max = 0.05
ws_n = 21
bs_n = 11

ws = np.linspace(ws_min, ws_max, ws_n) 
bs = np.linspace(bs_min, bs_max, bs_n) 

condition = sys.argv[1].lower()

# Mapping dictionary
mapping = {
    "true": True,
    "false": False
}

random_condition = mapping.get(condition)

if not random_condition:
    wmh_dict = get_wmh_load_homogeneous(subjs)
    MOD_DIR = SIM_DIR / f"a-weight_ws_{ws_min}-{ws_max}_bs_{bs_min}-{bs_max}_prova"
else:
    MOD_DIR = SIM_DIR / f"a-weight_ws_{ws_min}-{ws_max}_bs_{bs_min}-{bs_max}_random"
    wmh_dict_pre = get_wmh_load_homogeneous(subjs)
    # random
    wmh_rand = np.array([w for w in wmh_dict_pre.values()])
    np.random.seed(1991)
    np.random.shuffle(wmh_rand)
    wmh_dict = {k:wmh_rand[n] for n, k in enumerate(wmh_dict_pre.keys())}

if not Path.exists(MOD_DIR):
    Path.mkdir(MOD_DIR)
# Set the directory where to save results
paths.HDF_DIR = str(MOD_DIR)


#%%
if __name__ == "__main__":
    # Get the timeseries for the CN group
    group_CN_WMH, timeseries_CN_WMH = get_group_ts_for_freqs(CN_WMH, all_fMRI_clean)
    f_diff_CN_WMH = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_CN_WMH, TR)
    f_diff_CN_WMH[np.where(f_diff_CN_WMH == 0)] = np.mean(f_diff_CN_WMH[np.where(f_diff_CN_WMH != 0)])
    # Get the timeseries for the MCI group
    group_MCI_WMH, timeseries_MCI_WMH = get_group_ts_for_freqs(MCI_WMH, all_fMRI_clean)
    f_diff_MCI_WMH = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_MCI_WMH, TR)
    f_diff_MCI_WMH[np.where(f_diff_MCI_WMH == 0)] = np.mean(f_diff_MCI_WMH[np.where(f_diff_MCI_WMH != 0)])


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
    # model.params["Dmat"] = None if not delay else Dmat
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["K_gl"] = 1.9  # Set this to the best G previously found!!!!

    n_sim = 2
    for j, subj in enumerate(subjs_to_sim[:2]):
        print(f"Starting simulations for subject: {subj}, ({j + 1}/{len(subjs_to_sim)})")
        parameters, filename = prepare_subject_simulation(subj, ws, bs, random=random_condition)
        if subj in CN_WMH:
            f_diff = f_diff_CN_WMH
        elif subj in MCI_WMH:
            f_diff = f_diff_MCI_WMH
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
