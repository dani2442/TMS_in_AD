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
            "K_gl": [float(round(1.9 + w * WMH + b, 3)) for w in ws for b in bs],
        },
        kind="grid",
    )
    if not random:
        filename = f"{subj}_homogeneous_G-weight_model.hdf"
    else:
        filename = f"{subj}_homogeneous_G-weight_model_random.hdf"

    return parameters, filename


ws_min = -1.
ws_max = 1.
bs_min = -0.5
bs_max = 0.5
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
    print("Starting with non random simulations!")
    wmh_dict = get_wmh_load_homogeneous(subjs)
    MOD_DIR = (
        SIM_DIR / f"G-weight_ws_{ws_min}-{ws_max}_bs_{bs_min}-{bs_max}"
    )
else:
    print("Starting with random simulations!")
    # random
    wmh_dict_pre = get_wmh_load_homogeneous(subjs)
    wmh_rand = np.array([w for w in wmh_dict_pre.values()])
    np.random.seed(1991)
    np.random.shuffle(wmh_rand)
    wmh_dict = {k:wmh_rand[n] for n, k in enumerate(wmh_dict_pre.keys())}

    # random
    # wmh_dict_log = np.array([np.log(w) if w != 0 else 0 for w in wmh_dict_pre.values()])
    # wmh_dict_log_z = (wmh_dict_log - wmh_dict_log.min()) / (
    #     wmh_dict_log.max() - wmh_dict_log.min()
    # )
    # np.random.seed(1991)
    # np.random.shuffle(wmh_dict_log_z)
    # wmh_dict = {k: wmh_dict_log_z[n] for n, k in enumerate(wmh_dict_pre.keys())}
    MOD_DIR = (
        SIM_DIR
        / f"G-weight_ws_{ws_min}-{ws_max}_bs_{bs_min}-{bs_max}_random"
    )

if not Path.exists(MOD_DIR):
    Path.mkdir(MOD_DIR)
# Set the directory where to save results
paths.HDF_DIR = str(MOD_DIR)

# %%
if __name__ == "__main__":
    # Get the timeseries for the CN group
    group_CN, timeseries_CN = get_group_ts_for_freqs(CN_WMH, all_fMRI_clean)
    f_diff_CN = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_CN, TR)
    f_diff_CN[np.where(f_diff_CN == 0)] = np.mean(f_diff_CN[np.where(f_diff_CN != 0)])
    # Get the timeseries for the MCI group
    group_MCI, timeseries_MCI = get_group_ts_for_freqs(MCI_WMH, all_fMRI_clean)
    f_diff_MCI = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_MCI, TR)
    f_diff_MCI[np.where(f_diff_MCI == 0)] = np.mean(
        f_diff_MCI[np.where(f_diff_MCI != 0)]
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
    # model.params["Dmat"] = None if not delay else Dmat
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["a"] = np.ones(90) * -0.02

    n_sim = 1
    for j, subj in enumerate(subjs_to_sim):
        print(f"Starting simulations for subject: {subj}, ({j + 1}/{len(subjs_to_sim)})")
        parameters, filename = prepare_subject_simulation(subj, ws, bs, random=random_condition)
        if subj in CN:
            f_diff = f_diff_CN
        elif subj in MCI:
            f_diff = f_diff_MCI
        model.params["w"] = 2 * np.pi * f_diff
        for i in range(n_sim):
            print(f"Starting simulations n°: {i+1}/{n_sim}")
            # Initialize the search
            search = BoxSearch(
                model=model,
                evalFunction=evaluate,
                parameterSpace=parameters,
                filename=filename,
            )
            search.run(chunkwise=True, chunksize=60000, append=True)


# %%
