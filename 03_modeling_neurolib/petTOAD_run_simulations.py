#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 1.1
Last edit:  2023/08/08
Authors:    Leone, Riccardo (RL)
Notes:      - Model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Updated to run on HPC
To do:      - Model with delay
Comments:   

Sources: 
"""
# %% Initial imports
import sys

import matplotlib.pyplot as plt
import neurolib.utils.functions as func
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from neurolib.utils import pypetUtils as pu
from neurolib.utils.parameterSpace import ParameterSpace

import filteredPowerSpectralDensity as filtPowSpectr
import my_functions as my_func
from petTOAD_parameter_setup import *
from petTOAD_setup import *


def get_f_diff_group(group):
    # Get the timeseries for the CN no WMH group
    group, timeseries = get_group_ts_for_freqs(group, all_fMRI_clean)
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR)
    f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])
    return f_diff


# Define the evaluation function for the subjectwise simulations
def evaluate_subj(traj):
    model = search.getModelFromTraj(traj)
    bold_list = []

    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 120 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 12000::300]
    t = model.outputs.t[12000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    bold_list.append(ts_filt)
    if not np.isnan([np.isnan(t).any() for t in bold_list]).any():
        print("There are no nans. Continue with the simulations")
    else:
        print("There are some nans, aborting...")
        return
    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list).squeeze()
    result_dict["t"] = t

    search.saveToPypet(result_dict, traj)


def calculate_results_from_bolds_subject(bold_arr, n_sim, n_parms, n_nodes):
    # Create a new array to store the FC and phFCD values with the same shape as bold array
    fc_array = np.zeros([n_sim, n_parms, n_nodes, n_nodes])
    phfcd_array = np.zeros([n_sim, n_parms, 18145])

    # Iterate over each element in the bold array
    for i in range(n_sim):
        for j in range(n_parms):
            print(
                f"Now calculating results from the {i} simulation for parameter {j}..."
            )
            # Get the current timeseries
            timeseries = bold_arr[i, j].squeeze()

            # Perform FC and phFCD analysis
            if np.isnan(timeseries).any():
                print("Simulation has some nans, aborting!")
                continue
            else:
                print("Simulation has no nans, good to go")
                print("Calculating FC..")
                fc_value = func.fc(timeseries)
                print("Calculating phFCD")
                phfcd_value = my_func.phFCD(timeseries)
                # Store the FC and phFCD value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
                phfcd_array[i, j] = phfcd_value
    return fc_array, phfcd_array


def gather_results_from_repeated_simulations(res_dir, filename, ws=None, bs=None):
    trajs = pu.getTrajectorynamesInFile(f"{res_dir}/{filename}")
    big_list = []
    for traj in trajs:
        traj_list = []
        tr = pu.loadPypetTrajectory(f"{res_dir}/{filename}", traj)
        run_names = tr.f_get_run_names()
        n_run = len(run_names)
        ns = range(n_run)
        for i in ns:
            r = pu.getRun(i, tr)
            traj_list.append(r["BOLD"])
        big_list.append(traj_list)
    bold_arr = np.array(big_list)
    n_sim = len(trajs)
    if res_dir == str(SIM_DIR_SC):
        n_parms = 1
    else:
        n_parms = len(ws) * len(bs)
    fc_array, phfcd_array = calculate_results_from_bolds_subject(
        bold_arr, n_sim, n_parms, n_nodes
    )
    timeseries = all_fMRI_clean[subj]
    fc = func.fc(timeseries)
    print("Calculating phFCD")
    phfcd = my_func.phFCD(timeseries)
    # Get the average fc across the n simulations
    sim_fc = fc_array.mean(axis=0)
    print("Calculating fcs correlations...")
    fc_pearson = [func.matrix_correlation(row_fc, fc) for row_fc in sim_fc]
    print("Calculating phFCDs...")
    phfcd_ks = []
    for row in phfcd_array:
        row_phfcd_ks = [
            my_func.matrix_kolmogorov(phfcd, sim_phfcd) for sim_phfcd in row
        ]
        phfcd_ks.append(row_phfcd_ks)
        phfcd_ks_arr = np.array(phfcd_ks)
    phfcd_ks = phfcd_ks_arr.mean(axis=0)
    return fc_pearson, phfcd_ks


def create_df_results(SIM_DIR, model, ws, bs, fc_pearson, phfcd_ks):
    data = [
        [[(round(b, 3), round(w, 3)) for w in ws for b in bs], fc_pearson, phfcd_ks]
    ]
    columns = ["b_w", "fc_pearson", "phfcd_ks"]
    res_df = pd.DataFrame(data, columns=columns).explode(columns)
    res_df["b"], res_df["w"] = zip(*res_df.b_w)
    res_df = res_df.drop(columns=["b_w"])
    res_df.to_csv(SIM_DIR / f"sub-{subj}_df_results_{model}.csv")


def create_df_results_sc_disconn(fc_pearson, phfcd_ks):
    data = [[fc_pearson, phfcd_ks]]
    columns = ["fc_pearson", "phfcd_ks"]
    res_df = pd.DataFrame(data, columns=columns).explode(columns)
    res_df.to_csv(SIM_DIR_SC / f"sub-{subj}_df_results_disconn.csv")


# Prepare and define the subject-wise simulations
def prepare_subject_simulation_homogeneous_a(subj, WMH, ws, bs, random_cond):
    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "a": [(np.ones(n_nodes) * -0.02) + w * WMH + b for w in ws for b in bs],
        },
        kind="grid",
    )
    if not random_cond:
        filename = f"{subj}_homogeneous_a-weight_model.hdf"

    else:
        filename = f"{subj}_homogeneous_a-weight_model_random.hdf"

    return parameters, filename


def simulate_homogeneous_model_a(subj, f_diff, best_G, wmh_dict, ws, bs, random_cond):
    global search
    print(
        f"Now performing the simulations for the homogeneous a-weighted model with random = {random_cond}..."
    )
    WMH = wmh_dict[subj]
    Dmat = np.zeros_like(sc)
    # Initialize the model (neurolib wants a Dmat to initialize the mode,
    # so we gave it an empty Dmat, which we also later cancel by setting it to None)
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    model.params["Dmat"] = None
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["w"] = 2 * np.pi * f_diff
    model.params["K_gl"] = best_G
    # Set the directory where to save results
    savedir = str(SIM_DIR_A)
    paths.HDF_DIR = savedir
    parameters, filename = prepare_subject_simulation_homogeneous_a(
        subj, WMH, ws, bs, random_cond
    )
    for i in range(n_sim):
        print(f"Starting simulations n°: {i+1}/{n_sim}")
        # Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate_subj,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)
    fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(
        savedir, filename, ws=ws, bs=bs
    )
    create_df_results(
        SIM_DIR_A, "homogeneous_a-weight", ws, bs, fc_pearson, phfcd_ks
    )


###########################################################
############### HOMOGENEOUS G. MODEL ######################
###########################################################


def prepare_subject_simulation_homogeneous_G(subj, WMH, ws, bs, random_cond):
    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "K_gl": [float(round(best_G + w * WMH + b, 3)) for w in ws for b in bs],
        },
        kind="grid",
    )
    if not random_cond:
        filename = f"{subj}_homogeneous_G-weight_model.hdf"
    else:
        filename = f"{subj}_homogeneous_G-weight_model_random.hdf"

    return parameters, filename


def simulate_homogeneous_model_G(subj, f_diff, wmh_dict, ws, bs, random_cond):
    global search
    print(
        f"Now performing the simulations for the homogeneous G-weighted model with random = {random_cond}..."
    )
    WMH = wmh_dict[subj]
    Dmat = np.zeros_like(sc)
    # Initialize the model (neurolib wants a Dmat to initialize the mode,
    # so we gave it an empty Dmat, which we also later cancel by setting it to None)
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    model.params["Dmat"] = None
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["w"] = 2 * np.pi * f_diff
    model.params["a"] = np.ones(n_nodes) * -0.02
    # Set the directory where to save results
    savedir = str(SIM_DIR_G)
    paths.HDF_DIR = savedir
    parameters, filename = prepare_subject_simulation_homogeneous_G(
        subj, WMH, ws, bs, random_cond
    )
    for i in range(n_sim):
        print(f"Starting simulations n°: {i+1}/{n_sim}")
        # Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate_subj,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)
    fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(
        savedir, filename, ws=ws, bs=bs
    )
    create_df_results(
        SIM_DIR_G, "homogeneous_G-weight", ws, bs, fc_pearson, phfcd_ks
    )


###########################################################
############### DISCONNECTION MODEL #######################
###########################################################


def prepare_subject_simulation_disconn(model, subj):
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


def simulate_disconn_model(subj, best_G, f_diff, random_cond):
    global search
    print(f"Now performing the simulations for the structural disconnectivity model...")
    Dmat = np.zeros_like(sc)
    # Initialize the model (neurolib wants a Dmat to initialize the mode,
    # so we gave it an empty Dmat, which we also later cancel by setting it to None)
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    model.params["Dmat"] = None
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["a"] = np.ones(n_nodes) * -0.02
    model.params["K_gl"] = best_G
    model.params["w"] = 2 * np.pi * f_diff

    if not random_cond:
        # Set the directory where to save results
        savedir = str(SIM_DIR_SC)
        paths.HDF_DIR = savedir
        parameters, filename = prepare_subject_simulation_disconn(model, subj)
        for i in range(n_sim):
            print(f"Starting simulations n°: {i+1}/{n_sim}")
            # Initialize the search
            search = BoxSearch(
                model=model,
                evalFunction=evaluate_subj,
                parameterSpace=parameters,
                filename=filename,
            )
            search.run(chunkwise=True, chunksize=60000, append=True)
        fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(
            savedir, filename
        )
        create_df_results_sc_disconn(fc_pearson, phfcd_ks)
    else:
        print("There is no random for the disconnectivity model! Moving on!")


###########################################################
############### HETEROGENEOUS MODEL #######################
###########################################################


def prepare_subject_simulation_heterogeneous(subj, node_spared, ws, bs, random_cond):
    # Define the parameter space to explore
    # We load the spared structural connectivity matrix (e.g., if there were no WMH lesions on the tract, then the value of the matrix = 1) and calculate
    # how much the node was spared. If the node was totally spared, then n = 1, so w*(1-1) = 0, so we have a = -0.02 as normal. In case there is any type
    # of damage to the fiber connected to the node, then n < 1. The more damage to the node, the more n ~ 0 --> 1-n = 1 --> a = -0.02 - w *1 --> the more
    # negative - more chaotic - the behavior of each node is. Note that the same can be achieved by changing the node spared into a node damage matrix by
    # node_damage = 1 - node spared and changing: "a": [np.array([round(-0.02 - w * n + b, 5) for n in node_spared])for w in ws for b in bs] as described
    # for ease of use in the paper... Since the very lengthy simulations were run with this way, we keep it. Remember that if you opt for writing in the 
    # second form, you should also remove the line in the analysis script where we multiply the weights of the heterogeneous model by -1.
    parameters = ParameterSpace(
        {
            "a": [
                np.array([round(-0.02 - w * (1 - n) + b, 5) for n in node_spared])
                for w in ws
                for b in bs
            ]
        },
        kind="grid",
    )
    if not random_cond:
        filename = f"{subj}_heterogeneous_model.hdf"
    else:
        filename = f"{subj}_heterogeneous_model_random.hdf"

    return parameters, filename


def simulate_heterogeneous_model(subj, cn_wmh_arr, mci_wmh_arr, f_diff, best_G, ws, bs, random_cond):
    global search
    print(f"Now performing the simulations for the heterogeneous model...")
    if not random_cond:
        node_spared = get_node_spared(subj)
    else:
        np.random.seed(1991)
        node_spared = np.random.random(90)
        
    # Set the directory where to save results
    savedir = str(SIM_DIR_HET)
    paths.HDF_DIR = savedir

    Dmat = np.zeros_like(sc)
    # Initialize the model (neurolib wants a Dmat to initialize the mode,
    # so we gave it an empty Dmat, which we also later cancel by setting it to None)
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    model.params["Dmat"] = None
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["a"] = np.ones(n_nodes) * -0.02
    model.params["K_gl"] = best_G
    model.params["w"] = 2 * np.pi * f_diff
    parameters, filename = prepare_subject_simulation_heterogeneous(
        subj, node_spared, ws, bs, random_cond
    )
    for i in range(n_sim):
        print(f"Starting simulations n°: {i+1}/{n_sim}")
        # Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate_subj,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)
    fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(
        savedir, filename, ws=ws, bs=bs
    )
    create_df_results(SIM_DIR_HET, "heterogeneous", ws, bs, fc_pearson, phfcd_ks)


###########################################################
################### DELAY MODEL ###########################
###########################################################


def prepare_subject_simulation_delay(subj, ws, random_cond):
    print("Still to do")
    if not random_cond:
        filename = f"{subj}_delay_model.hdf"
    else:
        filename = f"{subj}_delay_model_random.hdf"

    parameters = None

    return parameters, filename


def simulate_delay_model():
    # Re initialize the model, this time with a delay matrix for the delay model
    print("The delay model is not ready yet!")
    print(
        "You have to download the new version of your neurolib and modify this accordingly"
    )
    # Dmat = np.zeros_like(sc)
    # # Initialize the model (neurolib wants a Dmat to initialize the model,
    # # so we gave it an empty Dmat, which we also later cancel by setting it to None)
    # model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    # model.params["Dmat"] = None
    # # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
    # model.params["duration"] = 11.65 * 60 * 1000
    # model.params["signalV"] = 0
    # model.params["dt"] = 0.1
    # model.params["sampling_dt"] = 10.0
    # model.params["sigma"] = 0.02

    # print(f"Now performing the simulations for the delay model...")
    # # Set the directory where to save results
    # paths.HDF_DIR = str(SIM_DIR_DELAY)
    # parameters, filename = prepare_subject_simulation_delay(subj, ws_del, random=random)
    # for i in range(n_sim):
    #     print(f"Starting simulations n°: {i+1}/{n_sim}")
    #     #Initialize the search
    #     search = BoxSearch(
    #         model=model,
    #         evalFunction=evaluate_subj,
    #         parameterSpace=parameters,
    #         filename=filename,
    #     )
    #     search.run(chunkwise=True, chunksize=60000, append=True)
    # fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(SIM_DIR_DEL, parameters, filename)
    # create_df_results(SIM_DIR_HET, "heterogeneous", ws, bs, fc_pearson, phfcd_ks)


# %% Get the frequencies for each group
f_diff_CN_no_wmh = get_f_diff_group(CN_no_WMH)
f_diff_CN_WMH = get_f_diff_group(CN_WMH)
f_diff_MCI_no_WMH = get_f_diff_group(MCI_no_WMH)
f_diff_MCI_WMH = get_f_diff_group(MCI_WMH)

###############################################################
#### Define the subject on which to perform the simulation ####
###############################################################
id_subj = int(sys.argv[1]) - 1
subj = subjs_to_sim[id_subj]
n_subjs = len(subjs_to_sim)
n_sim = 20
best_G = 2.0

#%%
################################################################
# Perform subject-wise simulations
################################################################
print(f"SLURM ARRAY TASK: {id_subj} corresponds to subject {subj}")
print(
    f"We are going to do {n_sim} simulations for subject {subj}..."
)

random_conditions = [True]

for random_value in random_conditions:
    if not random_value:
        SIM_DIR_A = (
            SIM_DIR / f"a-weight_ws_{ws_min_a}-{ws_max_a}_bs_{bs_min_a}-{bs_max_a}"
        )
        SIM_DIR_G = (
            SIM_DIR / f"G-weight_ws_{ws_min_G}-{ws_max_G}_bs_{bs_min_G}-{bs_max_G}"
        )
        SIM_DIR_SC = SIM_DIR / f"sc_disconn"
        SIM_DIR_HET = (
            SIM_DIR
            / f"heterogeneous_ws_{ws_min_het}-{ws_max_het}_bs_{bs_min_het}-{bs_max_het}"
        )
        SIM_DIR_DELAY = SIM_DIR / f"delay_ws_{ws_min_del}-{ws_max_del}"
        wmh_dict = get_wmh_load_homogeneous(subjs)
    else:
        SIM_DIR_A = (
            SIM_DIR
            / f"a-weight_ws_{ws_min_a}-{ws_max_a}_bs_{bs_min_a}-{bs_max_a}_random"
        )
        SIM_DIR_G = (
            SIM_DIR
            / f"G-weight_ws_{ws_min_G}-{ws_max_G}_bs_{bs_min_G}-{bs_max_G}_random"
        )
        SIM_DIR_SC = SIM_DIR / f"sc_disconn"
        SIM_DIR_HET = (
            SIM_DIR
            / f"heterogeneous_ws_{ws_min_het}-{ws_max_het}_bs_{bs_min_het}-{bs_max_het}_random_totally"
        )
        SIM_DIR_DELAY = SIM_DIR / f"delay_ws_{ws_min_del}-{ws_max_del}_random"
        wmh_dict = get_wmh_load_random(subjs)

    sim_dir = [SIM_DIR_HET] # SIM_DIR_A, SIM_DIR_G, SIM_DIR_HET, SIM_DIR_SC
    for path in sim_dir:
        if not Path.exists(path):
            Path.mkdir(path)

    if subj in CN_WMH:
        f_diff = f_diff_CN_WMH
    elif subj in MCI_WMH:
        f_diff = f_diff_MCI_WMH

    # simulate_homogeneous_model_a(
    #     subj=subj,
    #     f_diff=f_diff,
    #     best_G=best_G,
    #     wmh_dict=wmh_dict,
    #     ws=ws_a,
    #     bs=bs_a,
    #     random_cond=random_value,
    # )
    # simulate_homogeneous_model_G(
    #     subj=subj,
    #     f_diff=f_diff,
    #     wmh_dict=wmh_dict,
    #     ws=ws_G,
    #     bs=bs_G,
    #     random_cond=random_value,
    # )

    simulate_heterogeneous_model(
        subj=subj,
        cn_wmh_arr = CN_WMH,
        mci_wmh_arr = MCI_WMH,
        f_diff=f_diff,
        best_G=best_G,
        ws=ws_het,
        bs=bs_het,
        random_cond=random_value,
    )
    # simulate_disconn_model(
    #     subj=subj, best_G=best_G, f_diff=f_diff, random_cond=random_value
    # )
    print("The end.")
