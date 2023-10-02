#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 1.0
Last edit:  2023/07/11
Authors:    Leone, Riccardo (RL)
Notes:      - Model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * All together in one script
To do:      - Model with delay
            - Change simulation time to the correct time
Comments:   

Sources: 
"""
# %% Initial imports
import sys
import matplotlib.pyplot as plt
import filteredPowerSpectralDensity as filtPowSpectr
import neurolib.utils.functions as func
import my_functions as my_func
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.utils import paths
from neurolib.utils import pypetUtils as pu
from petTOAD_setup import *

def get_f_diff_group(group):
    # Get the timeseries for the HC no WMH group
    group, timeseries = get_group_ts_for_freqs(group, all_fMRI_clean)
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR)
    f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])
    return f_diff

# Define the evaluation function for the model
def evaluate_group(traj):
    # For G we want accrue several simulations of different patients, especially for phfcd,
    # so we do them one by one and we gather results at the end.
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

def calculate_results_from_bolds_group(bold_arr, nparms, nsim):
    # Create a new array to store the FC, FCD and phFCD values with the same shape as bold array
    fc_array = np.zeros([nparms, nsim, n_nodes, n_nodes])
    phfcd_array = np.zeros([nsim, nparms, 465])

    # Iterate over each element in the bold array
    for i in range(nparms):
        for j in range(nsim):
            print(f"Now calculating results from the {j} simulation for parameter {i}...")
            # Get the current timeseries
            timeseries = bold_arr[i, j].squeeze()

            # Perform FC, FCD and phFCD analysis
            if np.isnan(timeseries).any():
                print("Simulation has some nans, aborting!")
                continue
            else:
                print("Simulation has no nans, good to go")
                print("Calculating FC..")
                fc_value = func.fc(timeseries)
                print("Calculating phFCD")
                phfcd_value = my_func.phFCD(timeseries)
                # Store the FC, FCD, phFCD value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
                phfcd_array[i, j] = phfcd_value
    return fc_array, phfcd_array

# Define the evaluation function for the subjectwise simulations
# def evaluate_subj(traj):
#     model = search.getModelFromTraj(traj)
#     bold_list = []

#     model.randomICs()
#     model.run(chunkwise=True, chunksize=60000, append=True)
#     # skip the first 180 secs to "warm up" the timeseries
#     ts = model.outputs.x[:, 18000::300]
#     t = model.outputs.t[18000::300]
#     ts_filt = BOLDFilters.BandPassFilter(ts)
#     bold_list.append(ts_filt)
#     # if not np.isnan([np.isnan(t).any() for t in bold_list]).any():
#     print("There are no nans. Continue with the simulations")
#     result_dict = {}
#     result_dict["BOLD"] = np.array(bold_list).squeeze()
#     result_dict["t"] = t
#     search.saveToPypet(result_dict, traj)
#     # else:
#     #     print("There are some nans, aborting...")
#     #     return

def evaluate_subj_new(traj):
    # Remember to put the ts_subj in the main part of the script!
    model = search.getModelFromTraj(traj)
    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 180 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 18000::300]
    t = model.outputs.t[18000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    if not np.isnan(ts_filt).any():
        print("There are no nans. Continue with the simulations")
    else:
        print("There are some nans, aborting...")
        return
    result_dict = {}
    result_dict["fc_pearson"] = func.fc(ts_filt, func.fc(ts_subj))
    result_dict["phfcd_ks"] = my_func.ts_kolmogorov_phfcd(ts_filt, ts_subj)
    
    search.saveToPypet(result_dict, traj)


def calculate_results_from_bolds_subject(bold_arr, nsim, nparms, n_nodes):
    # Create a new array to store the FC and phFCD values with the same shape as bold array
    fc_array = np.zeros([nsim, nparms, n_nodes, n_nodes])
    phfcd_array = np.zeros([nsim, nparms, 465]) # 18145 need to change it so you don't have to update it manually..

    # Iterate over each element in the bold array
    for i in range(nsim):
        for j in range(nparms):
            print(f"Now calculating results from the {i} simulation for parameter {j}...")
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


def get_multiple_trajs(filename, trajs):
    big_list = []
    for traj in trajs:
        traj_list = []
        tr = pu.loadPypetTrajectory(filename, traj)
        run_names = tr.f_get_run_names()
        n_run = len(run_names)
        ns = range(n_run)
        for i in ns:
            r = pu.getRun(i, tr)
            traj_list.append(r['BOLD'])
        big_list.append(traj_list) 
    return np.array(big_list)

def plot_results_exploration_G(df_res, savedir, savename):
    plt.figure()
    plt.plot(df_res["K_gl"], df_res["fc_pearson"], label="FC")
    plt.plot(df_res["K_gl"], df_res["phfcd_ks"], label="phFCD")
    plt.xlabel("Coupling parameter (G)")
    plt.ylabel(r"Pearson's $\rho$ / KS-distance")
    plt.legend()
    plt.savefig(savedir / f"{savename}_plot.png")


def find_best_G(f_diff_hc_no_wmh):
    global search
    # Create the results dir for the HC_no_WMH group
    HC_no_WMH_DIR = SIM_DIR / "HC_no_WMH"
    if not Path.exists(HC_no_WMH_DIR):
        Path.mkdir(HC_no_WMH_DIR)
    # First we want to find the best G with a fixed a == -0.02
    HC_no_WMH_dict = {k:v for k, v in all_fMRI_clean.items() if k in HC_no_WMH}

    fc, _, phfcd = my_func.calc_and_save_group_stats(
        HC_no_WMH_dict, HC_no_WMH_DIR
    )

    # Set the directory where to save results and the filename
    paths.HDF_DIR = str(HC_no_WMH_DIR)
    filename = "initial_exploration_Gs.hdf"

    # Initialize the model (neurolib wants a Dmat to initialize the mode,
    # so we gave it an empty Dmat, which we also later cancel by setting it to None)
    Dmat = np.zeros_like(sc)
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
    model.params["duration"] = 4.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["w"] = 2 * np.pi * f_diff_hc_no_wmh
    model.params["a"] = np.ones(n_nodes) * (-0.02)
    # Set the minimum and maximum values of G to explore
    G_min = 0.0
    G_max = 3.5
    # Define the parameters space to explore for G (K_gl)
    parameters = ParameterSpace(
        {"K_gl": np.round(np.linspace(G_min, G_max, 2), 2)}, kind="grid"
    )

    nsim = 2 #len(group)

    for i in range(nsim):
        print(f"Now performing simulation number {i+1}/{nsim}...")
        # Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate_group,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)

    trajs = pu.getTrajectorynamesInFile(f"{paths.HDF_DIR}/{filename}")
    nparms = len(parameters.K_gl)  
    bold_arr = get_multiple_trajs(f"{paths.HDF_DIR}/{filename}", trajs)
    # Create a new array to store the FC and phFCD values with the same shape as bold array
    fc_array = np.zeros([nsim, nparms, n_nodes, n_nodes])
    phfcd_array = np.zeros([nsim, nparms, 465]) # need to change it so you don't have to update it manually..

    # Iterate over each element in the bold array
    for i in range(nsim):
        for j in range(nparms):
            print(f"Now calculating results from the {i} simulation for parameter {j}...")
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
                phfcd_array[i, j] = phfcd_value    # Calculate the mean fc for the whole group
    sim_fc = fc_array.mean(axis=0)
    print("Calculating fcs...")
    fc_pearson = [func.matrix_correlation(row_fc, fc) for row_fc in sim_fc]
    print("Calculating phFCDs...")
    phfcd_ks = [
        my_func.matrix_kolmogorov(phfcd, np.concatenate(row)) for row in phfcd_array
    ]
    data = [[parameters.K_gl, fc_pearson, phfcd_ks]]
    columns = ["K_gl", "fc_pearson", "phfcd_ks"]
    res_df = pd.DataFrame(data, columns=columns).explode(columns)
    res_df.to_csv(HC_no_WMH_DIR / f"df_results_{filename}.csv")
    plot_results_exploration_G(res_df, SIM_DIR, "G_plot")
    best_G = res_df[res_df["phfcd_ks"] == res_df["phfcd_ks"].min()]["K_gl"][0]
    print("Done with finding the G!")
    print(f"The best G in the explored range {G_min}-{G_max} is {best_G}")

    return best_G



#%% Prepare the subject-wise simulations
def prepare_subject_simulation_homogeneous_a(subj, WMH, ws, bs, random):

    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "a": [(np.ones(n_nodes) * -0.02) + w * WMH + b for w in ws for b in bs],
        },
        kind="grid",
    )
    if not random:
        filename = f"{subj}_homogeneous_a-weight_model.hdf"
    else:
        filename = f"{subj}_homogeneous_a-weight_model_random.hdf"
     
    return parameters, filename


def prepare_subject_simulation_homogeneous_G(subj, WMH, ws, bs, random):
    
    # Define the parametere space to explore
    parameters = ParameterSpace(
        {
            "K_gl": [float(round(best_G + w * WMH + b, 3)) for w in ws for b in bs],
        },
        kind="grid",
    )
    if not random:
        filename = f"{subj}_homogeneous_G-weight_model.hdf"
    else:
        filename = f"{subj}_homogeneous_G-weight_model_random.hdf"

    return parameters, filename

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


def prepare_subject_simulation_heterogeneous(subj, node_damage, ws, bs, random):
    # Define the parametere space to explore
    # We load the spared structural connectivity matrix (e.g., if there were no WMH lesions on the tract, then the value of the matrix = 1) and calculate
    # how much the node was spared. If the node was totally spared, then n = 1, so w*(1-1) = 0, so we have a = -0.02 as normal. In case there is any type
    # of damage to the fiber connected to the node, then n < 1. The more damage to the node, the more n ~ 0 --> 1-n = 1 --> a = -0.02 - w *1 --> the more 
    # negative - more chaotic - the behavior of each node is. 
    parameters = ParameterSpace(
        {
            "a": [np.array([round(-0.02 - w*(1-n) + b, 5) for n in node_damage]) for w in ws for b in bs]
        },
        kind="grid",
    )
    if not random:
        filename = f"{subj}_heterogeneous_model.hdf"
    else:
        filename = f"{subj}_heterogeneous_model_random.hdf"

    return parameters, filename

def prepare_subject_simulation_delay(subj, ws, random):
    print("Still to do")
    if not random:
        filename = f"{subj}_delay_model.hdf"
    else:
        filename = f"{subj}_delay_model_random.hdf"

    parameters = None

    return parameters, filename

def gather_results_from_repeated_simulations(res_dir, filename, ws, bs):
    
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
            traj_list.append(r['BOLD'])
        big_list.append(traj_list) 
    bold_arr = np.array(big_list)
    nsim = len(trajs)
    if res_dir == str(SIM_DIR_SC):
        nparms = 1     
    else:    
        nparms = len(ws * bs)
    fc_array, phfcd_array = calculate_results_from_bolds_subject(bold_arr, nsim, nparms, n_nodes)
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
        row_phfcd_ks = [my_func.matrix_kolmogorov(phfcd, sim_phfcd) for sim_phfcd in row]
        phfcd_ks.append(row_phfcd_ks)
        phfcd_ks_arr = np.array(phfcd_ks)
    phfcd_ks = phfcd_ks_arr.mean(axis=0)
    return fc_pearson, phfcd_ks

def create_df_plot_results(res_dir, model, ws, bs, fc_pearson, phfcd_ks):
    data = [[[(round(b, 3), round(w, 3)) for w in ws for b in bs], fc_pearson, phfcd_ks]]
    columns = ["b_w", "fc_pearson", "phfcd_ks"]
    res_df = pd.DataFrame(data, columns=columns).explode(columns)
    res_df['b'], res_df['w'] = zip(*res_df.b_w)
    res_df = res_df.drop(columns=['b_w'])
    res_df.to_csv(res_dir / f"sub-{subj}_df_results_{model}.csv")
    #save_plot_results(res_df)

def create_df_plot_results_sc_disconn(fc_pearson, phfcd_ks):
    data = [[fc_pearson, phfcd_ks]]
    columns = ["fc_pearson", "phfcd_ks"]
    res_df = pd.DataFrame(data, columns=columns).explode(columns)
    res_df.to_csv(SIM_DIR_SC / f"sub-{subj}_df_results_disconn.csv")

#%% Define the final simulations
def simulate_homogeneous_model_a(subj, f_diff, best_G, wmh_dict, ws, bs, random):
        global search
        WMH = wmh_dict[subj]
        print(f"Now performing the simulations for the homogeneous a-weighted model with random = {random}...")
        Dmat = np.zeros_like(sc)
        # Initialize the model (neurolib wants a Dmat to initialize the mode,
        # so we gave it an empty Dmat, which we also later cancel by setting it to None)
        model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
        # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
        model.params["duration"] = 4.65 * 60 * 1000
        model.params["signalV"] = 0
        model.params["dt"] = 0.1
        model.params["sampling_dt"] = 10.0
        model.params["sigma"] = 0.02
        model.params["w"] = 2 * np.pi * f_diff
        model.params["K_gl"] = best_G
        # Set the directory where to save results
        savedir = str(SIM_DIR_A)
        paths.HDF_DIR = savedir
        parameters, filename = prepare_subject_simulation_homogeneous_a(subj, WMH, ws, bs, random=random)        
        for i in range(nsim):
            print(f"Starting simulations n°: {i+1}/{nsim}")
            #Initialize the search
            search = BoxSearch(
                model=model,
                evalFunction=evaluate_group,
                parameterSpace=parameters,
                filename=filename,
            )
            search.run(chunkwise=True, chunksize=60000, append=True)
        fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(savedir, filename, ws, bs)
        create_df_plot_results(SIM_DIR_A, "homogeneous_a-weight", ws, bs, fc_pearson, phfcd_ks)

def simulate_homogeneous_model_G(subj, f_diff, wmh_dict, ws, bs, random): 
        global search       
        print(f"Now performing the simulations for the homogeneous G-weighted model with random = {random}...")
        WMH = wmh_dict[subj]
        Dmat = np.zeros_like(sc)
        # Initialize the model (neurolib wants a Dmat to initialize the mode,
        # so we gave it an empty Dmat, which we also later cancel by setting it to None)
        model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
        # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
        model.params["duration"] = 4.65 * 60 * 1000
        model.params["signalV"] = 0
        model.params["dt"] = 0.1
        model.params["sampling_dt"] = 10.0
        model.params["sigma"] = 0.02
        model.params["w"] = 2 * np.pi * f_diff
        model.params["a"] = np.ones(n_nodes) * -0.02
        # Set the directory where to save results
        savedir = str(SIM_DIR_G)
        paths.HDF_DIR = savedir
        parameters, filename = prepare_subject_simulation_homogeneous_G(subj, WMH, ws, bs, random=random)        
        for i in range(nsim):
            print(f"Starting simulations n°: {i+1}/{nsim}")
            #Initialize the search
            search = BoxSearch(
                model=model,
                evalFunction=evaluate_group,
                parameterSpace=parameters,
                filename=filename,
            )
            search.run(chunkwise=True, chunksize=60000, append=True)
        fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(savedir, filename, ws, bs)
        create_df_plot_results(SIM_DIR_G, "homogeneous_G-weight", ws, bs, fc_pearson, phfcd_ks)

def simulate_disconn_model(subj, f_diff, random):
        global search       
        print(f"Now performing the simulations for the structural disconnectivity model...")
        Dmat = np.zeros_like(sc)
        # Initialize the model (neurolib wants a Dmat to initialize the mode,
        # so we gave it an empty Dmat, which we also later cancel by setting it to None)
        model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
        # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
        model.params["duration"] = 4.65 * 60 * 1000
        model.params["signalV"] = 0
        model.params["dt"] = 0.1
        model.params["sampling_dt"] = 10.0
        model.params["sigma"] = 0.02
        model.params["a"] = np.ones(n_nodes) * -0.02
        model.params["K_gl"] = best_G
        model.params["w"] = 2 * np.pi * f_diff

        if not random:
            # Set the directory where to save results
            savedir = str(SIM_DIR_SC)
            paths.HDF_DIR = savedir
            parameters, filename = prepare_subject_simulation_disconn(model, subj)        
            for i in range(nsim):
                print(f"Starting simulations n°: {i+1}/{nsim}")
                #Initialize the search
                search = BoxSearch(
                    model=model,
                    evalFunction=evaluate_group,
                    parameterSpace=parameters,
                    filename=filename,
                )
                search.run(chunkwise=True, chunksize=60000, append=True)
        else:
            print("There is no random for the disconnectivity model! Moving on!")
        fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(savedir, filename)
        create_df_plot_results_sc_disconn(fc_pearson, phfcd_ks)

def simulate_heterogeneous_model(subj, f_diff, best_G, ws, bs, random):        
    global search
    print(f"Now performing the simulations for the heterogeneous model...")
    if not random:
        node_damage = get_node_damage(subj)
    else:
        node_damage = get_node_damage(subj)
    # random still to do
    # wmh_rand = np.array([w for w in wmh_damage.values()])
    # np.random.seed(1991)
    # np.random.shuffle(wmh_rand)
    Dmat = np.zeros_like(sc)
    # Initialize the model (neurolib wants a Dmat to initialize the mode,
    # so we gave it an empty Dmat, which we also later cancel by setting it to None)
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
    model.params["duration"] = 4.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["a"] = np.ones(n_nodes) * -0.02
    model.params["K_gl"] = best_G
    model.params["w"] = 2 * np.pi * f_diff
    # Set the directory where to save results (need to convert as string)
    savedir = str(SIM_DIR_HET)
    paths.HDF_DIR = savedir
    parameters, filename = prepare_subject_simulation_heterogeneous(subj, node_damage, ws_het, bs_het, random=random_conditions[0])        
    for i in range(nsim):
        print(f"Starting simulations n°: {i+1}/{nsim}")
        #Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate_group,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)
        fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(savedir, filename, ws, bs)
        create_df_plot_results(SIM_DIR_HET, "heterogeneous", ws, bs, fc_pearson, phfcd_ks)    

def simulate_delay_model():
        # Re initialize the model, this time with a delay matrix for the delay model
        print("The delay model is not ready yet!")            
        print("You have to download the new version of your neurolib and modify this accordingly")
        # Dmat = np.zeros_like(sc)
        # # Initialize the model (neurolib wants a Dmat to initialize the model,
        # # so we gave it an empty Dmat, which we also later cancel by setting it to None)
        # model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
        # model.params["Dmat"] = None 
        # # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
        # model.params["duration"] = 12.65 * 60 * 1000
        # model.params["signalV"] = 0
        # model.params["dt"] = 0.1
        # model.params["sampling_dt"] = 10.0
        # model.params["sigma"] = 0.02

        # print(f"Now performing the simulations for the delay model...")
        # # Set the directory where to save results
        # savedir = str(SIM_DIR_DELAY)
        # paths.HDF_DIR = savedir
        # parameters, filename = prepare_subject_simulation_delay(subj, ws_del, random=random)        
        # for i in range(nsim):
        #     print(f"Starting simulations n°: {i+1}/{nsim}")
        #     #Initialize the search
        #     search = BoxSearch(
        #         model=model,
        #         evalFunction=evaluate_subj,
        #         parameterSpace=parameters,
        #         filename=filename,
        #     )
        #     search.run(chunkwise=True, chunksize=60000, append=True)
        #fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(savedir, filename, ws, bs)
        #create_df_plot_results(SIM_DIR_HET, "heterogeneous", ws, bs, fc_pearson, phfcd_ks)    

# %% Get the frequencies for each group
f_diff_hc_no_wmh = get_f_diff_group(HC_no_WMH)
f_diff_hc_wmh = get_f_diff_group(HC_WMH)
f_diff_mci_no_wmh = get_f_diff_group(MCI_no_WMH)
f_diff_mci_wmh = get_f_diff_group(MCI_WMH)

perform_G_search = sys.argv[1]
best_G = find_best_G(f_diff_hc_no_wmh)
short_subjs = ["ADNI002S1261"]
#%%
#############################################################
# Parameter setup for subject-wise simulations
#############################################################

#  Set the exploration values for the homogeneous model of a
# Set the minimum and maximum values of w and b you want to explore for the bifurcation parameters 
# homogeneous model and random model.

# VERY IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Be sure to include w = 0 and b = 0, because this would be the baseline model!!!
ws_min_a = -0.1
ws_max_a = 0.1
bs_min_a = -0.025
bs_max_a = 0.025
# Set the number of parameters you want your min-max interval to be split into (remember that the n
# of simulations is equal to n_ws_a * n_bs_a, so make it reasonable to the type of PC/HPC you are running
# your simulations in)
n_ws_a = 2 #21
n_bs_a = 1 #5
# Create the final array with all the ws and bs you want to explore
ws_a = np.linspace(ws_min_a, ws_max_a, n_ws_a)
bs_a = np.linspace(bs_min_a, bs_max_a, n_bs_a)

#  Set the exploration values for the homogeneous model of G
# Set the minimum and maximum values of w and b you want to explore for the coupling parameter
# homogeneous model and random model
ws_min_G = -1
ws_max_G = 1
bs_min_G = -0.8
bs_max_G = 0.08
# Set the number of parameters you want your min-max interval to be split into (remember that the n
# of simulations is equal to n_ws_a * n_bs_a, so make it reasonable to the type of PC/HPC you are running
# your simulations in)
n_ws_G = 2 #21
n_bs_G = 1 #5
# Create the final array with all the ws and bs you want to explore
ws_G = np.linspace(ws_min_G, ws_max_G, n_ws_G)
bs_G = np.linspace(bs_min_G, bs_max_G, n_bs_G)

#  Set the exploration values for the heterogeneous model
# Set the minimum and maximum values of w and b you want to explore for the coupling parameter
# heterogeneous model and random model
ws_min_het = 0.05
ws_max_het = 0.15
bs_min_het = -0.8
bs_max_het = 0.08
# Set the number of parameters you want your min-max interval to be split into 
n_ws_het = 2 # 21
n_bs_het = 1 # 5
# Create the final array with all the ws and bs you want to explore
ws_het = np.linspace(ws_min_het, ws_max_het, n_ws_het)
bs_het = np.linspace(bs_min_het, bs_max_het, n_bs_het)

#  Set the exploration values for the delay model in terms of number of dt
ws_min_del = 1
ws_max_del = 1000
# Set the number of parameters you want your min-max interval to be split into 
n_ws_del = 1 # 21
# Create the final array with all the ws and bs you want to explore
ws_del = np.linspace(ws_min_del, ws_max_del, n_ws_del)

n_subjs = len([s for s in subjs if s in HC_WMH or s in MCI_WMH])
nsim = 1

#%%
# Prompt the user for confirmation
user_input = input(f"Please check that the graph makes sense and that G = {best_G} corresponds to what you see. Is it correct? (y/n): ")

# Check user input and continue based on the response
if user_input.lower() == "n":
    # Proceed with the script
    print("Script execution aborted...")

else:
    #############################################################
    # Perform subject-wise simulations
    ################################################################
    print(f"We are going to do {nsim} simulations for a total of {n_subjs} in the HC and MCI WMH groups")

    random_conditions = [False, True]

    for random_value in random_conditions:

        if not random_value:
            SIM_DIR_A = RES_DIR / f"homogeneous_a-weight_ws_{ws_min_a}-{ws_max_a}_bs_{bs_min_a}-{bs_max_a}"
            SIM_DIR_G = RES_DIR / f"homogeneous_G-weight_ws_{ws_min_G}-{ws_max_G}_bs_{bs_min_G}-{bs_max_G}"
            SIM_DIR_SC = RES_DIR / f"sc_disconn"
            SIM_DIR_HET = RES_DIR / f"heterogeneous_ws_{ws_min_het}-{ws_max_het}_bs_{bs_min_het}-{bs_max_het}"
            SIM_DIR_DELAY = RES_DIR / f"delay_ws_{ws_min_del}-{ws_max_del}"
            wmh_dict = get_wmh_load_homogeneous(subjs)

        else:
            SIM_DIR_A = RES_DIR / f"homogeneous_a-weight_ws_{ws_min_a}-{ws_max_a}_bs_{bs_min_a}-{bs_max_a}_random"
            SIM_DIR_G = RES_DIR / f"homogeneous_G-weight_ws_{ws_min_G}-{ws_max_G}_bs_{bs_min_G}-{bs_max_G}_random"
            SIM_DIR_HET = RES_DIR / f"heterogeneous_ws_{ws_min_het}-{ws_max_het}_bs_{bs_min_het}-{bs_max_het}_random"
            SIM_DIR_DELAY = RES_DIR / f"delay_ws_{ws_min_del}-{ws_max_del}_random"

            print("You should look at the random again because I am not convinced that the log is the best...")
            # wmh_dict_pre = get_wmh_load_homogeneous(subjs)
            # wmh_dict_log = np.array([np.log(w) if w != 0 else 0 for w in wmh_dict_pre.values()])
            # wmh_dict_log_z = (wmh_dict_log - wmh_dict_log.min()) / (
            #     wmh_dict_log.max() - wmh_dict_log.min()
            # )
            # np.random.seed(1991)
            # np.random.shuffle(wmh_dict_log_z)
            # wmh_dict = {k: wmh_dict_log_z[n] for n, k in enumerate(wmh_dict_pre.keys())}


        sim_dirs = [SIM_DIR_A, SIM_DIR_G, SIM_DIR_SC, SIM_DIR_HET, SIM_DIR_DELAY]
        for sim_dir in sim_dirs:
            if not Path.exists(sim_dir):
                Path.mkdir(sim_dir)
    
        for j, subj in enumerate(short_subjs):
            print(f"Starting simulations for subject: {subj}, ({j + 1}/{n_subjs})")
            if subj in HC_WMH:
                f_diff = f_diff_hc_wmh
            elif subj in MCI_WMH:
                f_diff = f_diff_mci_wmh
            elif subj in HC_no_WMH:
                continue
            elif subj in MCI_no_WMH:
                continue
            ts_subj = all_fMRI_clean[subj]        
            #simulate_homogeneous_model_a(subj = subj, f_diff = f_diff, best_G = best_G, wmh_dict = wmh_dict, ws = ws_a, bs = bs_a, random = random_value)
            #simulate_homogeneous_model_G(subj = subj, f_diff = f_diff, wmh_dict = wmh_dict, ws = ws_G, bs = bs_G, random = random_value)  
            #simulate_disconn_model(subj = subj, f_diff = f_diff, random = random_value)
            simulate_heterogeneous_model(subj = subj, f_diff = f_diff, best_G = best_G, ws = ws_het, bs = bs_het, random = random_value)
            # simulate_delay_model()        
# %%
