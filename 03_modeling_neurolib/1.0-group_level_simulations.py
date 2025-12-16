# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Group model simulation with neurolib   -- Version 1.0
Last edit:  2023/11/13
Authors:    Leone, Riccardo (RL)
Notes:      - Running group-level simulations for CN WMH, MCI no/with WMH at a group level
            - Release notes:
                * Initial release
To do:      - 
Comments:   

Sources: 
"""

# %%
# %% Initial imports
from neurolib.models.hopf import HopfModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from neurolib.utils import pypetUtils as pu
from neurolib.utils.parameterSpace import ParameterSpace

import filteredPowerSpectralDensity as filtPowSpectr
import my_functions as my_func
from petTOAD_setup import *
from phFCD import *

def get_f_diff_group(group):
    # Get the timeseries for the chosen group
    group, timeseries = get_group_ts_for_freqs(group, all_fMRI_clean)
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR)
    f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])
    return f_diff


# Define the evaluation function for the subjectwise simulations
def evaluate(traj):
    # Get the trajectory from the search object
    model = search.getModelFromTraj(traj)
    # Create an empty list to store simulated BOLD
    bold_list = []
    # Initialize model with random initial conditions
    model.randomICs()
    # Run the model
    model.run(chunkwise=True, chunksize=60000, append=True)
    # Skip the first 120 secs to "warm up" the timeseries
    # Sample every 3 secs (3000/10 of downsampling = 300)
    ts = model.outputs.x[:, 12000::300]
    t = model.outputs.t[12000::300]
    # Apply the same z-score and detrending as the empirical data
    ts_filt = BOLDFilters.BandPassFilter(ts)
    # Append to the BOLD list
    bold_list.append(ts_filt)
    # We don't want the timeseries to have nans. If it has it means that the model "exploded"
    if not np.isnan([np.isnan(t).any() for t in bold_list]).any():
        print("No nans, good to go...")
    else:
        print("There are some nans, aborting...")

    # Create and save a dictionary with the resulting processed BOLD
    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list).squeeze()
    result_dict["t"] = t
    search.saveToPypet(result_dict, traj)

# %%
def initialize_Hopf(fdiff):
    Dmat = np.zeros_like(sc)
    # Initialize the model (neurolib wants a Dmat to initialize the mode,
    model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
    # Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 2 min of initial warm up of the timeseries
    model.params["duration"] = 11.65 * 60 * 1000
    model.params["signalV"] = 0
    model.params["dt"] = 0.1
    model.params["sampling_dt"] = 10.0
    model.params["sigma"] = 0.02
    model.params["w"] = 2 * np.pi * fdiff
    return model

def prepare_parms(group, model_type, G):
    filename = f"group-{group}_model-{model_type}.hdf"
    if model_type == "homogeneous_a":
        # Define the parametere space to explore
        parameters = ParameterSpace(
            {
                "a": [
                    np.round(np.ones(n_nodes) * new_a, 3)
                    for new_a in np.linspace(-0.1, -0.02, 81)
                ],
                "K_gl": [G],
            },
            kind="grid",
        )
        return parameters, filename

    elif model_type == "homogeneous_G":
        if group == "CN_no_WMH":
            # dg otherwise the rightmost value is not included, sets how the delta G you want to explore
            # Since we explore a potentially much larger parameter space in CN_no_WMH we set the exploration
            # "granularity" to  a
            dg = 0.02
            parameters = ParameterSpace(
                {
                    "a": [np.ones(n_nodes) * -0.02],
                    "K_gl": np.round(np.arange(0, G + dg, dg), 2),
                },
                kind="grid",
            )
            return parameters, filename

        elif group == "MCI_no_WMH":
            return
        else:
            # Define the parametere space to explore
            dg = 0.01
            parameters = ParameterSpace(
                {
                    "a": [np.ones(n_nodes) * -0.02],
                    "K_gl": np.round(np.arange(0, G + dg, dg), 2),
                },
                kind="grid",
            )
            return parameters, filename

def calculate_results_from_bolds(bold_arr, n_sim, n_parms, n_nodes):
    # Create a new array to store the FC and phFCD values for each parameter combination and simulation
    fc_array = np.zeros([n_sim, n_parms, n_nodes, n_nodes])
    phfcd_array = np.zeros([n_sim, n_parms, 18336])

    # Iterate over each element in the bold array
    for i in range(n_sim):
        for j in range(n_parms):
            print(
                f"Now calculating results from the {i+1}th simulation for parameter {j}..."
            )
            # Get the current timeseries
            timeseries = bold_arr[i, j].squeeze()

            # Recheck the timeseries for NaNs
            if np.isnan(timeseries).any():
                print("Simulation has some nans, aborting!")
                continue
            else:
                print("Simulation has no nans, good to go")
                print("Calculating FC..")
                fc_value = my_func.fc(timeseries)
                print("Calculating phFCD")
                phfcd_value = phFCD(timeseries)
                # Store the FC and phFCD value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
                phfcd_array[i, j] = phfcd_value
    return fc_array, phfcd_array


def gather_results_from_repeated_simulations(model_type, group, grouplist, savedir, filename):
    # We get the trajectory names in the group simulation we want
    trajs = pu.getTrajectorynamesInFile(f"{savedir}/{filename}")
    # Create a big list to store all results
    big_list = []
    # For every trajectory name we load the corresponding trajectory and store the associated bold for all runs
    for traj in trajs:
        traj_list = []
        tr = pu.loadPypetTrajectory(f"{savedir}/{filename}", traj)
        run_names = tr.f_get_run_names()
        n_run = len(run_names)
        ns = range(n_run)
        for i in ns:
            r = pu.getRun(i, tr)
            traj_list.append(r["BOLD"])
        big_list.append(traj_list)
    # Convert the big list of BOLD for every run (combination of parameters) for every trajector (number of simulations)
    # to a numpy array
    bold_arr = np.array(big_list)
    n_sim = bold_arr.shape[0]
    n_parms = bold_arr.shape[1]
    # Calculate the arrays of FC and phFCD
    fc_array, phfcd_array = calculate_results_from_bolds(
        bold_arr, n_sim, n_parms, n_nodes
    )
    np.savez_compressed(f"{savedir}/group-{group}_data-simulated_model-{model_type}_desc-fc-phfcd-arr.npz", fc = fc_array, phfcd = phfcd_array)
    dict_group = {subj: all_fMRI_clean[subj] for subj in grouplist}
    # Get the group-averaged empirical FC and the concatenated empirical phFCD
    emp_fc, _, emp_phFCD = my_func.calc_and_save_group_stats(dict_group, Path(savedir))
    # Store the correlation between each run of the simulation and the empirical average and between each phFCD and the concatenated average
    sim_fc = fc_array.mean(axis=0)
    fc_pearson = [my_func.matrix_correlation(row_sim_fc, emp_fc) for row_sim_fc in sim_fc]
    phfcd_ks = [my_func.matrix_kolmogorov(phfcd_array[:, n].flatten(), emp_phFCD) for n in range(phfcd_array.shape[1])]
    return fc_pearson, phfcd_ks
    
def create_df_results_homo(parm_name, parms, fc_pearson, phfcd_ks, savedir, group):
    if parm_name == "a":
        model_name = "homogeneous_a"
    elif parm_name == "K_gl":
        model_name = "homogeneous_G"
        
    res_df = pd.DataFrame({parm_name: parms,
                           "fc_pearson": fc_pearson,
                           "phfcd_ks": phfcd_ks})
    res_df.to_csv(f"{savedir}/group-{group}_model-{model_name}_desc-df-fitting-results.csv")
    return res_df


def create_df_results(model_type, parameters, fc_pearson, phfcd_ks, savedir, group):
    
    if model_type == "homogeneous_a":
        parm_name = "a"
        parms = parameters.a
        res_df = create_df_results_homo(parm_name, parms, fc_pearson, phfcd_ks, savedir, group)         
        return parm_name, res_df
           
    elif model_type == "homogeneous_G":
        parm_name = "K_gl"
        parms = parameters.K_gl
        res_df = create_df_results_homo(parm_name, parms, fc_pearson, phfcd_ks, savedir, group)
        if group == "CN_no_WMH":
            best_G = res_df[res_df["phfcd_ks"] == res_df["phfcd_ks"].min()]["K_gl"]
            best_G.to_csv(SIM_GROUP_DIR / "group-CN-no-WMH_desc-best-G.csv")
        return parm_name, res_df


def simulate_group(group, grouplist, fdiff, G, nsim, model_type):
    global search
    # Set the directory where to save results
    savedir = str(SIM_GROUP_DIR / group)
    paths.HDF_DIR = savedir
    print(f"Now performing the simulations for the {group} group, model: {model_type}")
    model = initialize_Hopf(fdiff)
    parameters, filename = prepare_parms(group, model_type, G)
    for i in range(nsim):
        print(f"Starting simulations n°: {i+1}/{nsim}")
        # Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)
    fc_pearson, phfcd_ks = gather_results_from_repeated_simulations(
        model_type,
        group,
        grouplist,
        savedir,
        filename,
    )
    create_df_results(
            model_type, parameters, fc_pearson, phfcd_ks, savedir, group
        )
    print("Done!")

def run_group_level_simulations(groupdict, dict_model_types_group):
    for group_name, group_list in groupdict.items():
        nsim = len(group_list)
        for model_type in dict_model_types_group[group_name]:
            fdiff = get_f_diff_group(group_list)
            if group_name == "CN_no_WMH":
                G = 3.5
            else:
                np_G = pd.read_csv(SIM_GROUP_DIR / "group-CN-no-WMH_desc-best-G.csv")["K_gl"][0]
                G = float(np_G)
            simulate_group(group_name, group_list, fdiff, G, nsim, model_type)

###############################################################################################
###################################### Groups and models ######################################
###############################################################################################
group_dict = {
    "CN_no_WMH": CN_no_WMH,
    # "MCI_no_WMH": MCI_no_WMH,
    # "CN_WMH": CN_WMH,
    # "MCI_WMH": MCI_WMH,
}
model_types_per_group = {
    "CN_no_WMH": ["homogeneous_G"],
    # "MCI_no_WMH": ["homogeneous_a"],
    # "CN_WMH": ["homogeneous_a", "homogeneous_G"], 
    # "MCI_WMH": ["homogeneous_a","homogeneous_G"], 
}

if __name__ == "__main__":
    run_group_level_simulations(group_dict, model_types_per_group)