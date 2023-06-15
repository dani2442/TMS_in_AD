# %% # -*- coding: utf-8 -*-
"""   Gather results of the repeated simulations   -- Version 1.0
Last edit:  2023/05/16
Authors:    Leone, Riccardo (RL)
Notes:      - Gather results of model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Initial release
To do:      - Change the code so that it doesn't re-calculate powSpectra and so on
Comments:   

Sources: 
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurolib.utils.functions as func
import my_functions as my_func
import seaborn as sns
from neurolib.utils import pypetUtils as pu
from petTOAD_setup import *

#%%
def calculate_results_from_bolds(bold_arr):
    # Create a new array to store the FC, FCD and phFCD values with the same shape as bold array
    fc_array = np.zeros([nsim, nparms, 90, 90])
    fcd_array = np.zeros([nsim, nparms, 528])
    phfcd_array = np.zeros([nsim, nparms, 18145])

    # Iterate over each element in the bold array
    for i in range(nsim):
        for j in range(nparms):
            print(f"Now calculating results from the {i} simulation for parameter {j}...")
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
                print("Calculating fcd..")
                fcd_value = func.fcd(timeseries)
                triu_ind_fcd = np.triu_indices(fcd_value.shape[0], k=1)
                fcd_vals = fcd_value[triu_ind_fcd]
                print("Calculating phFCD")
                phfcd_value = my_func.phFCD(timeseries)
                # Store the FC, FCD, phFCD value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
                fcd_array[i, j] = fcd_vals
                phfcd_array[i, j] = phfcd_value
    return fc_array, fcd_array, phfcd_array


def save_plot_results(res_df):
    table_fc = pd.pivot_table(res_df, values='fc_pearson', index='b', columns='w')
    table_fcd = pd.pivot_table(res_df, values='fcd_ks', index='b', columns='w')
    table_phfcd = pd.pivot_table(res_df, values='phfcd_ks', index='b', columns='w')
    plt.figure(figsize = (6,24))
    plt.subplot(311)
    sns.heatmap(table_fc.astype(float))
    plt.title("FC")
    plt.subplot(312)
    sns.heatmap(table_fcd.astype(float))
    plt.title("FCD")
    plt.subplot(313)
    sns.heatmap(table_phfcd.astype(float))
    plt.title("phFCD")
    plt.savefig(sim.EXPL_DIR / f"sub-{subj}_results_heatmap.png")



#%%
group_names_dict = pd.read_csv(RES_DIR / "group_names_for_best_G.csv", index_col=0).to_dict()['0']
# I am running on windows, so overwrite for now..
group_names_dict = {'MCI_noWMH': 'U:\\petTOAD\\results\\model_simulations\\MCI_noWMH\\MCI_noWMH_with_best_G.hdf',
                    'HC_WMH': 'U:\petTOAD\\results\\model_simulations\\HC_WMH\\HC_WMH_with_best_G.hdf',
                    'MCI_WMH': 'U:\petTOAD\\results\\model_simulations\\MCI_WMH\\MCI_WMH_with_best_G.hdf'}
#%%

print("Now processing group MCI no WMH...")
trajs = pu.getTrajectorynamesInFile(group_names_dict['MCI_noWMH'])
nparms = 8
big_list = []
for traj in trajs:
    traj_list = []
    tr = pu.loadPypetTrajectory(group_names_dict['MCI_noWMH'], traj)
    run_names = tr.f_get_run_names()
    n_run = len(run_names)
    ns = range(n_run)
    for i in ns:
        r = pu.getRun(i, tr)
        traj_list.append(r['BOLD'])

    big_list.append(traj_list) 
bold_arr = np.array(big_list)
nsim = len(trajs)
fc_array, fcd_array, phfcd_array = calculate_results_from_bolds(bold_arr)


#%%
EXPL_DIR = RES_DIR / "exploratory"
list_group = ['HC_WMH', 'MCI_WMH']
for group_name in list_group:
    print(f"Now processing group {group_name}...")
    # Get the names of the trajectories in the file stored with neurolib
    trajs = pu.getTrajectorynamesInFile(group_names_dict[group_name])
    # Create a big list to accumulate results from all the trajectories
    big_list = []
    # Loop through each trajectory
    for traj in trajs:
        # Create list to accumulate results of this trajectory
        traj_list = []
        # Load the full trajectory comprising many runs
        tr = pu.loadPypetTrajectory(group_names_dict[group_name], traj)
        # Get the list of run names
        run_names = tr.f_get_run_names()
        # Get the number of total runs
        n_run = len(run_names)
        # Although the trajectories are called "run_000001", pu.getRun works with just the integere (e.g., 1)
        ns = range(n_run)
        # Loop through all the results and accumulate them into the list.
        for i in ns:
            r = pu.getRun(i, tr)
            traj_list.append(r['BOLD'])
        # Append this trajectory results to the big list of all trajectories
        big_list.append(traj_list) 
    # Conver to array and squeeze
    bold_arr = np.array(big_list).squeeze()
    group, timeseries = get_group_ts_for_freqs(group_name, all_fMRI_clean)
    # Calculate group results
    fc, fcd, phfcd = my_func.calc_and_save_group_stats(group, EXPL_DIR / group_name)
    # Starts with simulated vs. empirical comparisons.
    print("Calculating FCs comparison")
    fc_pearson = [func.matrix_correlation(func.fc(row_fc), fc) for row_fc in bold_arr]
    print("Calculating FCDs KS...")
    fcd_ks = [my_func.matrix_kolmogorov(fcd, np.concatenate(my_func.fcd(row))) for row in fcd_array]
    print("Calculating phFCDs KS...")
    phfcd_ks = [
        my_func.matrix_kolmogorov(phfcd, np.concatenate(func.phfcd(row))) for row in phfcd_array
    ]
    res_df_group = pd.DataFrame([fc_pearson, fcd_ks, phfcd_ks], columns=['fc_pearson', 'fcd_ks', 'phfcd_ks'])
    res_df_group.to_csv(EXPL_DIR / group_name / f"df_results_{group_name}.csv")
    











# #%%
# fc = func.fc(timeseries)
# print("Calculating fcd..")
# fcd = func.fcd(timeseries)
# triu_ind_fcd = np.triu_indices(fcd.shape[0], k=1)
# fcd = fcd[triu_ind_fcd]
# print("Calculating phFCD")
# phfcd = my_func.phFCD(timeseries)
# # Get the average fc across the n simulations
# sim_fc = fc_array.mean(axis=0)
# print("Calculating fcs correlations...")
# fc_pearson = [func.matrix_correlation(row_fc, fc) for row_fc in sim_fc]
# print("Calculating FCDs...")
# fcd_ks = []
# for row in fcd_array:
#     row_ks = [my_func.matrix_kolmogorov(fcd, sim_fcd) for sim_fcd in row]
#     fcd_ks.append(row_ks)
#     fcd_ks_arr = np.array(fcd_ks)
# fcd_ks = fcd_ks_arr.mean(axis=0)
# print("Calculating phFCDs...")
# phfcd_ks = []
# for row in fcd_array:
#     row_phfcd_ks = [my_func.matrix_kolmogorov(phfcd, sim_phfcd) for sim_phfcd in row]
#     phfcd_ks.append(row_ks)
#     phfcd_ks_arr = np.array(phfcd_ks)
# phfcd_ks = phfcd_ks_arr.mean(axis=0)

#     data = [[[(round(b,3), round(w,3)) for w in sim.ws for b in sim.bs], fc_pearson, fcd_ks, phfcd_ks]]
#     columns = ["b_w", "fc_pearson", "fcd_ks", "phfcd_ks"]
#     res_df = pd.DataFrame(data, columns=columns).explode(columns)
#     res_df['b'], res_df['w'] = zip(*res_df.b_w)
#     res_df = res_df.drop(columns=['b_w'])
#     res_df.to_csv(sim.EXPL_DIR / f"sub-{subj}_df_results_initial_exploration_wmh.csv")
#     save_plot_results(res_df)