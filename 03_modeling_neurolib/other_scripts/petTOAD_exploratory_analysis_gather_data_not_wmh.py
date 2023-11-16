# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurolib.utils.functions as func
import petTOAD_exploratory_analysis_not_wmh as sim
import my_functions as my_func
import seaborn as sns
from neurolib.utils import pypetUtils as pu

#%%
def calculate_results_from_bolds(bold_arr, nsim, nparms):
    # Create a new array to store the FC and phFCD values with the same shape as bold array
    fc_array = np.zeros([nsim, nparms, 90, 90])
    phfcd_array = np.zeros([nsim, nparms, 18145])

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
                # Store the FC, FCD, phFCD value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
                phfcd_array[i, j] = phfcd_value
    return fc_array, phfcd_array 


def save_plot_results(res_df):
    table_fc = pd.pivot_table(res_df, values='fc_pearson', index='a', columns='G')
    # table_fcd = pd.pivot_table(res_df, values='fcd_ks', index='b', columns='w')
    table_phfcd = pd.pivot_table(res_df, values='phfcd_ks', index='a', columns='G')
    plt.figure(figsize = (6,18))
    plt.subplot(211)
    sns.heatmap(table_fc.astype(float))
    plt.title("FC")
    # plt.subplot(312)
    # sns.heatmap(table_fcd.astype(float))
    # plt.title("FCD")
    plt.subplot(212)
    sns.heatmap(table_phfcd.astype(float))
    plt.title("phFCD")
    plt.savefig(sim.EXPL_DIR / f"sub-{subj}_results_heatmap.png")



#%%
def calculate_results_stats(group):
    filename = f"{sim.paths.HDF_DIR}/homogeneous_model_not_WMH-weight_{group}.hdf"
    trajs = pu.getTrajectorynamesInFile(f"{sim.paths.HDF_DIR}/homogeneous_model_not_WMH-weight_{group}.hdf")
    nsim = 5
    nparms = 101*21
    big_list = []
    for n_traj, traj in enumerate(trajs[:5]):
        print(f"Now loading trajectory {n_traj + 1}/{len(trajs)}...")
        traj_list = []
        tr = pu.loadPypetTrajectory(filename, traj)
        run_names = tr.f_get_run_names()
        n_run = len(run_names)
        ns = range(n_run)
        for i in ns:
            r = pu.getRun(i, tr)
            traj_list.append(r['BOLD'])
        big_list.append(traj_list) 
    bold_arr = np.array(big_list)
    fc_array, phfcd_array = calculate_results_from_bolds(bold_arr, nsim, nparms)
    fc_array_mean = fc_array.mean(axis = 0)
    phfcd_array_mean = phfcd_array.mean(axis = 0)
    return fc_array_mean, phfcd_array_mean

fc_array_hc, phfcd_array_hc = calculate_results_stats("HC")
fc_array_mci, phfcd_array_mci = calculate_results_stats("MCI")

#%%
for subj_n, subj in enumerate(short_subjs):
    if subj in sim.HC:
        fc_array = fc_array_hc
        phfcd_array = phfcd_array_hc
    elif subj in sim.MCI:
        fc_array = fc_array_mci
        phfcd_array = phfcd_array_mci

    print(f"Now processing subject {subj} ({subj_n + 1} / {len(short_subjs)})")
    timeseries = sim.all_fMRI_clean[subj]
    fc = func.fc(timeseries)
    print("Calculating phFCD")
    phfcd = my_func.phFCD(timeseries)
    # Get the average fc across the n simulations
    print("Calculating fcs correlations...")
    fc_pearson = [func.matrix_correlation(row_fc, fc) for row_fc in fc_array]
    print("Calculating phFCDs...")
    phfcd_ks = [my_func.matrix_kolmogorov(phfcd, sim_phfcd) for sim_phfcd in phfcd_array]
    phfcd_ks_arr = np.array(phfcd_ks)
    res_df = pd.DataFrame([(round(a,3), round(G,3)) for a in np.linspace(-0.08, 0.02, 101) for G in np.linspace(0.5, 2.5, 21)])
    res_df["fc_pearson"] = fc_pearson
    res_df["phfcd_ks"] = phfcd_ks
    res_df.columns = ["a", "G", "fc_pearson", "phfcd_ks"]
    res_df.to_csv(sim.EXPL_DIR / f"sub-{subj}_df_results_initial_exploration_not_wmh.csv")
    save_plot_results(res_df)
# %%
