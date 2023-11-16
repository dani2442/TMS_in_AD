
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
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurolib.utils.functions as func
import petTOAD_exploratory_analysis_WMH_groups_G as sim
import my_functions as my_func
import seaborn as sns
from neurolib.utils import pypetUtils as pu
from neurolib.optimize.exploration import BoxSearch

#%%
def get_bolds_from_trajs(trajs):
    list_bold = []
    search = BoxSearch(
        model=sim.model,
        evalFunction=sim.evaluate,
        parameterSpace=sim.parameters,
        filename=sim.filename,
    )
    for n, traj in enumerate(trajs):
        print(f"Loading trajectory {n} out of {len(trajs)}")
        search.loadResults(trajectoryName=traj)
        bold = search.dfResults["BOLD"]
        list_bold.append(bold)
    return np.array(list_bold)


def calculate_results_from_bolds(bold_arr):
    # Create a new array to store the FC, FCD and phFCD values with the same shape as bold array
    fc_array = np.zeros([nsim, nparms, 90, 90])
    #fcd_array = np.zeros([nsim, nparms, 528])
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
                # print("Calculating fcd..")
                # fcd_value = func.fcd(timeseries)
                # triu_ind_fcd = np.triu_indices(fcd_value.shape[0], k=1)
                # fcd_vals = fcd_value[triu_ind_fcd]
                print("Calculating phFCD")
                phfcd_value = my_func.phFCD(timeseries)
                # Store the FC, FCD, phFCD value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
                # fcd_array[i, j] = fcd_vals
                phfcd_array[i, j] = phfcd_value
    return fc_array, phfcd_array #, phfcd_array


def save_plot_results(res_df):
    table_fc = pd.pivot_table(res_df, values='fc_pearson', index='b', columns='w')
    # table_fcd = pd.pivot_table(res_df, values='fcd_ks', index='b', columns='w')
    table_phfcd = pd.pivot_table(res_df, values='phfcd_ks', index='b', columns='w')
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



condition = sys.argv[1].lower()

# Mapping dictionary
mapping = {
    "true": True,
    "false": False
}

random_condition = mapping.get(condition)

#%%

for subj_n, subj in enumerate(sim.subjs_to_sim[:2]):
    print(f"Now processing subject {subj} ({subj_n + 1} / {len(sim.subjs_to_sim)})")
    if not random_condition:
        filename = f"{sim.paths.HDF_DIR}/{subj}_homogeneous_G-weight_model.hdf"
        trajs = pu.getTrajectorynamesInFile(f"{sim.paths.HDF_DIR}/{subj}_homogeneous_G-weight_model.hdf")
    else:
        filename = f"{sim.paths.HDF_DIR}/{subj}_homogeneous_G-weight_model_random.hdf"
        trajs = pu.getTrajectorynamesInFile(f"{sim.paths.HDF_DIR}/{subj}_homogeneous_G-weight_model_random.hdf")  
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
    bold_arr = np.array(big_list)
    nsim = len(trajs)
    nparms = len([1.9 * w * sim.wmh_dict[subj] + b for w in sim.ws for b in sim.bs])
    fc_array, phfcd_array = calculate_results_from_bolds(bold_arr)
    timeseries = sim.all_fMRI_clean[subj]
    fc = func.fc(timeseries)
    # print("Calculating fcd..")
    # fcd = func.fcd(timeseries)
    # triu_ind_fcd = np.triu_indices(fcd.shape[0], k=1)
    # fcd = fcd[triu_ind_fcd]
    print("Calculating phFCD")
    phfcd = my_func.phFCD(timeseries)
    # Get the average fc across the n simulations
    sim_fc = fc_array.mean(axis=0)
    print("Calculating fcs correlations...")
    fc_pearson = [func.matrix_correlation(row_fc, fc) for row_fc in sim_fc]
    # print("Calculating FCDs...")
    # fcd_ks = []
    # for row in fcd_array:
    #     row_ks = [my_func.matrix_kolmogorov(fcd, sim_fcd) for sim_fcd in row]
    #     fcd_ks.append(row_ks)
    #     fcd_ks_arr = np.array(fcd_ks)
    # fcd_ks = fcd_ks_arr.mean(axis=0)
    print("Calculating phFCDs...")
    phfcd_ks = []
    for row in phfcd_array:
        row_phfcd_ks = [my_func.matrix_kolmogorov(phfcd, sim_phfcd) for sim_phfcd in row]
        phfcd_ks.append(row_phfcd_ks)
        phfcd_ks_arr = np.array(phfcd_ks)
    phfcd_ks = phfcd_ks_arr.mean(axis=0)

    data = [[[(round(b,3), round(w,3)) for w in sim.ws for b in sim.bs], fc_pearson, phfcd_ks]]
    columns = ["b_w", "fc_pearson", "phfcd_ks"]
    res_df = pd.DataFrame(data, columns=columns).explode(columns)
    res_df['b'], res_df['w'] = zip(*res_df.b_w)
    res_df = res_df.drop(columns=['b_w'])
    res_df.to_csv(sim.EXPL_DIR / f"sub-{subj}_df_results_initial_exploration_wmh_G.csv")
    save_plot_results(res_df)
# %%
