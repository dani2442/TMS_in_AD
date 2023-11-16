# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Gather results of the repeated simulations   -- Version 1.0
Last edit:  2023/05/16
Authors:    Leone, Riccardo (RL)
Notes:      - Gather results of model simulation of the phenomenological Hopf model with Neurolib in MCI no WMH
            - Release notes:
                * Initial release
To do:      - 
Comments:   

Sources: 
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurolib.utils.functions as func
import petTOAD_exploratory_analysis_WMH_groups as sim
import my_functions as my_func
import seaborn as sns
from neurolib.utils import pypetUtils as pu
from neurolib.optimize.exploration import BoxSearch

#%%

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


#%%
filedir = sim.RES_DIR / "model_simulations" / "MCI_noWMH"
filename = filedir / "MCI_noWMH_with_best_G.hdf"
trajs = pu.getTrajectorynamesInFile(filename)
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
nparms = len([np.ones(90) * a for a in np.round(np.arange(-0.15, 0.05, 0.025), 3)])
fc_array, phfcd_array = calculate_results_from_bolds(bold_arr)

#%%

fc, fcd, phfcd = my_func.calc_and_save_group_stats(sim.MCI_no_WMH, filedir)
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
#%%
res_dict = {'a': np.round(np.arange(-0.15, 0.05, 0.025),3),
            'fc_pearson': fc_pearson,
            'phfcd_ks': phfcd_ks}
res_df = pd.DataFrame.from_dict(res_dict)
res_df.to_csv(filedir / f"MCI_df_results_initial_exploration_wmh.csv")