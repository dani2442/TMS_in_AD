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
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurolib.utils.functions as func
import petTOAD_exploratory_analysis_WMH_groups as sim
import my_functions as my_func
from neurolib.utils import pypetUtils as pu
from neurolib.optimize.exploration import BoxSearch


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


def plot_results_exploration_G():
    plt.figure()
    plt.plot(res_df["K_gl"], res_df["fc_pearson"], label="FC")
    plt.plot(res_df["K_gl"], res_df["fcd_ks"], label="FCD")
    plt.plot(res_df["K_gl"], res_df["phfcd_ks"], label="phFCD")
    plt.xlabel("Coupling parameter (G)")
    plt.ylabel(r"Pearson's $\rho$ / KS-distance")
    plt.legend()
    plt.savefig(sim.SIM_DIR_GROUP / f"{sim.filename}_plot.png")



#%%

subj = sim.short_subjs[1]
filename = f"{sim.paths.HDF_DIR}/{subj}_homogeneous_model.hdf"
trajs = pu.getTrajectorynamesInFile(f"{sim.paths.HDF_DIR}/{subj}_homogeneous_model.hdf")
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
nparms = len([(np.ones(90) * -0.02) * w * sim.wmh_dict[subj] + b for w in sim.ws for b in sim.bs])
fc_array, fcd_array, phfcd_array = calculate_results_from_bolds(bold_arr)
#%%
timeseries = sim.all_fMRI_clean[subj]
fc = func.fc(timeseries)
print("Calculating fcd..")
fcd = func.fcd(timeseries)
triu_ind_fcd = np.triu_indices(fcd.shape[0], k=1)
fcd = fcd[triu_ind_fcd]
print("Calculating phFCD")
phfcd = my_func.phFCD(timeseries)

sim_fc = fc_array.mean(axis=0)
print("Calculating fcs...")
start_time = time.time()
fc_pearson = [func.matrix_correlation(row_fc, fc) for row_fc in sim_fc]
end_time = time.time()
print(f"It took {round(end_time-start_time, 3)/ 60} mins to process FCs")
print("Calculating FCDs...")
fcd_ks = [my_func.matrix_kolmogorov(fcd, np.concatenate(row)) for row in fcd_array]
end_end_time = time.time()
print(f"It took {round(end_end_time - end_time, 3) / 60} mins to process FCDs")
print("Calculating phFCDs...")
phfcd_ks = [
    my_func.matrix_kolmogorov(phfcd, np.concatenate(row)) for row in phfcd_array
]
real_end_time = time.time()
print(f"It took {round(real_end_time - end_end_time, 3) / 60} mins to process phFCDs")
data = [[sim.parameters.K_gl, fc_pearson, fcd_ks, phfcd_ks]]
columns = ["K_gl", "fc_pearson", "fcd_ks", "phfcd_ks"]
res_df = pd.DataFrame(data, columns=columns).explode(columns)
res_df.to_csv(sim.SIM_DIR_GROUP / f"df_results_{sim.filename}.csv")
plot_results_exploration_G()
print("Done!")
print(f"Total run time for the script: {round(real_end_time - start_time, 3) / 60} mins")

# %%
