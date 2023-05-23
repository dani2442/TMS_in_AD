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
import petTOAD_neurolib_simulations as sim
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
    for traj in trajs:
        search.loadResults(trajectoryName=traj)
        bold = search.dfResults["BOLD"]
        list_bold.append(bold)
    return np.array(list_bold)


def calculate_results_from_bolds(bold_arr):
    # Create a new array to store the FC, FCD and phFCD values with the same shape as bold array
    fc_array = np.empty_like(bold_arr)
    fcd_array = np.empty_like(bold_arr)
    phfcd_array = np.empty_like(bold_arr)

    # Iterate over each element in the bold array
    for i in range(nparms):
        for j in range(nsim):
            # Get the current timeseries
            timeseries = bold_arr[i, j].squeeze()

            # Perform FC, FCD and phFCD analysis
            fc_value = func.fc(timeseries)
            fcd_value = func.fcd(timeseries)
            triu_ind_fcd = np.triu_indices(fcd_value.shape[0], k=1)
            fcd_vals = fcd_value[triu_ind_fcd]
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


fc, fcd, phfcd = my_func.calc_and_save_group_stats(
    sim.all_HC_fMRI_clean, sim.SIM_DIR_GROUP
)
#%%
trajs = pu.getTrajectorynamesInFile(f"{sim.paths.HDF_DIR}/{sim.filename}")
nsim = len(trajs)
nparms = len(sim.parameters.K_gl)
# Better to work nparms x nsims for later storage in pandas
bold_arr = get_bolds_from_trajs(trajs).T
fc_array, fcd_array, phfcd_array = calculate_results_from_bolds(bold_arr)
sim_fc = fc_array.mean(axis=1)
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
plot_results_exploration_G()
print("Done!")
print(f"Total run time for the script: {round(real_end_time - start_time, 3) / 60} mins")

# %%
