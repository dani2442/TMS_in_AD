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
import petTOAD_find_best_G as sim
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
    fc_array = np.empty_like(bold_arr)
    phfcd_array = np.empty_like(bold_arr)

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


def plot_results_exploration_G():
    plt.figure()
    plt.plot(res_df["K_gl"], res_df["fc_pearson"], label="FC")
    plt.plot(res_df["K_gl"], res_df["phfcd_ks"], label="phFCD")
    plt.xlabel("Coupling parameter (G)")
    plt.ylabel(r"Pearson's $\rho$ / KS-distance")
    plt.legend()
    plt.savefig(sim.SIM_DIR_GROUP / f"{sim.filename}_plot.png")


fc, _, phfcd = my_func.calc_and_save_group_stats(
    sim.group, sim.SIM_DIR_GROUP
)
#%%
trajs = pu.getTrajectorynamesInFile(f"{sim.paths.HDF_DIR}/{sim.filename}")
nsim = len(trajs)
nparms = len(sim.parameters.K_gl)
# Better to work nparms x nsims for later storage in pandas
bold_arr = get_bolds_from_trajs(trajs).T
fc_array, phfcd_array = calculate_results_from_bolds(bold_arr)
sim_fc = fc_array.mean(axis=1)
#%%
print("Calculating fcs...")
fc_pearson = [func.matrix_correlation(row_fc, fc) for row_fc in sim_fc]
print("Calculating phFCDs...")
phfcd_ks = [
    my_func.matrix_kolmogorov(phfcd, np.concatenate(row)) for row in phfcd_array
]
data = [[sim.parameters.K_gl, fc_pearson, phfcd_ks]]
columns = ["K_gl", "fc_pearson", "phfcd_ks"]
res_df = pd.DataFrame(data, columns=columns).explode(columns)
res_df.to_csv(sim.SIM_DIR_GROUP / f"df_results_{sim.filename}.csv")
plot_results_exploration_G()
print("Done!")

# %%
