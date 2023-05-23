#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 2.2
Last edit:  2023/05/11
Authors:    Leone, Riccardo (RL)
Notes:      - Model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Added plot labels
To do:      - 
Comments:   

Sources: 
"""
# %% Initial imports
import matplotlib.pyplot as plt
import neurolib.utils.functions as func
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from petTOAD_setup import *

# Create a figure dir
FIG_DIR = RES_DIR / "Figures"
if not Path.exists(FIG_DIR):
    Path.mkdir(FIG_DIR)


# Set the directory where to save results
paths.HDF_DIR = str(RES_DIR / "neurolib")


# %% Define functions
def calc_and_save_fc(ts_dict):
    """
    Calculate the empirical FC matrix out of bandpass-filtered timeseries.
    Args:

    Returns:
    """

    fcs = []
    for ts in ts_dict.values():
        fcs.append(func.fc(ts))
    avg_fc = np.array(fcs).mean(axis=0)
    np.savetxt(RES_DIR / "average_FC_HC.csv", avg_fc, delimiter=",")
    return avg_fc


# Define the evaluation function
def evaluate(traj):
    model = search.getModelFromTraj(traj)
    bold_list = []
    sFC_list = []
    fc_corr_list = []

    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 180 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 18000::300]
    t = model.outputs.t[18000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    sFC = func.fc(ts_filt)
    fc_corr = func.matrix_correlation(sFC, avg_fc)
    bold_list.append(ts_filt)
    sFC_list.append(sFC)
    fc_corr_list.append(fc_corr)

    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list)
    result_dict["FC"] = np.array(sFC_list)
    result_dict["t"] = t
    result_dict["fc_corr"] = np.array(fc_corr_list)

    search.saveToPypet(result_dict, traj)

def plot_and_save_exploration(df):
    plt.figure()
    plt.plot(df['K_gl'], df['mean_fc_corr'])
    plt.fill_between(df['K_gl'], 
    (df['mean_fc_corr'] + 1.96 * df['std_fc_corr']),
    (df['mean_fc_corr'] - 1.96 * df['std_fc_corr']),
    alpha = 0.1
    )
    plt.ylabel("sFC-eFC correlation")
    plt.xlabel("G")
    plt.savefig(FIG_DIR / "initial_exploration_Gs.png")

# Choose the group on which to perform analyses
group = all_HC_fMRI_clean
group_name = "HC"

avg_fc = calc_and_save_fc(all_HC_fMRI_clean)

# Set if the model has delay
delay = False
if not delay:
    Dmat_dummy = np.zeros_like(sc)
    Dmat = Dmat_dummy
else:
    pass

# Initialize the model (neurolib wants a Dmat to initialize the mode,
# so we gave it an empty Dmat, which we also later cancel by setting it to None)
model = PhenoHopfModel(Cmat=sc, Dmat=Dmat_dummy)
if not delay:
    model.params["Dmat"] = None
else:
    pass
# Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
model.params["duration"] = 12.65 * 60 * 1000
model.params["signalV"] = 0
model.params["w"] = 2 * np.pi * f_diff
model.params["dt"] = 0.1
model.params["sampling_dt"] = 10.0
model.params["sigma"] = 0.02
model.params["a"] = np.ones(90) * (-0.02)

# Define the parametere space to explore
parameters = ParameterSpace(
    {"K_gl": np.round(np.linspace(0.0, 6.0, 288), 3)}, kind="grid"
)


# %% Run the parameter Search and save results
for _ in range(50):
    # Initialize the search
    search = BoxSearch(
        model=model,
        evalFunction=evaluate,
        parameterSpace=parameters,
        filename="initial_exploration_Gs.hdf",
    )

    search.run(chunkwise=True, chunksize=60000, append=True)

#search.loadResults()
#%%
#df = search.dfResults
#df['mean_fc_corr'] = df['fc_corr'].apply(lambda x: np.mean(x))
#df['std_fc_corr'] = df['fc_corr'].apply(lambda x: np.std(x))
#plot_and_save_exploration(df)