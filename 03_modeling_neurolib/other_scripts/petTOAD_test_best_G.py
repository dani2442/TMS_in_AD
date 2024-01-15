#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Test the best G on other groups   -- Version 1.0
Last edit:  2023/05/26
Authors:    Leone, Riccardo (RL)
Notes:      - Test the best G on other groups
            - Release notes:
                * Initial commit
To do:      - 
Comments:   

Sources: 
"""
# %% Initial imports
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from neurolib.utils.parameterSpace import ParameterSpace

import filteredPowerSpectralDensity as filtPowSpectr
from petTOAD_setup import *


# %% Define functions
# Define the evaluation function for the model
def evaluate(traj):
    model = search.getModelFromTraj(traj)
    bold_list = []

    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 120 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 12000::300]
    t = model.outputs.t[12000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    bold_list.append(ts_filt)

    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list).squeeze()
    result_dict["t"] = t

    search.saveToPypet(result_dict, traj)


# Choose the group on which to perform analyses ("CN_no_ WMH", "CN_WMH", "MCI_noWMH", "MCI_WMH")
group_name = 'MCI_no_WMH'
group_list = MCI_no_WMH

# Create the results dir for the group
SIM_DIR_GROUP = SIM_DIR / group_name
if not Path.exists(SIM_DIR_GROUP):
    Path.mkdir(SIM_DIR_GROUP)

# Set the directory where to save results
paths.HDF_DIR = str(SIM_DIR_GROUP)

# Get the timeseries for the chosen group
group, timeseries = get_group_ts_for_freqs(group_list, all_fMRI_clean)
# Get the mean frequencies (narrow bandwidth) for the chosen group and set the model frequencies
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR)
f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])

delay = False
if not delay:
    Dmat_dummy = np.zeros_like(sc)
    Dmat = Dmat_dummy
else:
    pass

# Initialize the model (neurolib wants a Dmat to initialize the mode,
# so we gave it an empty Dmat, which we also later cancel by setting it to None after the initialization of the model)
model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
# Empirical fmri is 193 timepoints with a TR=3s --> 9.65 min. We simulate 9.65 min + 2 min of initial warm up of the timeseries --> 11.65
model.params["duration"] = 11.65 * 60 * 1000
model.params["signalV"] = 0
model.params["dt"] = 0.1
model.params["sampling_dt"] = 10.0
model.params["sigma"] = 0.02
model.params["w"] = 2 * np.pi * f_diff
model.params["K_gl"] = 2.0
a_mci_no_wmh = [np.ones(n_nodes) * a for a in np.round(np.linspace(-0.1, 0.0, 101), 3)]
# Define the parameters space to explore for G (K_gl)
parameters = ParameterSpace(
    {"a": a_mci_no_wmh}, kind="grid"
)

#%%
filename = "testing_best_G.hdf"
if __name__ == "__main__":
    n_sim = len(group)
    for i in range(n_sim):
        print(f"Now performing simulation number {i+1}/{n_sim}...")
        # Initialize the search
        search = BoxSearch(
            model=model,
            evalFunction=evaluate,
            parameterSpace=parameters,
            filename=filename,
        )
        search.run(chunkwise=True, chunksize=60000, append=True)
    print(f"Done with simulations for {group}")