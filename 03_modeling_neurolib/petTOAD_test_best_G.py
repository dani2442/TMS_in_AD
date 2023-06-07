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
import filteredPowerSpectralDensity as filtPowSpectr
from neurolib.models.pheno_hopf import PhenoHopfModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils import paths
from petTOAD_setup import *

group_list = ["MCI_noWMH", "HC_WMH", "MCI_WMH"]


# %% Define functions
# Define the evaluation function for the model
def evaluate(traj):
    model = search.getModelFromTraj(traj)
    bold_list = []

    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 180 secs to "warm up" the timeseries
    ts = model.outputs.x[:, 18000::300]
    t = model.outputs.t[18000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    bold_list.append(ts_filt)

    result_dict = {}
    result_dict["BOLD"] = np.array(bold_list).squeeze()
    result_dict["t"] = t

    search.saveToPypet(result_dict, traj)


def get_group_freqs(group_name):

    # Get the timeseries for the chosen group
    _, timeseries = get_group_ts_for_freqs(group_name, all_fMRI_clean)
    n_subjs, n_nodes, _ = timeseries.shape

    # Get the frequencies (narrow bandwidth)
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR)
    f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])

    return f_diff, n_subjs, n_nodes


def main():
    global group_data, search
    group_data = {}
    for group_name in group_list:
        print(
            f"Now simulating group {group_name} with the best G found in HC without WMH..."
        )
        # Create the results dir for the group
        SIM_DIR_GROUP = SIM_DIR / group_name
        if not Path.exists(SIM_DIR_GROUP):
            Path.mkdir(SIM_DIR_GROUP)

        # Set the directory where to save results
        paths.HDF_DIR = str(SIM_DIR_GROUP)
        f_diff, n_subjs, n_nodes = get_group_freqs(
            group_name
        )
        filename = f"{group_name}_with_best_G"
        group_data[group_name] = SIM_DIR_GROUP / filename
        n_sim = n_subjs
        
        model.params["w"] = 2 * np.pi * f_diff
        # Define the group-specific simulations
        if group_name == "MCI_noWMH":
            parameters = ParameterSpace(
                {
                    "a": a_MCI,
                    "K_gl": K_gl,
                },
                kind="grid",
            )
        else:
            parameters = ParameterSpace(
                {"a": a_WMH, "K_gl": K_gl}, kind="grid"
            )
    
        for i in range(n_sim):
            print(f"Starting with simulation n°: {i+1}/{n_sim}")
            # Initialize the search
            search = BoxSearch(
                model=model,
                evalFunction=evaluate,
                parameterSpace=parameters,
                filename=filename,
            )
            search.run(chunkwise=True, chunksize=60000, append=True)
    group_dict = pd.DataFrame.from_dict(group_data, orient= "index")    
    group_dict.to_csv(RES_DIR / "group_names_for_best_G.csv")

# %%
# Now setup the model
# Set if the model has delay
delay = False
if not delay:
    Dmat_dummy = np.zeros_like(sc)
    Dmat = Dmat_dummy
else:
    pass
# Initialize the model (neurolib wants a Dmat to initialize the mode,
# so we gave it an empty Dmat, which we also later cancel by setting it to None)
model = PhenoHopfModel(Cmat=sc, Dmat=Dmat)
model.params["Dmat"] = None if not delay else Dmat
# Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries
model.params["duration"] = 12.65 * 60 * 1000
model.params["signalV"] = 0
model.params["dt"] = 0.1
model.params["sampling_dt"] = 10.0
model.params["sigma"] = 0.02


a_MCI = [np.ones(n_nodes) * a for a in np.round(np.arange(-0.1, 0.1, 0.001), 3)]
a_WMH = [np.ones(n_nodes) * -0.02]
K_gl = [1.9]


if __name__ == "__main__":
    main()
# %%
