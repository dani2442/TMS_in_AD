#%%
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
import neurolib.utils.functions as func
import petTOAD_neurolib_simulations as sim
from neurolib.utils import pypetUtils as pu
#%%
from neurolib.optimize.exploration import BoxSearch


#%%%

# df = search.dfResults
# df["mean_fc_corr"] = df["fc_corr"].apply(lambda x: np.mean(x))
# df["std_fc_corr"] = df["fc_corr"].apply(lambda x: np.std(x))
# plot_and_save_exploration(df)

# #%%
# res_df = pd.DataFrame(columns=
#     [
#         "K_gl",
#         "fc_corr_mean",
#         "fc_corr_std",
#         "fcd_ks_mean",
#         "fcd_ks_std",
#         "phfcd_ks_mean",
#         "phfcd_ks_std",
#     ]
# )
# fc_corrs = []
# fcds_all = []
# fcd_ks = []
# list_bold = []

#res_df['K_gl'] = search.parameterSpace.K_gl
#%%
trajs = pu.getTrajectorynamesInFile(f'{sim.paths.HDF_DIR}/{sim.filename}')
list_bold = []
search = BoxSearch(
            model=sim.model,
            evalFunction=sim.evaluate,
            parameterSpace=sim.parameters,
            filename=sim.filename,
        )
for traj in trajs:
    search.loadResults(trajectoryName=traj)
    bold = search.dfResults['BOLD']
    list_bold.append(bold)
bold_arr = np.array(list_bold)
# Create a new array to store the FCD values with the same shape as timeseries_array
fc_array = np.empty_like(bold_arr)
fcd_array = np.empty_like(bold_arr)
phfcd_array = np.empty_like(bold_arr)

# Iterate over each element in the timeseries_array
for i in range(bold_arr.shape[0]):
    for j in range(bold_arr.shape[1]):
        # Get the current timeseries
        timeseries = bold_arr[i, j]
        
        # Perform FCD analysis using func.fcd function (replace with the actual function call)
        fc_value = func.fc(timeseries)
        fcd_value = func.fcd(timeseries)
        phfcd_value = my_func.phFCD(timeseries)
        # Store the FCD value in the corresponding position in the fcd_array
        fc_array[i, j] = fc_value
        fcd_array[i, j] = fcd_value
        phfcd_array[i, j] = phfcd_value
