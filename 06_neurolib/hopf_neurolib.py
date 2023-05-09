# %%
# All neurolib imports
import neurolib.utils.functions as func

# import neurolib.utils.pypetUtils as pu
# import neurolib.utils.paths as paths
# import neurolib.optimize.exploration.explorationUtils as eu
from neurolib.models.hopf import HopfModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch

# Other imports
import matplotlib.pyplot as plt
from petTOAD_setup import *

plt.rcParams["image.cmap"] = "plasma"

# Set if the model has delay
delay = False
if not delay:
    Dmat_dummy = np.zeros_like(sc)
    Dmat = Dmat_dummy
else:
    pass

# %% Initialize the model
model = HopfModel(Cmat=sc_norm, Dmat=Dmat)
model.params["Dmat"] = None
model.params["duration"] = 0.2 * 60 * 1000
model.params["signalV"] = 0
model.params["coupling"] = "diffusive"
model.params["sampling_dt"] = 1000.0
model.params["w"] = 2 * np.pi * f_diff
model.params["dt"] = 0.1


def evaluate(traj):
    ts_list = []
    for i in range(2):
        model = search.getModelFromTraj(traj)
        model.randomICs()
        model.run(chunkwise=True, chunksize=60000, append=True)
        ts = clean(
            model.outputs.x,
            detrend=True,
            standardize="zscore",
            filter="butterworth",
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=3.0,
        )
        ts_list.append(ts)
    result_dict = {'t': model.outputs.t, 'sim_ts': np.array(ts_list)}
    search.saveToPypet(result_dict, traj)

parameters = ParameterSpace(
    {"K_gl": np.linspace(0, 2, 2), "sigma_ou": np.linspace(0.1, 0.4, 2)}, kind="grid"
)


search = BoxSearch(
    model=model,
    evalFunction=evaluate,
    parameterSpace=parameters,
    filename="prova.hdf",
)
search.run(chunkwise=True, chunksize=60000, append=True)
search.loadResults()



# model.run(chunksize=30000, chunkwise=True, append = True)


# print(f"Number of results: {format(len(search.results))}")
# for rId in range(len(search.dfResults['a'])):
#     mean_corr = np.mean([func.matrix_correlation(func.fc(search.results[rId]['x']), fc) for fc in ds.FCs])
#     print(f"Mean correlation of run {rId} with empirical FC matrices is {mean_corr:.02}")

# %%

# for rId in range(len(search.dfResults['x'])):
#     plt.figure()
#     plt.plot(search.dfResults['t'][rId], search.dfResults['x'][rId].T)
#     plt.show()

# %%
