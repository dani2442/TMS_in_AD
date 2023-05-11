# %%
# All neurolib imports
import neurolib.utils.functions as func
from neurolib.models.pheno_hopf import HopfModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch

# Other imports
import matplotlib.pyplot as plt
import scipy.io as sio
from petTOAD_setup import *
#%%
# Choose the group on which to perform analyses
group = all_HC_fMRI_clean
group_name = 'HC'
# Create results dir for the group
RES_DIR_GROUP = RES_DIR / group_name
if not Path.exists(RES_DIR_GROUP):
    Path.mkdir(RES_DIR_GROUP)

# Set if the model has delay
delay = False
if not delay:
    Dmat_dummy = np.zeros_like(sc)
    Dmat = Dmat_dummy
else:
    pass

#%% Define functions
# Define the evaluation function
fcs_clean = []
for ts in all_HC_fMRI_clean.values():
    fcs_clean.append(func.fc(ts))
avg_fc_clean = np.array(fcs_clean).mean(axis = 0)

def evaluate(traj):
    model = search.getModelFromTraj(traj)
    model.randomICs()
    model.run(chunkwise=True, chunksize=60000, append=True)
    # skip the first 180 secs
    ts = model.outputs.x #[:, 18000::300]
    t = model.outputs.t #[18000::300]
    ts_filt = BOLDFilters.BandPassFilter(ts)
    sFC = func.fc(ts_filt)
    result_dict = {}
    result_dict['BOLD'] = ts_filt
    result_dict['FC'] = sFC
    result_dict['t'] = t
    result_dict['fc_corr'] = func.matrix_correlation(sFC, avg_fc_clean)

    search.saveToPypet(result_dict, traj)



#%% Choose the group to analyse
group = all_HC_fMRI_clean
group_name = 'HC'
# save the FC, FCD and phFCD for the group (in order to calculate it only once)
# func.calc_and_save_group_fc(group, RES_DIR_GROUP)
# func.save_group_dynamics(group, RES_DIR_GROUP)


# %% Initialize the model
model = HopfModel(Cmat=sc, Dmat=Dmat_dummy)
model.params["Dmat"] = None
# Empirical fmri is 193 timepoints at TR=3s (9.65 min) + 3 min of initial warm up of the timeseries 
model.params["duration"] = 2. * 60 * 1000 
model.params["signalV"] = 0
model.params["w"] = 2 * np.pi * f_diff
model.params["dt"] = .1
model.params["sampling_dt"] = 10.
model.params['sigma_ou'] = 0.02
model.params['a'] = np.ones(90) * (-0.02)

#%%
parameters = ParameterSpace({'K_gl': np.round(np.linspace(1, 4, 2), 3)}, kind = 'grid')

search = BoxSearch(
    model=model,
    evalFunction=evaluate,
    parameterSpace=parameters,
    filename="new.hdf",
)
search.run(chunkwise=True, chunksize=60000, append=True)
search.loadResults()


print(search.dfResults['fc_corr'][search.dfResults['fc_corr'].argmax()])



    # BOLDFilters.flp = 0.04
    # BOLDFilters.fhi = 0.07
    # sfc_30000 = []
    # sfc_3000 = []
    # model = search.getModelFromTraj(traj)
    # model.randomICs()
    # model.run(chunkwise=True, chunksize = 60000, append = True)
    # model_ts = BOLDFilters.BandPassFilter(model.x[:,::30000])
    # model_ts2 = BOLDFilters.BandPassFilter(model.x[:,::3000])
    # sfc_30000.append(func.fc(model_ts))
    # sfc_3000.append(func.fc(model_ts2))

    # avg_sfc_30000 = np.array(sfc_30000).mean(axis = 0)
    # avg_sfc_3000 = np.array(sfc_3000).mean(axis = 0)
    # fc30000_fc = func.matrix_correlation(avg_sfc_30000, avg_fc_clean)
    # fc3000_fc = func.matrix_correlation(avg_sfc_3000, avg_fc_clean)
    # result_dict = {}
    # result_dict['BOLD30000'] = model_ts
    # result_dict['BOLD3000'] = model_ts2
    # result_dict['fc_fc30000'] = fc30000_fc
    # result_dict['fc_fc3000'] = fc3000_fc
    # result_dict['fc30000'] = avg_sfc_30000
    # result_dict['fc30000'] = avg_sfc_3000








# print(f"Number of results: {format(len(search.results))}")
# for rId in range(len(search.dfResults['a'])):
#     mean_corr = np.mean([func.matrix_correlation(func.fc(search.results[rId]['x']), fc) for fc in ds.FCs])
#     print(f"Mean correlation of run {rId} with empirical FC matrices is {mean_corr:.02}")
# %%
