#%%
import matplotlib.pyplot as plt
import numpy as np
import json 
# import hopf single node model and neurolib wrapper `MultiModel`
import neurolib.utils.functions as func
from neurolib.models.multimodel import HopfNetwork, HopfNode, MultiModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch

from load_petTOAD import *

# Get subjs and ts
_, subjs = get_layout_subjs()
ts_dict = get_all_ts(subjs)
all_fMRI = ts_dict['24P']

emp_FC = [func.fc(ts) for id, ts in all_fMRI.items()]
emp_phFCD = [func.ph]
#%% Get and prepare the SC matrix for Schaefer200
sc_pre = get_sc()
sc_pre[sc_pre < 0] = 0
# Prevent full synchronization of the model
sc_norm = sc_pre * 0.2 / sc_pre.max()
#Get the empirical power spectrum 
w_s = get_frequencies(all_fMRI, flp=0.04, fhi=0.07, TR=3.0)

#%% init MultiModelnetwork with no delays
mm_net = HopfNetwork(connectivity_matrix=sc_norm, delay_matrix=None)
# Initialize the multimodel network
hopf_net = MultiModel(mm_net)
hopf_net.params["duration"] = 2000.0
hopf_net.params["backend"] = "numba"
# numba uses Euler scheme so dt is important!
hopf_net.params["dt"] = 0.1
hopf_net.params["sampling_dt"] = 1.0
# Parameters setup
original_cmat = hopf_net.params['Cmat']
# Set the bifurcation parameters and intrinsic frequencies for all nodes
a_s = np.repeat(-0.02, len(sc_norm[0]))
for idx, a in enumerate(hopf_net.params['*a|sigma']):
    hopf_net.params.update({f'{a}':a_s[idx]})
for idx, w in enumerate(hopf_net.params['*w']):
    hopf_net.params.update({f'{w}':w_s[idx]})

#%%
# Explore different global coupling parameters
k_gls = np.linspace(0,5,3)
cmats = [original_cmat * k for k in k_gls]
parameters1 = ParameterSpace({'Cmat': cmats}, allow_star_notation= True)
parms_expl_filename = str(RES_DIR) + '/first_try.hdf'
search = BoxSearch(hopf_net, parameters1, filename=parms_expl_filename)
# #%% 
# netw_desc = display(mm_net.describe())
# out_file = open(RES_DIR / "hopf_network_description.json", "w")  
# json.dump(netw_desc, out_file, indent = 6)
# out_file.close()

# %%
search.run()
# hopf_net.run()
# %%
