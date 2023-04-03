#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Find the best G coupling parameter based on healthy controls -- Version 1.0
Last edit:  2023/03/20
Authors:    Leone, Riccardo (RL)
Notes:      - Script for finding the best G on HC
            - Release notes:
                * Initial release
To do:      - 
Comments: 

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""

#%% Hopf code: Pre-processing (finding G)
#  -------------------------------------------------------------------------------------
import pickle
from petTOAD_setup import *
import matplotlib.pyplot as plt
import WholeBrain.Utils.plotFitting as plotFitting

# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def preprocessingPipeline(all_fMRI,  #, abeta,
                          distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                          wes, a, val):
    print("\n\n###################################################################")
    print("# Compute ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    balancedParms = [{'we': we} for we in wes]
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects=5, #len(all_fMRI),
                                            distanceSettings=distanceSettings,
                                            parmLabel=f'a-{np.round(a, 3)}_synch-{val}_we',
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal, fitting


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================


visualizeAll = True
subjectName = 'Best_synch'

if not Path.is_dir(OUT_DIR):
    Path.mkdir(OUT_DIR)

HC_DIR = OUT_DIR / f'{subjectName}'

if not Path.is_dir(HC_DIR):
    Path.mkdir(HC_DIR)

outFilePath = str(HC_DIR)

#%%
a_s = np.round(np.arange(-0.020, 0.000, 0.002), 3)
synch_vals = np.arange(0.1,0.21, 0.01)

opt_dict = {}
fit_dict = {}
for a in a_s:
    opt_dict[a] = {}
    fit_dict[a] = {}
    Hopf.setParms({'a': a})    
for val in synch_vals:

    sc_norm = sc * val / sc.max()      
    Hopf.setParms({"SC": sc_norm})
    wes = np.arange(0, 6, .5)
    all_HC_fMRI = {k:v for k,v in all_HC_fMRI.items() if k in list(all_HC_fMRI.keys())[:5]}
    print(f'Processing {a}, {val}')
    optimal, fitting = preprocessingPipeline(all_HC_fMRI,
                                        distanceSettings,
                                        wes, a, val)

    opt_dict[a][val] = optimal
    fit_dict[a][val] = optimal

f = open(outFilePath + f"/optimal_dict_synch.pkl","wb")
# write json object to file
pickle.dump(opt_dict, f)
# close file
f.close()

g = open(outFilePath + f"/fitting_dict_synch.pkl","wb")
# write json object to file
pickle.dump(opt_dict, g)
# close file
g.close()


# %%
