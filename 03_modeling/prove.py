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
from new_petTOAD_setup import *

# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def preprocessingPipeline(all_fMRI,  #, abeta,
                          distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                          wes):
    print("\n\n###################################################################")
    
    print("# Compute ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    balancedParms = [{'we': we} for we in wes]
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects=10, #len(all_fMRI),
                                            distanceSettings=distanceSettings,
                                            parmLabel=f'G',
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal, fitting

#%%
if not Path.exists(RES_DIR / "prove"):
    Path.mkdir(RES_DIR / "prove")
outFilePath = str(RES_DIR / "prove")

a = -0.02
one_sub = {k:v for k,v in all_HC_fMRI.items() if k in ['ADNI002S1261']}
Hopf.setParms({'a': a})     
Hopf.setParms({"SC": sc_norm})
DM = np.zeros_like(sc_norm)
Hopf.setParms({"DM": DM})
wes = np.array([1])
optimal, fitting = preprocessingPipeline(one_sub,
                                    distanceSettings,
                                    wes)

# %%
