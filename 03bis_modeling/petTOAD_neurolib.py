#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Find the best G coupling parameter based on healthy controls -- Version 1.0
Last edit:  2023/03/20
Authors:    Leone, Riccardo (RL)
Notes:      - Script for finding the best G on HC using Neurolib
            - Release notes:
                * Initial release
To do:      - 
Comments: 

Sources:  Neurolib
"""

#%% Hopf code: Pre-processing (finding G)
#  -------------------------------------------------------------------------------------
from petTOAD_setup import *
from neurolib.models.hopf import HopfModel
# Some useful functions are provided here
import neurolib.utils.functions as func
import matplotlib.pyplot as plt
# a nice color map
plt.rcParams['image.cmap'] = 'plasma'
#%%
model1 = HopfModel(Cmat = sc_norm, Dmat=sc_norm)
model1.params['Dmat'] = None
model1.params['a'] = -0.02
model1.params['w'] = 0.3
model1.params['duration'] = 3 * 60 * 1000 
model1.params['signalV'] = 0
model1.params['sigma_ou'] = 0.01
model1.params['K_gl'] = 0.2

model1.run(chunkwise=True, append = True, continue_run = True, bold=True)

#%%
