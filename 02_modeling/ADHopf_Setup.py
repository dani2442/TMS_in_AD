#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Nipype support -- Version 1
Last edit:  2023/03/02
Authors:    Leone, Riccardo (RL)
Notes:      - Setup file for ADNIHopf simulation
            - Release notes:
                * Initial releas
To do:      - 
Comments:   

Sources:    
"""

#%% ~~ Imports and directories ~~ %%#
# Import needed packages
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

# Directories
SPINE = Path.cwd().parents[2]
DATA_DIR = SPINE / "data"
PREP_DIR = DATA_DIR / "preprocessed"
FPRP_DIR = PREP_DIR / "fmriprep"
UTL_DIR = DATA_DIR / "utils"
WMH_DIR = PREP_DIR / "WMH_segmentation"
XCP_DIR = PREP_DIR / "xcp_d"

REL_SES = "M00"

#%%  Begin setup...
# Import the supercritical Hopf model and set its initial value to 0.1 (don't really know why this initial value has to be set to 1)
import WholeBrain.Models.supHopf as Hopf
Hopf.initialValue = 0.1
neuronalModel = Hopf 

import WholeBrain.Integrator_EulerMaruyama as myIntegrator 
integrator = myIntegrator
integrator.neuronalModel = neuronalModel
integrator.clamping = False
integrator.verbose = False

# Set up the integration parameters
dt = 5e-5
# tmax is equal to the number of timepoints: 193 
tmax= 193
ds = 1e-4
Tmaxneuronal = int((tmax+dt))

import WholeBrain.simulate_SimOnly as simulateBOLD 
simulateBOLD.warmUp = True
simulateBOLD.warmUpFactor = 606./2000.
simulateBOLD.integrator = integrator

import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.phFCD as phFCD
import WholeBrain.Optimizers.ParmSeep as ParmSeep
ParmSeep.simulateBOLD = simulateBOLD
ParmSeep.integrator = integrator
ParmSeep.verbose = True

# It is pretty standard in the literature to set the bandwidth frequencies this way.
import WholeBrain.Utils.filteredPowerSpectralDensity as filtPowSpectr
import WholeBrain.BOLDFilters as BOLDFilters
# NARROW LOW BANDPASS
BOLDFilters.flp = 0.008      # lowpass frequency of filter
BOLDFilters.fhi = 0.08       # highpass
BOLDFilters.TR = 3.
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------