#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Evaluate different preprocessing strategies on modeling -- Version 1
Last edit:  2023/03/03
Authors:    Leone, Riccardo (RL)
Notes:      - Data loader file to evaluate the impact of preprocessing on modeling
            - Release notes:
                * Initial release
To do:      - 
Comments:   All the code is taken from Gustavo Patow's WholeBrain module (https://github.com/dagush/WholeBrain)
            and slightly modified to suit our needs

Sources:    WholeBrain module (https://github.com/dagush/WholeBrain)
"""

#%% ~~ Imports ~~ %%#
# Import needed packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
from numba import jit
from petTOAD_loadXCP import *

#%% ~~ Setup ~~ %%#
# Set up Hopf as our model
import WholeBrain.Models.supHopf as Hopf
from WholeBrain.simulate_SimOnly import Tmaxneuronal

Hopf.initialValue = 0.1
neuronalModel = Hopf
# Set up our integrator
import WholeBrain.Integrator_EulerMaruyama as myIntegrator

integrator = myIntegrator
integrator.neuronalModel = neuronalModel
integrator.verbose = False
integrator.clamping = False
# Set up the integration parameters
dt = 5e-5
# tmax is equal to the number of timepoints: 193
tmax = 193
ds = 1e-4
Tmaxneuronal = int((tmax + dt))
# Simulation
import WholeBrain.simulate_SimOnly as simulateBOLD

simulateBOLD.warmUp = True
simulateBOLD.integrator = integrator
simulateBOLD.warmUpFactor = 606.0 / 2000.0
# Set up the code to obtain the variables we want to maximize similarity to empirical FC
import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.phFCD as phFCD
import WholeBrain.Optimizers.ParmSeep as ParmSeep

ParmSeep.simulateBOLD = simulateBOLD
ParmSeep.integrator = integrator
ParmSeep.verbose = True
# set BOLD filter settings
import WholeBrain.Utils.filteredPowerSpectralDensity as filtPowSpectr
import WholeBrain.BOLDFilters as BOLDfilters

# These filters are applied in the filtPowSpectr function that we use to extract the intrinsic frequencies of each region.
# They are also applied to process the FC and swFCD and phFCD, but you can set the corresponding parameter to False later on. 0.04-0.07 Hz common to extract intrinsic frequencies
BOLDfilters.flp = 0.01
BOLDfilters.fhi = 0.08
BOLDfilters.TR = 3.0

# We want to warmup the timeseries before modeling
warmUp = True
warmUpFactor = 10.0

#%% ~~ Load data ~~ %%#
# Set file path for SC matrix
sc = get_sc()
sc[sc < 0] = 0
# Prevent full synchronization of the model
SCnorm = sc * 0.2 / sc.max()
print("SCnorm.shape={}".format(SCnorm.shape))
Hopf.setParms({"SC": SCnorm})
# Setup standard parameters for the simulation
base_a_value = -0.02
Hopf.setParms({"a": base_a_value})

#%% ~~ Define useful functions ~~ %%#


def computeSubjectSimulation():
    # integrator.neuronalModel.SC = C
    # integrator.initBookkeeping(N, Tmaxneuronal)
    if warmUp:
        currObsVars = integrator.warmUpAndSimulate(
            dt, Tmaxneuronal, TWarmUp=Tmaxneuronal / warmUpFactor
        )
    else:
        currObsVars = integrator.simulate(dt, Tmaxneuronal)
    # currObsVars = integrator.returnBookkeeping()  # curr_xn, curr_rn
    neuro_act = currObsVars[:, 1, :]  # curr_rn
    return neuro_act


def fittingPipeline_homogeneous(
    all_fMRI,
    distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
    gs, outFilePath
):
    print("\n\n###################################################################")
    print("# Fitting with ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    gParms = [{"we": g} for g in gs]
    fitting = ParmSeep.distanceForAll_Parms(
        all_fMRI,
        gs,
        gParms,
        NumSimSubjects=10,
        distanceSettings=distanceSettings,
        parmLabel="finding_best_G_",
        outFilePath=outFilePath,
    )

    optimal = {
        sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings
    }
    return optimal, fitting


# ------------------------------------------------
# Configure and compute Simulation for fixed a = -0.02 and fitting G to HC to find the best G
# ------------------------------------------------
# Load the list of names for all HC
_, subjs = get_layout_subjs()
ts_dict = get_all_ts(subjs)
#%% ~~ Standard  ~~ %%#
warmUp = True
warmUpFactor = 10.0
subjectName = ""
conditionToStudy = "hc"
mode = "homogeneous"  # homogeneous/heterogeneous

# Data is already filtered
distanceSettings = {"FC": (FC, False), "phFCD": (phFCD, False)}

def process_methi(methi):
    all_fMRI = ts_dict[methi]
    nsubjects = len(all_fMRI)
    nNodes, Tmax = list(all_fMRI.values())[0].shape

    # Set simulation things
    simulateBOLD.TR = 3.0  # Recording interval: 1 sample every 3 seconds
    simulateBOLD.dt = 0.1 * simulateBOLD.TR / 2.0
    simulateBOLD.Tmax = Tmax  # This is the length, in seconds
    simulateBOLD.dtt = simulateBOLD.TR  # We are not using milliseconds
    simulateBOLD.t_min = 10 * simulateBOLD.TR
    # simulateBOLD.recomputeTmaxneuronal() <- do not update Tmaxneuronal this way!
    # simulateBOLD.warmUpFactor = 6.
    simulateBOLD.Tmaxneuronal = (Tmax - 1) * simulateBOLD.TR + 30
    integrator.ds = simulateBOLD.TR  # record every TR millisecond
    # Hopf.beta = 0.01
    timeseries_4freq = np.array([v for k, v in all_fMRI.items()])
    f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(
        timeseries_4freq, TR=3.0
    )  # should be baseline_group_ts .. or baseling_group[0].reshape((1,52,193))
    f_diff[np.where(f_diff == 0)] = np.mean(
        f_diff[np.where(f_diff != 0)]
    )  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
    Hopf.omega = 2 * np.pi * f_diff

    print(f"ADHopf Setup for {methi} done!")
    METH_PATH = OUT_XCP_DIR / f"{methi}"
    if not Path.is_dir(METH_PATH):
        Path.mkdir(METH_PATH)

    outFilePath = str(OUT_XCP_DIR) + f"/{methi}"

    Gs = np.round(np.arange(0.5, 5, 0.05), 3)

    best_parameters, fitting = fittingPipeline_homogeneous(
        all_fMRI=all_fMRI, distanceSettings=distanceSettings, gs=Gs, outFilePath=outFilePath
    )

    for ds in distanceSettings:
        plt.plot(Gs, fitting[ds], label=ds)
        plt.legend()
        optimValDist = distanceSettings[ds][0].findMinMax(fitting[ds])
        parmPos = [a for a in np.nditer(Gs)][optimValDist[1]]
        print(
            f"# Optimal {ds} =     {optimValDist[0]} @ {np.round(parmPos, decimals=3)}"
        )
    plt.savefig(outFilePath / "initial_exploration_plot.png")

list_methods = [k for k, v in ts_dict.items()]


for methi in list_methods:
    process_methi(methi)
# %%
