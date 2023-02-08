# Load all the packages needed for analyses
import sys
import numpy as np
import scipy.io as sio
from numba import jit
import matplotlib.pyplot as plt
import pickle


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
tmax= 193
ds = 1e-4
Tmaxneuronal = int((tmax+dt))

import WholeBrain.simulate_SimOnly as simulateBOLD
simulateBOLD.warmUp = True
simulateBOLD.integrator = integrator
simulateBOLD.warmUpFactor = 606./2000.

# Set up the code to obtain the variables we want to maximize similarity to empirical FC
import WholeBrain.Observables.FC as FC
#import WholeBrain.Observables.swFCD as swFCD
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
BOLDfilters.flp = 0.04
BOLDfilters.fhi = 0.07
BOLDfilters.TR = 3.0

# Get the list of names of all regions in the AAL atlas. This is needed to get the right indices, to then filter the FC
import csv
# This is a sublist of labels of the cortical regions that were included in the paper by Demirtas et al. - AAL atlas (78 regions, excluding infratentorial and deep)
with open ('/home/riccardo/ADNI_Hopf/Utils/aal_regions_included.csv', newline='') as f:
    new_reader = csv.reader(f)
    included_regions = list(new_reader)
f.close()

# Get the AAL atlas labels
import nilearn.datasets as datasets
aal = datasets.fetch_atlas_aal()
labels = np.array(aal.labels)
# create an array with the indices of each label (note that these are not the label number from the nifti image)
indices = np.array([i for i in enumerate(labels)])
SC_regions_index = np.isin(labels, included_regions)
# filter the indices that we want based on the position so to have a final SC matrix only for the regions we considered.
SC_78_regions_aal_atlas = indices[SC_regions_index]
filter_SC = np.array([int(i) for i in SC_78_regions_aal_atlas[:,0]])

# Set file path for SC matrix
x_path = '/home/riccardo/ADNI_Hopf/Utils/'
# Load structural connectivity matrix and use it as parameter in Hopf model
xfile = 'SCmatrices88healthy.mat' 
M = sio.loadmat(x_path + xfile); 
mat = M['SCmatrices']
# averaging the SC among subjects
mat0 = np.mean(mat,axis = 0)
# Filter the SC to have just the 78 regions we considered
x_mat0 = mat0[filter_SC]
new_mat0 = x_mat0.T[filter_SC]
# Prevent full synchronization of the model
SCnorm = new_mat0 * 0.2 / new_mat0.max() 
np.fill_diagonal(SCnorm,0)
print('SCnorm.shape={}'.format(new_mat0.shape))    
Hopf.setParms({'SC':SCnorm})

# ------------------------------------------------
# Retrieve the data for all subjects 
# ------------------------------------------------
timeseries = np.load('/home/riccardo/ADNI_Hopf/Results/timeseries_HC.npy')
nsubjects, nNodes, Tmax = timeseries.shape
all_fMRI = {s: d for s,d in enumerate(timeseries)} 
# Since we aleardy filtered the data in the previous step from Nilearn, we aren't going to filter them again. Otherwise, a possible alternative, could be to add another
# BOLDfilters to actually re-set the filters after the f_diff was extracted and before the call to the simulation.
distanceSettings = {'FC': (FC, True), 'phFCD': (phFCD, True)}

simulateBOLD.TR = 3.  # Recording interval: 1 sample every 3 seconds
simulateBOLD.dt = 0.1 * simulateBOLD.TR / 2.
simulateBOLD.Tmax = Tmax  # This is the length, in seconds
simulateBOLD.dtt = simulateBOLD.TR  # We are not using milliseconds
simulateBOLD.t_min = 10 * simulateBOLD.TR
# simulateBOLD.recomputeTmaxneuronal() <- do not update Tmaxneuronal this way!
# simulateBOLD.warmUpFactor = 6.
simulateBOLD.Tmaxneuronal = (Tmax-1) * simulateBOLD.TR + 30
integrator.ds = simulateBOLD.TR  # record every TR millisecond

# Hopf.beta = 0.01
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries, TR=3.)  # should be baseline_group_ts .. or baseling_group[0].reshape((1,52,193))
f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))

Hopf.omega = 2 * np.pi * f_diff

print("ADHopf Setup done!")

base_a_value = -0.02
warmUp = True
warmUpFactor = 10.

def computeSubjectSimulation():
    # integrator.neuronalModel.SC = C
    # integrator.initBookkeeping(N, Tmaxneuronal)
    if warmUp:
        currObsVars = integrator.warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=Tmaxneuronal/warmUpFactor)
    else:
        currObsVars = integrator.simulate(dt, Tmaxneuronal)
    # currObsVars = integrator.returnBookkeeping()  # curr_xn, curr_rn
    neuro_act = currObsVars[:,1,:]  # curr_rn
    return neuro_act
    

def loadXBurden(condition):
    # ------------------- load and stack the different wm burdens
    wm_hc = np.load('/home/riccardo/ADNI_Hopf/Results/wmh_volumes_HC.npy')
    wm_mci = np.load('/home/riccardo/ADNI_Hopf/Results/wmh_volumes_MCI.npy')
    wm_overall = np.load('/home/riccardo/ADNI_Hopf/Results/wmh_volumes_ALL.npy')
    # ------------------- load the specific subject wm
    if condition == 'hc':
        wmBurden = wm_hc
    elif condition == 'mci':
        wmBurden = wm_mci
    elif condition == 'all':
        wmBurden = wm_overall
    # ------------------- normalize and return
    # wmBurdenNorm = (wmBurden - np.min(wmBurden))/np.ptp(wmBurden)  # Normalize each individual in [0,1]
    wmBurdenNorm = (wmBurden - np.min(wm_overall))/np.ptp(wm_overall)  # Normalize the whole group in [0,1]
    return wmBurdenNorm

# ------------ load wm burden
conditionToStudy='mci' #lower case, can be hc or mci or all
wmBurden = loadXBurden(conditionToStudy)
mode = 'random'  # homogeneous/heterogeneous/random

# Note that this homogeneous is intended as homogeneous inside the same patient, so all regions of one patient have the same wmBurden, but different patients have different wmBurdens.
# Heterogenous, instead, means that different regions in the same patient have different wmBurdens
if mode == 'homogeneous':
    #avgwm = np.average(wmBurden)
    wmBurden = np.array([np.ones([nNodes]) * wmBurden[i] for i in range(len(wmBurden))])
elif mode == 'random':
    np.random.shuffle(wmBurden)


outFilePath = '/home/riccardo/ADNI_Hopf/Results/G_fitted_to_HC-minimalWMH/heterogeneous_model_across_subject_random_weights'

def fittingPipeline_homogeneous(subj_fMRI,
                    distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                    wms, wmBurden, subjectName):
    print("\n\n###################################################################")
    print("# Fitting with ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    wmParms = [{'a': base_a_value + (wmW * wmBurden)} for wmW in wmWs]
    fitting = ParmSeep.distanceForAll_Parms(subj_fMRI,
                                            wmWs, 
                                            wmParms,
                                            NumSimSubjects=1,
                                            distanceSettings=distanceSettings,
                                            parmLabel='a_random_between_subjects_',
                                            fileNameSuffix='_'+subjectName,
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal, fitting


if conditionToStudy == 'hc':
    subj_list = np.load('/home/riccardo/ADNI_Hopf/Results/subject_list_timeseries_HC.npy')
elif conditionToStudy == 'mci':
    subj_list = np.load('/home/riccardo/ADNI_Hopf/Results/subject_list_timeseries_MCI.npy')

subjectName = ''
warmUp = True
warmUpFactor = 10.
Hopf.setParms({'we': 2.9})
wmWs = np.round(np.arange(-0.1,0.1,0.005), 4)


def fittingPipeline_heterogeneous(all_fMRI, wmBurden, wmWs):
    best_parameters_dict = {}
    fitting_parameters_dict = {}

    for k, subjectName in enumerate(subj_list):
        subj_fMRI = {k:all_fMRI[k]}
        wmBurden_subj = wmBurden[k]
        best_parameters, fitting_parameters = fittingPipeline_homogeneous(subj_fMRI=subj_fMRI, distanceSettings=distanceSettings, subjectName=subjectName, wms=wmWs, wmBurden = wmBurden_subj)
        best_parameters_dict[subjectName] = best_parameters
        fitting_parameters_dict[subjectName] = fitting_parameters

    return best_parameters_dict, fitting_parameters_dict

    
best_parms_dict, fitting_parms_dict = fittingPipeline_heterogeneous(all_fMRI=all_fMRI, wmBurden=wmBurden, wmWs=wmWs)

if conditionToStudy == 'hc':

    # open file for writing, "w" 
    f = open("/home/riccardo/ADNI_Hopf/Results/G_fitted_to_HC-minimalWMH/random_model_best_parameters_dictionary_HC.pkl","wb")
    # write json object to file
    pickle.dump(best_parms_dict, f)
    # close file
    f.close()
    # open file for writing, "w" 
    g = open("/home/riccardo/ADNI_Hopf/Results/G_fitted_to_HC-minimalWMH/random_model_fitting_parameters_dictionary_HC.pkl","wb")
    # write json object to file
    pickle.dump(fitting_parms_dict, g)
    # close file
    g.close()

elif conditionToStudy == 'mci':

    # open file for writing, "w" 
    f = open("/home/riccardo/ADNI_Hopf/Results/G_fitted_to_HC-minimalWMH/random_model_best_parameters_dictionary_MCI_.pkl","wb")
    # write json object to file
    pickle.dump(best_parms_dict, f)
    # close file
    f.close()
    # open file for writing, "w" 
    g = open("/home/riccardo/ADNI_Hopf/Results/G_fitted_to_HC-minimalWMH/random_model_fitting_parameters_dictionary_MCI.pkl","wb")
    # write json object to file
    pickle.dump(fitting_parms_dict, g)
    # close file
    g.close()