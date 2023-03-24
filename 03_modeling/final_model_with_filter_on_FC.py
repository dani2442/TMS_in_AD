# Note that all the code is basically taken from Gustavo Patow's WholeBrain module (https://github.com/dagush/WholeBrain)

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


# Set file path for SC matrix
home_dir = '/home/riccardo/ADNI_Hopf/'
Utils = home_dir + 'Utils/'

# Get the list of names of all regions in the AAL atlas. This is needed to get the right indices, to then filter the FC
import csv
# This is a sublist of labels of the cortical regions that were included in the paper by Demirtas et al. - AAL atlas (78 regions, excluding infratentorial and deep)
with open (Utils + 'aal_regions_included.csv', newline='') as f:
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

# Load structural connectivity matrix and use it as parameter in Hopf model
xfile = 'SCmatrices88healthy.mat' 
M = sio.loadmat(Utils + xfile); 
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

# Set the base_a_value to the same as previous studies
base_a_value = -0.02
# We want to warmup the timeseries before modeling
warmUp = True
warmUpFactor = 10.

# Set up useful functions that we will use later

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
    

def loadXBurden(dict_type):

    # ------------------- load and stack the different wm burdens
    if dict_type == 'global':
        homo_wm_file = open(home_dir + 'Results/overall_WMH_burden_all_normalized.pkl','rb')
        dict_wm_overall = pickle.load(homo_wm_file)
    
    elif dict_type == 'regional':
        hetero_wm_file = open(home_dir + 'Results/normalized_WMH_lobewise_all.pkl','rb')
        dict_wm_overall = pickle.load(hetero_wm_file)
            
    # wm_hc = np.load('/home/riccardo/ADNI_Hopf/Results/wmh_volumes_HC.npy')
    # wm_mci = np.load('/home/riccardo/ADNI_Hopf/Results/wmh_volumes_MCI.npy')
    # wm_overall = np.load('/home/riccardo/ADNI_Hopf/Results/wmh_volumes_ALL.npy')
    # ------------------- load the specific subject wm

    # ------------------- normalize and return
    # wmBurdenNorm = (wmBurden - np.min(wmBurden))/np.ptp(wmBurden)  # Normalize each individual in [0,1]
    #wmBurdenNorm = (wmBurden - np.min(wm_overall))/np.ptp(wm_overall)  # Normalize the whole group in [0,1]
    return dict_wm_overall


# ------------------------------------------------
# Retrieve the data for all subjects 
# ------------------------------------------------
conditionToStudy = 'hc' # one of 'hc', 'mci', 'all'
mode = 'heterogeneous' # one of 'homogeneous', 'heterogeneous'
random = True # set to True if you want to shuffle the wmh weights

# Probably all of this could be done in a different script....
# Load the timeseries for all subjects
all_timeseries = np.load(home_dir + 'Results/timeseries_all.npy')
# Get the number of nodes and the Tmax
nNodes, Tmax = all_timeseries.shape[1:]
# Load the subject names for all subjects
all_subjects_names = np.load(home_dir + 'Results/subject_list_timeseries_all.npy')
# Load the overall normalized WMH burden for each subject
global_wm_overall = loadXBurden('global')
# Load the normalized regional WMH burden for each subject
regional_wm_overall = loadXBurden('regional')
# create a unified dictionary to make sure we don't make mistakes when filtering
all_dictionary = {k:{'timeseries': all_timeseries[n],
                     'total_WMH_load': global_wm_overall[k],
                     'regional_WMH_load': regional_wm_overall[k]} for n, k in np.ndenumerate(all_subjects_names)}

if conditionToStudy == 'hc':

    #load the names of HC
    subjects_names = np.load(home_dir + 'Results/subject_list_timeseries_HC.npy')
    # filter the unified dictionary to retain just HC
    all_fMRI = {k:v['timeseries'] for k, v in all_dictionary.items() if k in subjects_names}
    nsubjects = len(all_fMRI) 

    if mode == 'homogeneous':
        wmBurden_dict = {k:v['total_WMH_load'] for k, v in all_dictionary.items() if k in subjects_names}
        wmBurden = np.array([v for v in wmBurden_dict.values()])

    elif mode == 'heterogeneous':
        wmBurden_dict ={k:v['regional_WMH_load'] for k, v in all_dictionary.items() if k in subjects_names} 
        wmBurden = np.array([v for v in wmBurden_dict.values()])
        
elif conditionToStudy == 'mci':

    #load the names of MCI
    subjects_names = np.load(home_dir + 'Results/subject_list_timeseries_MCI.npy')
    # filter the unified dictionary to retain just MCI
    all_fMRI = {k:v['timeseries'] for k, v in all_dictionary.items() if k in subjects_names}
    nsubjects = len(all_fMRI) 

    if mode == 'homogeneous':
        wmBurden_dict = {k:v['total_WMH_load'] for k, v in all_dictionary.items() if k in subjects_names}
        wmBurden = np.array([v for v in wmBurden_dict.values()])

    elif mode == 'heterogeneous':
        wmBurden_dict ={k:v['regional_WMH_load'] for k, v in all_dictionary.items() if k in subjects_names} 
        wmBurden = np.array([v for v in wmBurden_dict.values()])

if random:
    np.random.shuffle(wmBurden)


# We haven't filtered in Nilearn, so we are going to filter now.
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

timeseries_condition_4freq = np.array([v for k,v in all_fMRI.items()])
# Hopf.beta = 0.01
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(timeseries_condition_4freq, TR=3.)  # should be baseline_group_ts .. or baseling_group[0].reshape((1,52,193))
f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))

Hopf.omega = 2 * np.pi * f_diff

print("ADHopf Setup done!")


# Change the file to where you want to save results
if not random:
    outFilePath = home_dir + f'Results/G_fitted_to_HC-minimalWMH/10iterations_{mode}_model' 
else:
    outFilePath = home_dir + f'Results/G_fitted_to_HC-minimalWMH/random_10iterations_model' 


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
                                            NumSimSubjects=10,
                                            distanceSettings=distanceSettings,
                                            parmLabel=f'a_{mode}_{conditionToStudy}_random_{random}',
                                            fileNameSuffix='_'+subjectName,
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal, fitting

    subjectName = ''
warmUp = True
warmUpFactor = 10.
Hopf.setParms({'we': 2.9})

# This is the set of weights I used for all the simulations:
# Set of weights for homogeneous model: wmWs = np.round(np.arange(-0.09,0.05,0.001), 4)
wmWs = np.round(np.arange(-0.08,0.0501,0.001), 4)


def fittingPipeline_heterogeneous(all_fMRI, wmBurden, wmWs):

    best_parameters_dict = {}
    fitting_parameters_dict = {}

    for subjectName, subj_ts in all_fMRI.items():

        subj_fMRI = {subjectName:subj_ts}
        wmBurden_subj = wmBurden_dict[subjectName]
        best_parameters, fitting_parameters = fittingPipeline_homogeneous(subj_fMRI=subj_fMRI, distanceSettings=distanceSettings, subjectName=subjectName, wms=wmWs, wmBurden = wmBurden_subj)
        best_parameters_dict[subjectName] = best_parameters
        fitting_parameters_dict[subjectName] = fitting_parameters

    return best_parameters_dict, fitting_parameters_dict


    
best_parms_dict, fitting_parms_dict = fittingPipeline_heterogeneous(all_fMRI=all_fMRI, wmBurden=wmBurden, wmWs=wmWs)

if not random:
    # open file for writing, "w" 
    f = open(outFilePath + f"/{mode}_model_best_parameters_dictionary_{conditionToStudy}.pkl","wb")
    # write json object to file
    pickle.dump(best_parms_dict, f)
    # close file
    f.close()
    # open file for writing, "w" 
    g = open(outFilePath + f"/{mode}_model_fitting_parameters_dictionary_{conditionToStudy}.pkl","wb")
    # write json object to file
    pickle.dump(fitting_parms_dict, g)
    # close file
    g.close()

else:

    # open file for writing, "w" 
    f = open(outFilePath + f"/random_{mode}_model_best_parameters_dictionary_{conditionToStudy}.pkl","wb")
    # write json object to file
    pickle.dump(best_parms_dict, f)
    # close file
    f.close()
    # open file for writing, "w" 
    g = open(outFilePath + f"/random_{mode}_model_fitting_parameters_dictionary_{conditionToStudy}.pkl","wb")
    # write json object to file
    pickle.dump(fitting_parms_dict, g)
    # close file
    g.close()