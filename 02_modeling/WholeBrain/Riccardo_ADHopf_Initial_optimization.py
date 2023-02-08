# ==========================================================================
# ==========================================================================
#  Setup for the code for Hopf simulations
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import sys
import matplotlib.pyplot as plt
# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
# Import the supercritical Hopf model and set its initial value to 0.1 (don't really know why this initial value has to be set to 1)
import WholeBrain.Models.supHopf as Hopf
Hopf.initialValue = 0.1
neuronalModel = Hopf 

import WholeBrain.Integrator_EulerMaruyama as myIntegrator 
# import functions.Integrator_Euler as myIntegrator. Basically this piece of code serves to create the integration of the Hopf model
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
import WholeBrain.Observables.swFCD as swFCD
import WholeBrain.Observables.phFCD as phFCD
import WholeBrain.Optimizers.ParmSeep as ParmSeep
ParmSeep.simulateBOLD = simulateBOLD
ParmSeep.integrator = integrator
ParmSeep.verbose = True

# It is pretty standard in the literature to set the bandwidth frequencies this way.
import WholeBrain.Utils.filteredPowerSpectralDensity as filtPowSpectr
import WholeBrain.BOLDFilters as BOLDFilters
# NARROW LOW BANDPASS
BOLDFilters.flp = 0.04      # lowpass frequency of filter
BOLDFilters.fhi = 0.07       # highpass
BOLDFilters.TR = 3.
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------

#Load data

# Get the list of names of all regions in the AAL atlas. This is needed to get the right indices, to then filter the FC
import csv
# This is a sublist of label of the cortical regions that were included in the paper by Demirtas et al. - AAL atlas (78 regions, excluding infratentorial and deep)
with open ('/home/riccardo/ADNI_Hopf/Utils/aal_regions_included.csv', newline='') as g:
    new_reader = csv.reader(g)
    included_regions = list(new_reader)
g.close()

# Get the AAL atlas labels
import nilearn.datasets as datasets
aal = datasets.fetch_atlas_aal()
labels = np.array(aal.labels)
# create an array with the indices of each label (note that these are not the label number from the nifti image)
indices = np.array([i for i in enumerate(labels)])
FC_regions_index = np.isin(labels, included_regions)
# filter the indices that we want based on the position 
FC_78_regions_aal_atlas = indices[FC_regions_index]
filter_FC = np.array([int(i) for i in FC_78_regions_aal_atlas[:,0]])

# Set file path.
x_path = '/home/riccardo/ADNI_Hopf/Utils/'

# Load structural connectivity matrix and use it as parameter in Hopf model
xfile = 'SCmatrices88healthy.mat' 
M = sio.loadmat(x_path + xfile); 
mat = M['SCmatrices']
# averaging the SC among subjects
mat0 = np.mean(mat,axis = 0)
x_mat0 = mat0[filter_FC]
new_mat0 = x_mat0.T[filter_FC]
SCnorm = new_mat0 * 0.2 / new_mat0.max() 
np.fill_diagonal(SCnorm,0)
print('SCnorm.shape={}'.format(new_mat0.shape))    
Hopf.setParms({'SC':SCnorm})

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
    # ------------------- normalize and return
    # wmBurdenNorm = (wmBurden - np.min(wmBurden))/np.ptp(wmBurden)  # Normalize each individual in [0,1]
    wmBurdenNorm = (wmBurden - np.min(wm_overall))/np.ptp(wm_overall)  # Normalize the whole group in [0,1]
    return wmBurdenNorm

# ------------ load wm burden
conditionToStudy='hc' #lower case, can be hc or mci
wmBurden = loadXBurden(conditionToStudy)

# ------------ Load timeseries
baseline_group_ts = np.load('/home/riccardo/ADNI_Hopf/Results/timeseries_HC.npy')
nsubjects, nNodes, Tmax = baseline_group_ts.shape
all_HC_fMRI = {s: d for s,d in enumerate(baseline_group_ts)} 
subjectName = ''
mode = 'homogeneous'  # homogeneous/heterogeneous

# Note that this homogeneous is intended as homogeneous inside the same patient, so all regions of one patient have the same wmBurden, but different patients have different wmBurdens.
# Heterogenous, instead, means that different regions in the same patient have different wmBurdens
if mode == 'homogeneous':
    #avgwm = np.average(wmBurden)
    wmBurden = np.array([np.ones([nNodes]) * wmBurden[i] for i in range(len(wmBurden))])

# ------------------------------------------------
# Configure and compute Simulation
# ------------------------------------------------
# Since we aleardy filtered the data in the previous step from Nilearn, we aren't going to filter them again. Otherwise, a possible alternative, could be to add another
# BOLDfilters to actually re-set the filters after the f_diff was extracted and before the call to the simulation.
distanceSettings = {'FC': (FC, False), 'phFCD': (phFCD, False)}
selectedObservable = 'phFCD'

simulateBOLD.TR = 3.  # Recording interval: 1 sample every 3 seconds
simulateBOLD.dt = 0.1 * simulateBOLD.TR / 2.
simulateBOLD.Tmax = Tmax  # This is the length, in seconds
simulateBOLD.dtt = simulateBOLD.TR  # We are not using milliseconds
simulateBOLD.t_min = 10 * simulateBOLD.TR
# simulateBOLD.recomputeTmaxneuronal() <- do not update Tmaxneuronal this way!
# simulateBOLD.warmUpFactor = 6.
simulateBOLD.Tmaxneuronal = (Tmax-1) * simulateBOLD.TR + 30
integrator.ds = simulateBOLD.TR  # record every TR millisecond

base_a_value = -0.02
Hopf.setParms({'a': base_a_value})
# Hopf.beta = 0.01
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(baseline_group_ts, TR=3.)  # baseline_group[0].reshape((1,78,193))
f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
# Hopf.omega = repmat(2*pi*f_diff',1,2);     # f_diff is the frequency power
Hopf.omega = 2 * np.pi * f_diff

print("ADHopf Setup done!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF

# --------------------------------------------------------------------------------------
#  Hopf code: Pre-processing (finding G)
#  By gustavo Patow
#  Based on the code by Xenia Kobeleva
# --------------------------------------------------------------------------------------
import WholeBrain.Utils.plotFitting as plotFitting

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
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects=len(all_fMRI), 
                                            distanceSettings=distanceSettings,
                                            parmLabel='we',
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["wStart=","wEnd=","wStep="])
    except getopt.GetoptError:
        print('AD_Prepro.py --wStart <wStartValue> --wEnd <wEndValue> --wStep <wStepValue>')
        sys.exit(2)
    wStart = 0.; wEnd = 1.0; wStep = 0.5
    for opt, arg in opts:
        if opt == '-h':
            print('AD_Prepro.py -wStart <wStartValue> -wEnd <wEndValue> -wStep <wStepValue>')
            sys.exit()
        elif opt in ("--wStart"):
            wStart = float(arg)
        elif opt in ("--wEnd"):
            wEnd = float(arg)
        elif opt in ("--wStep"):
            wStep = float(arg)
    print(f'Input values are: wStart={wStart}, wEnd={wEnd}, wStep={wStep}')
    return wStart, wEnd, wStep


visualizeAll = True
subjectName = 'AvgHC'
outFilePath = '/home/riccardo/ADNI_Hopf/Results/Modeling/'+subjectName+'_78'
if __name__ == '__main__':
    wStart, wEnd, wStep = processRangeValues(sys.argv[1:])

    plt.rcParams.update({'font.size': 22})

    # ----------- Plot whatever results we have collected ------------
    # quite useful to peep at intermediate results
    # G_optim.loadAndPlot(outFilePath='Data_Produced/AD/'+subjectName+'-temp', distanceSettings=distanceSettings)

    wes = np.arange(wStart, wEnd + wStep, wStep)
    optimal = preprocessingPipeline(all_HC_fMRI,
                                    distanceSettings,
                                    wes)
    # =======  Only for quick load'n plot test...
    #plotFitting.loadAndPlot(outFilePath+'/fitting_we{}.mat', distanceSettings, WEs=np.arange(wStart, wEnd+wStep, wStep),
                            #empFilePath=outFilePath+'/fNeuro_emp.mat')

    print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    print("".join(f" - Optimal {k}({optimal[k][1]})={optimal[k][0]}\n" for k in optimal))

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF





# --------------------------------------------------------------------------------------
#  Hopf code: fitting
#  By Gustavo Patow
#  Based on the code by Xenia Koblebeva
# --------------------------------------------------------------------------------------
import WholeBrain.Utils.plotFitting as plotFitting


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline:  ParmSweep
# =====================================================================
# =====================================================================

N = nNodes
NumTrials = 5

def fittingPipeline(all_fMRI,
                    distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                    wmBurden): #,wmWs
    print("\n\n###################################################################")
    print("# Fitting with ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    wmBurden = wmBurden[0]
    wmParms = [{'a': base_a_value + wmW * wmBurden} for wmW in wmWs]
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI,
                                            wmWs, wmParms,
                                            NumSimSubjects=14,
                                            distanceSettings=distanceSettings,
                                            parmLabel='scaling',
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["wmStart=","wmEnd=","wmStep="])
    except getopt.GetoptError:
        print('AD_Prepro.py --wmStart <wmStartValue> --wmEnd <wmEndValue> --wmStep <wmStepValue>')
        sys.exit(2)
    wmStart = -1.; wmEnd = 1.05; wmStep = 0.5  # -1:0.025:1.025
    for opt, arg in opts:
        if opt == '-h':
            print('AD_Prepro.py -wmStart <wmStartValue> -wmEnd <wmEndValue> -wmStep <wmStepValue>')
            sys.exit()
        elif opt in ("--wmStart"):
            wmStart = float(arg)
        elif opt in ("--wmEnd"):
            wmEnd = float(arg)
        elif opt in ("--wmStep"):
            wmStep = float(arg)
    print(f'Input values are: wmStart={wmStart}, wmEnd={wmEnd}, wmStep={wmStep}')
    return wmStart, wmEnd, wmStep


visualizeAll = True
outFilePath = '/home/riccardo/ADNI_Hopf/Results/Modeling/'+subjectName+'_78'
if __name__ == '__main__':
    # Bias = -0.01: +0.01 (steps : 0.0025)
    # Scaling in [-1, 1.05] with steps 0.025
    print("\n\n########################################")
    print(f"Processing: {subjectName}")
    print(f"(To folder: {outFilePath})")
    print("########################################\n\n")
    wmStart, wmEnd, wmStep = processRangeValues(sys.argv[1:])
    plt.rcParams.update({'font.size': 22})

    # Set G to 2.55, obtained in the prepro stage (?)
    neuronalModel.setParms({'we': 2.55})
    # wmStart = 0.5
    wmWs = np.arange(wmStart, wmEnd + wmStep, wmStep)
    optimal = fittingPipeline(all_HC_fMRI, distanceSettings, wmBurden)
    # ------- Save result
    sio.savemat(outFilePath + f'/fittingResult-scaling-{mode}-{selectedObservable}.mat', optimal)

    # =======  Only for quick load'n plot test...
    #plotFitting.loadAndPlot(outFilePath+'/fitting_scaling{}.mat', distanceSettings, weName='scaling', title=subjectName)

    print("====================================================")
    print(f"Last info: Optimal in the CONSIDERED INTERVAL only: {wmStart}, {wmEnd}, {wmStep} (not in the whole set of results!!!)")
    print(f" - Optimal {selectedObservable}({optimal['parms']})={optimal['value']}\n")
    print("====================================================")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
