# ==========================================================================
# ==========================================================================
#  Setup for the code for Hopf simulations
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import sys
import AD_Loader

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
import functions.Models.supHopf as Hopf
Hopf.initialValue = 0.1
neuronalModel = Hopf

import functions.Integrator_EulerMaruyama as myIntegrator
# import functions.Integrator_Euler as myIntegrator
integrator = myIntegrator
integrator.neuronalModel = neuronalModel
integrator.clamping = False
integrator.verbose = False
# Integration parms...
# dt = 5e-5
# tmax = 20.
# ds = 1e-4
# Tmaxneuronal = int((tmax+dt))

import functions.simulate_SimOnly as simulateBOLD
simulateBOLD.warmUp = True
simulateBOLD.warmUpFactor = 606./2000.
simulateBOLD.integrator = integrator

import functions.Observables.FC as FC
import functions.Observables.swFCD as swFCD
import functions.Observables.phFCD as phFCD
import functions.Optimizers.ParmSeep as ParmSeep
ParmSeep.simulateBOLD = simulateBOLD
ParmSeep.integrator = integrator
ParmSeep.verbose = True

import functions.Utils.filteredPowerSpectralDensity as filtPowSpectr
import functions.BOLDFilters as BOLDFilters
# NARROW LOW BANDPASS
BOLDFilters.flp = 0.008      # lowpass frequency of filter
BOLDFilters.fhi = 0.08       # highpass
BOLDFilters.TR = 3.
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------

x_path = "Data_Raw/from_Xenia/"
def loadXData(dataset=1, condition='hc'):
    print("============================================================================")
    print("= Loading Xenia's data...")
    print("============================================================================")
    if dataset == 0:
        xfile = 'sc_fromXenia.mat'
        M = sio.loadmat(x_path + xfile); print('{} File contents:'.format(xfile), [k for k in M.keys()])
        mat0 = M['mat_zero']; print('mat_zero.shape={}'.format(mat0.shape))
        SCnorm = AD_Loader.correctSC(mat0)
        fc = M['fc']; print('fc.shape={}'.format(fc.shape))
        ts = M['timeseries']; print('timeseries.shape={}'.format(ts.shape))
        return SCnorm, fc, ts
    elif dataset == 1:
        xfile = '002_S_0413-reduced-sc.mat'
        M = sio.loadmat(x_path + xfile); print('{} File contents:'.format(xfile), [k for k in M.keys()])
        xfile_ts = '002_S_0413-reduced-timeseries.mat'
        ts = sio.loadmat(x_path + xfile_ts); print('{} File contents:'.format(xfile_ts), [k for k in ts.keys()])

        mat0 = M['mat_zero']; print('mat_zero.shape={}'.format(mat0.shape))
        SCnorm = AD_Loader.correctSC(mat0)
        mat = M['mat']; print('mat.shape={}'.format(mat.shape))
        FC = ts['FC']; print('FC.shape={}'.format(FC.shape))
        timeseries = ts['timeseries']; print('timeseries.shape={}'.format(timeseries.shape))
        return SCnorm, FC, timeseries
    elif dataset == 2:
        xfile = 'timeseries.mat'
        ts = sio.loadmat(x_path + xfile); print('{} File contents:'.format(xfile), [k for k in ts.keys()])
        timeseries = ts['timeseries']; print('timeseries.shape={}'.format(timeseries.shape))
        return None, None, timeseries
    elif dataset == 3:
        xfile = '52reg.mat'
        M = sio.loadmat(x_path + xfile); print(f'{xfile} File contents:', [k for k in M.keys()])
        mat0 = M['sc']; print('sc.shape={}'.format(mat0.shape))
        # SCnorm = AD_Auxiliar.correctSC(mat0)
        SCnorm = 0.2 * mat0 / mat0.max()
        FC = M['fc']; print('FC.shape={}'.format(FC.shape))
        timeseries = M['timeseries']; print('timeseries.shape={}'.format(timeseries.shape))
        return SCnorm, FC, timeseries
    elif dataset == 4:
        # ------------------- load SC
        # This dataset consist of a 52x52 SC matrix
        xfile = 'sc_norm.mat'
        M = sio.loadmat(x_path + xfile); print('{} File contents:'.format(x_path + xfile), [k for k in M.keys()])
        mat0 = M['sc']; print('mat_zero.shape={}'.format(mat0.shape))
        SCnorm = 0.2 * mat0 / mat0.max()
        # ------------------- timeseries
        # This dataset consist of:
        #     20 subjects with 52 regions x 197 time-points for hc
        #     1 subject with 52 regions x 197 time-points for mci
        xfile_ts = f'ts_{condition}.mat'
        timeseries = sio.loadmat(x_path + xfile_ts); print('{} File contents:'.format(x_path + xfile_ts), [k for k in timeseries.keys()])
        ts = timeseries['ts']; print('timeseries.shape={}'.format(ts.shape))
        return SCnorm, None, ts
    elif dataset == 5:
        # ------------------- load SC
        # This dataset consist of a 52x52 SC matrix
        xfile = 'mci_ADNI002S1261.mat'
        M = sio.loadmat(x_path + xfile); print('{} File contents:'.format(x_path + xfile), [k for k in M.keys()])
        mat0 = M['sc']; print('mat_zero.shape={}'.format(mat0.shape))
        SCnorm = 0.2 * mat0 / mat0.max()
        # ------------------- load timeseries
        if condition == 'hc':
            # This dataset consist of 20 subjects with 52 regions x 197 time-points (hc)
            xfile_ts = f'ts_HC.mat'
            timeseries = sio.loadmat(x_path + xfile_ts); print('{} File contents:'.format(x_path + xfile_ts), [k for k in timeseries.keys()])
            ts = timeseries['ts']; print('timeseries.shape={}'.format(ts.shape))
        else:
            # timeseries already loaded
            # This dataset consist of 1 subject with 52 regions x 197 time-points (mci)
            ts = M['ts']; print('timeseries.shape={}'.format(ts.shape))
        # ------------------- load subjectName
        global subjectName
        subjectName = M['ts_name'][0]
        return SCnorm, None, ts
    else:
        print("ERROR, no dataset recognized!!!")
        exit()


def loadXBurden(dataset, condition):
    # ------------------- load and stack the different tau burdens
    tau_hc = sio.loadmat(x_path + 'tau_hc.mat', squeeze_me=True)['tau'].astype('float64')  #print('{} File contents:'.format(tau_hc), [k for k in tau_hc.keys()])
    tau_mci = sio.loadmat(x_path + 'tau_mci.mat', squeeze_me=True)['tau'].astype('float64')  #print('{} File contents:'.format(tau_mci), [k for k in tau_hc.keys()])
    tau_ad = sio.loadmat(x_path + 'tau_ad.mat', squeeze_me=True)['tau'].astype('float64')  #print('{} File contents:'.format(tau_ad), [k for k in tau_hc.keys()])
    tau_overall = np.vstack((tau_hc, tau_mci, tau_ad))
    # ------------------- load the specific subject tau
    if dataset <= 4:
        xfile = x_path + f'tau_{condition}.mat'
        xBurden = sio.loadmat(xfile, squeeze_me=True); print('{} File contents:'.format(xfile), [k for k in xBurden.keys()])
        tauBurden = xBurden['tau'].astype('float64')
    elif dataset == 5:
        xfile = 'mci_ADNI002S1261.mat'
        M = sio.loadmat(x_path + xfile); print('{} File contents:'.format(x_path + xfile), [k for k in M.keys()])
        tauBurden = M['tau'].astype('float64').flatten()
    # ------------------- normalize and return
    # tauBurdenNorm = (tauBurden - np.min(tauBurden))/np.ptp(tauBurden)  # Normalize each individual in [0,1]
    tauBurdenNorm = (tauBurden - np.min(tau_overall))/np.ptp(tau_overall)  # Normalize the whole group in [0,1]
    return tauBurdenNorm

# --------------------------------------------------
# load all HC fMRI data from the 52-region matrix
# --------------------------------------------------
XDataToLoad = 5
subjectName = 'MCI_1'
conditionToStudy='mci'
mode = 'heterogeneous'  # homogeneous/heterogeneous

# ------------ load SCs and timelines
SCnorm, _, baseline_group_ts = loadXData(XDataToLoad, condition='hc')  # OK, SCnorm is overwritten below...
SCnorm, _, timeseries = loadXData(XDataToLoad, condition=conditionToStudy)
# ------------ load tau burden
tauBurden = loadXBurden(XDataToLoad, conditionToStudy)

# ------------ convert the SC to the "right" format
np.fill_diagonal(SCnorm,0)  # zero all diagonal elements... why?
Hopf.setParms({'SC':SCnorm})
# ------------ Convert (& trim) timeseries to WholeBrain's dictionary system.
all_HC_fMRI = {}
if XDataToLoad == 3:
    timeseries = timeseries[:,4:]  # Remove first elements to avoid initialization artifacts...
    baseline_group_ts = baseline_group_ts[:,4:]
    all_HC_fMRI = {'52reg': timeseries}
elif XDataToLoad == 4:
    timeseries = timeseries[:,:,4:]
    baseline_group_ts = baseline_group_ts[:,:,4:]
    nsubjects, nNodes, Tmax = timeseries.shape
    all_HC_fMRI = {s: d for s,d in enumerate(timeseries)}
elif XDataToLoad == 5:
    timeseries = timeseries[:,4:]  # Remove first elements to avoid initialization artifacts...
    baseline_group_ts = baseline_group_ts[:,:,4:]
    all_HC_fMRI = {'52reg': timeseries}

nNodes, Tmax = timeseries.shape

if mode == 'homogeneous':
    avgTau = np.average(tauBurden)
    tauBurden = np.ones(nNodes) * avgTau

# ------------------------------------------------
# Configure and compute Simulation
# ------------------------------------------------
# distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}
selectedObservable = 'phFCD'
distanceSettings = {'phFCD': (phFCD, True)}

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
f_diff = filtPowSpectr.filtPowSpetraMultipleSubjects(baseline_group_ts, TR=3.)  # baseline_group[0].reshape((1,52,193))
f_diff[np.where(f_diff == 0)] = np.mean(f_diff[np.where(f_diff != 0)])  # f_diff(find(f_diff==0))=mean(f_diff(find(f_diff~=0)))
# Hopf.omega = repmat(2*pi*f_diff',1,2);     # f_diff is the frequency power
Hopf.omega = 2 * np.pi * f_diff

print("ADHopf Setup done!")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
