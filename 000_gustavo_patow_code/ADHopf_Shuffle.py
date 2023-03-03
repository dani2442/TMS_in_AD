# --------------------------------------------------------------------------------------
# Full pipeline for AD-Hopf subject processing
#
# --------------------------------------------------------------------------------------
import time, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import functions.Utils.p_values as p_values
from functions.Utils.decorators import loadOrCompute

from ADHopf_setup import *


posShuffled = 1; posDefault = 2; posOptim = 3
def plotShuffling(result, label, yLimits = None):
    # print(result)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    dataShuffled = [float(result[s]['shuffled']) for s in result.keys()]
    dataDef = [float(result[s]['default']) for s in result.keys()]  # result['011_S_4547']['default']
    dataOptim = [float(result[s]['optim']) for s in result.keys()]
    data = {'Shuffled': dataShuffled, 'Homogeneous': dataDef, 'Optim': dataOptim}

    if yLimits is not None:
        ax.set_ylim(yLimits)
    positions = {'Shuffled': posShuffled, 'Homogeneous': posDefault, 'Optim': posOptim}
    p_values.plotMeanVars(ax, data, positions, title=f'Shuffling Comparison ({label})')
    test = p_values.computeWilcoxonTests(data)
    p_values.plotWilcoxonTest(ax, test, positions, plotOrder=['Shuffled_Homogeneous', 'Homogeneous_Optim', 'Shuffled_Optim'])
    plt.show()


verbose = True
def processBOLDSignals(BOLDsignals, distanceSettings):
    NumSubjects = len(BOLDsignals)
    N = BOLDsignals[next(iter(BOLDsignals))].shape[0]  # get the first key to retrieve the value of N = number of areas

    # First, let's create a data structure for the distance measurement operations...
    measureValues = {}
    for ds in distanceSettings:  # Initialize data structs for each distance measure
        measureValues[ds] = distanceSettings[ds][0].init(NumSubjects, N)

    # Loop over subjects
    for pos, s in enumerate(BOLDsignals):
        if verbose: print('   BOLD {}/{} Subject: {} ({}x{})'.format(pos, NumSubjects-1, s, BOLDsignals[s].shape[0], BOLDsignals[s].shape[1]), end='', flush=True)
        signal = BOLDsignals[s]  # LR_version_symm(tc[s])
        start_time = time.clock()

        for ds in distanceSettings:  # Now, let's compute each measure and store the results
            measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
            applyFilters = distanceSettings[ds][1]  # whether we apply filters or not...
            procSignal = measure.from_fMRI(signal, applyFilters=applyFilters)
            measureValues[ds] = measure.accumulate(measureValues[ds], pos, procSignal)

        if verbose: print(" -> computed in {} seconds".format(time.clock() - start_time))

    for ds in distanceSettings:  # finish computing each distance measure
        measure = distanceSettings[ds][0]  # FC, swFCD, phFCD, ...
        measureValues[ds] = measure.postprocess(measureValues[ds])

    return measureValues


# ========================================================================
# Do the shuffling computations
# Returns the average of the shuffling trials
# ========================================================================
# cachePath = None
numTrials = 5
# @vectorCache(filePath=cachePath)
def evaluateFunc(parms, measure, processedEmp):
    applyFilters = distanceSettings[selectedObservable][1]
    # ------ Every time we test a new tauBurden, which was shuffled... let's put it into the neuronalModel.
    tauW = parms['parms'].flatten()
    tau = parms['tauBurden'].flatten()
    neuronalModel.setParms({'a': base_a_value + tauW * tau})
    integrator.recompileSignatures()
    # ------ And now, simulate it (numTrials times), and accumulate the results!!!
    measuredValues = measure.init(shufflingTrials, len(tau))
    for i in range(numTrials):
        bds = simulateBOLD.simulateSingleSubject().T
        procSignal = measure.from_fMRI(bds, applyFilters=applyFilters)
        measuredValues = measure.accumulate(measuredValues, i, procSignal)
    # ------ OK, let's ,compare this result with the empirical values...
    measuredValues = measure.postprocess(measuredValues)
    valueShuffled = measure.distance(measuredValues, processedEmp)
    return valueShuffled


shufflingTrials = 10
# @loadOrCompute
def testShuffledParms(subjectName, targetBOLDSeries, measure,   #distanceSetting,
                      tauBurden, optimizedParms, processedEmp):
    # ======================= Initialization =======================
    # A few global vars to simplify parameter passing... shhh... don't tell anyone! ;-)
    # global measure, N, applyFilters, SC, angles_emp, subjectName
    # AD_functions.SC = SCMatrix
    # subjectName = subjectToTest
    # AD_functions.measure= measure
    # AD_functions.applyFilters = distanceSetting[1]
    # processedBOLDemp = processBOLDSignals({subjectName: targetBOLDSeries}, {'dist': distanceSetting})['dist']
    # print("Measuring empirical data from_fMRI...")
    # AD_functions.angles_emp = measure.from_fMRI(targetBOLDSeries)
    # start_time = None
    (N, Tmax) = targetBOLDSeries.shape
    # AD_functions.N = N
    # numVars = len(setDefaultSimParms())
    # x0 = np.zeros(numVars)
    simulateBOLD.Tmax = Tmax
    simulateBOLD.recomputeTmaxneuronal()

    print("\n\n##################################################################")
    print(f"#  # of Shuffling Trials: {shufflingTrials}")
    print("##################################################################\n\n")
    results = np.zeros((shufflingTrials))
    for n in range(shufflingTrials):
        trialCounter = 0
        valueShuffled = measure.ERROR_VALUE
        while valueShuffled == measure.ERROR_VALUE:  # Retry until we do not get an error...
            # Statistically, it cannot hang the computer indefinitely... I hope! ;-)
            trialCounter += 1
            print(f"Times retried this one (trial {n}): {trialCounter}")
            np.random.shuffle(tauBurden)
            optimizedParms['tauBurden'] = tauBurden
            valueShuffled = evaluateFunc(optimizedParms, measure, processedEmp)
        print(f"Result {n}: {valueShuffled}")
        results[n] = valueShuffled

    print(f"got the following results: {results}")
    print(f"which average to {np.average(results)}")
    return {subjectName: np.average(results)}


# ========================================================================
# If we have an optimization result (we may not have it... yet!), let's
# shuffle the heterogeneous parms) and let's see how it goes...
# ========================================================================
def ShuffleSubject(subjectName,
                   distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                   measureToUse,
                   processedEmp):  # The reference data, already processed for later use
    measure = distanceSettings[measureToUse][0]
    fileName = outFilePath + f'/fittingResult-scaling-{selectedObservable}.mat'
    if Path(fileName).is_file():
        optimizedParms = sio.loadmat(fileName)
        print("\n\n##################################################################")
        print(f"#  Evaluating {subjectName} at shuffled burden!!!")
        print(f"#  With parms: {optimizedParms['parms']}")
        print(f"#  OptimValue: {optimizedParms['value']}")
        if 'default' in optimizedParms:
            print(f"#  Default Value: {optimizedParms['default']}")

        # ------------------------------------------------
        # Configure simulation
        # ------------------------------------------------
        we = 1.11  # Result from previous preprocessing using phFCD...
        neuronalModel.setParms({'we': we})

        # ------------------------------------------------
        # Simulation with default params (without tau)
        # ------------------------------------------------
        defaultParms = {'parms': np.array([0.0]), 'tauBurden': np.zeros_like(tauBurden)}
        defaultValue = evaluateFunc(defaultParms, measure, processedEmp)
        print(f"#  Default: {defaultValue}")

        # ------------------------------------------------
        # Now, the specific AD simulation
        # ------------------------------------------------
        result = testShuffledParms(subjectName, timeseries, measure,
                                   tauBurden, optimizedParms, processedEmp)
        print(f"#  Result: {result}")
        print("##################################################################\n\n")
        return {subjectName: {'shuffled': result[subjectName], 'default': defaultValue, 'optim': optimizedParms['value']}}
    else:
        return None


# ========================================================================
# Loop over subjects to test the fitting...
# ========================================================================
def testShuffling(subjects, distanceSettings, measureToUse):
    # --------- Process the empirical data to be used as a reference for comparisons
    from functions.Optimizers.preprocessSignal import processEmpiricalSubjects  # processBOLDSignals
    outEmpFileName = outFilePath + '/fNeuro_emp_L2L.mat'
    processedEmp = processEmpiricalSubjects(all_HC_fMRI,
                                            distanceSettings,
                                            outEmpFileName)[selectedObservable]  # reference values (e.g., empirical) to compare to.
    # -------- OK, let's shuffle!!!
    results = {}
    for s in subjects:
        check = ShuffleSubject(s, distanceSettings, measureToUse, processedEmp)
        if check is not None:
            results[s] = check[s]
    return results


visualizeAll = True
outFilePath = 'Data_Produced/ADHopf/'+subjectName+'_52'
if __name__ == '__main__':
    # import sys
    # group = processParmValues(sys.argv[1:])

    plt.rcParams.update({'font.size': 12})

    # ------------------------------------------------
    # Load individual classification
    # ------------------------------------------------
    # subjects = [os.path.basename(f.path) for f in os.scandir(base_folder+"/connectomes/") if f.is_dir()]
    # classification = AD_Loader.checkClassifications(subjects)
    # ADSubjects = [s for s in classification if classification[s] == 'AD']
    # MCISubjects = [s for s in classification if classification[s] == 'MCI']
    # HCSubjects = [s for s in classification if classification[s] == 'HC']

    # ------------------------------------------------
    # Load the Avg SC matrix
    # ------------------------------------------------
    # AvgHC = sio.loadmat('Data_Produced/AD/AvgHC_SC.mat')['SC']
    # AD_Loader.analyzeMatrix("AvgHC norm", AvgHC)
    # print("# of elements in AVG connectome: {}".format(AvgHC.shape))

    # ------------------------------------------------
    # Simulation settings
    # ------------------------------------------------
    # distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}

    # ------------------------------------------------
    # Run shuffling pipeline tests for the ADSubjects
    # ------------------------------------------------
    # label = group  # AD, MCI, HC
    # for label in ['AD', 'MCI', 'HC' ]:
    #     setToTest = [s for s in classification if classification[s] == label]
    #     print(f" Running for: {label}")

    setToTest = [subjectName]
    measureToUse = 'phFCD'
    result = testShuffling(setToTest, distanceSettings, measureToUse)

    # plotShuffling(result, label, [0.0, 1.0])

    # ------------------------------------------------
    # Done !!!
    # ------------------------------------------------
    print("DONE !!!")
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
