# --------------------------------------------------------------------------------------
#  Hopf code: fitting
#  By Gustavo Patow
#  Based on the code by Xenia Koblebeva
# --------------------------------------------------------------------------------------
from ADHopf_setup import *
import matplotlib.pyplot as plt
import functions.Utils.plotFitting as plotFitting


# =====================================================================
# =====================================================================
#                      Single Subject Pipeline:  L2L
# =====================================================================
# =====================================================================
def setupFunc(parms):
    tauW = parms['scaling']
    # tauParms = [{'a': base_a_value + tauW * tauBurden} for tauW in tauWs]
    neuronalModel.setParms({'a': base_a_value + tauW * tauBurden})


N = nNodes
NumTrials = 5
def fittingPipelineL2L(all_fMRI,
                       distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                       selectedObservable):
    from collections import namedtuple
    from l2l.utils.experiment import Experiment
    from l2l.optimizers.gridsearch import GridSearchOptimizer, GridSearchParameters
    import functions.Optimizers.L2LOptimizee as WBOptimizee
    from functions.Optimizers.preprocessSignal import processEmpiricalSubjects  # processBOLDSignals

    # Model Simulations
    # ------------------------------------------
    # Now, optimize all alpha (B), beta (Z) values: determine optimal (B,Z) to work with
    print("\n\n###################################################################")
    print("# Fitting (scaling)")
    print("###################################################################\n")
    experiment = Experiment(root_dir_path='Data_Produced/L2L')
    name = 'L2L-ADHopf-Prepro'
    traj, _ = experiment.prepare_experiment(name=name, log_stdout=True, multiprocessing=False)

    # Setup the WhileBrain optimizee
    WBOptimizee.neuronalModel = neuronalModel
    WBOptimizee.integrator = integrator
    WBOptimizee.simulateBOLD = simulateBOLD

    # for k in list(distanceSettings):
    #     if k != selectedObservable:
    #         del distanceSettings[k]
    # distanceSettings = {'swFCD': (swFCD, True)}
    # del distanceSettings['FC'], distanceSettings['swFCD'] #, distanceSettings['phFCD']
    WBOptimizee.measure = distanceSettings[selectedObservable][0]  # Measure to use to compute the error
    WBOptimizee.applyFilters = distanceSettings[selectedObservable][1]  # Whether to apply filters to the resulting signal or not
    outEmpFileName = outFilePath + '/fNeuro_emp_L2L.mat'
    WBOptimizee.processedEmp = processEmpiricalSubjects(all_fMRI,
                                                        distanceSettings,
                                                        outEmpFileName)[selectedObservable]  # reference values (e.g., empirical) to compare to.
    WBOptimizee.N = N  # Number of regions in the parcellation
    WBOptimizee.trials = NumTrials  # Number of trials to try
    optimizee_parameters = namedtuple('OptimizeeParameters', [])

    filePattern = outFilePath + ('/fitting_{}_L2L.mat' if mode != 'homogeneous' else '/fitting_{}_homogeneous-L2L.mat')
    optimizee = WBOptimizee.WholeBrainOptimizee(traj, {'scaling': (tauStart, tauEnd)}, setupFunc=setupFunc, outFilenamePattern=filePattern)

    # =================== Test for debug only
    # traj.individual = sdict(optimizee.create_individual())
    # testing_error = optimizee.simulate(traj)
    # print("Testing error is %s", testing_error)
    # =================== end Test

    # Setup the GridSearchOptimizer
    optimizer_parameters = GridSearchParameters(param_grid={
        'scaling': (tauStart, tauEnd, int((tauEnd+tauStep-tauStart)/tauStep))
    })
    optimizer = GridSearchOptimizer(traj,
                                    optimizee_create_individual=optimizee.create_individual,
                                    optimizee_fitness_weights=(-1.,),  # minimize!
                                    parameters=optimizer_parameters)

    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)
    experiment.end_experiment(optimizer)
    print(f"best: scaling={experiment.optimizer.best_individual['scaling']}")
    return {'subject': subjectName, 'value': experiment.optimizer.best_fitness, 'parms': experiment.optimizer.best_individual['scaling']}


# # =====================================================================
# # =====================================================================
# #                      Single Subject Pipeline:  ParmSweep
# # =====================================================================
# # =====================================================================
# def fittingPipeline(all_fMRI,
#                     distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
#                     tauWs):
#     print("\n\n###################################################################")
#     print("# Fitting with ParmSeep")
#     print("###################################################################\n")
#     # Now, optimize all we (G) values: determine optimal G to work with
#
#     tauParms = [{'a': base_a_value + tauW * tauBurden} for tauW in tauWs]
#     fitting = ParmSeep.distanceForAll_Parms(all_fMRI,
#                                             tauWs, tauParms,
#                                             NumSimSubjects=5,  #len(all_fMRI)
#                                             distanceSettings=distanceSettings,
#                                             parmLabel='scaling',
#                                             outFilePath=outFilePath)
#
#     optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
#     return optimal


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
def processRangeValues(argv):
    import getopt
    try:
        opts, args = getopt.getopt(argv,'',["tauStart=","tauEnd=","tauStep="])
    except getopt.GetoptError:
        print('AD_Prepro.py --tauStart <tauStartValue> --tauEnd <tauEndValue> --tauStep <tauStepValue>')
        sys.exit(2)
    tauStart = -1.; tauEnd = 1.05; tauStep = 0.025  # -1:0.025:1.025
    for opt, arg in opts:
        if opt == '-h':
            print('AD_Prepro.py -tauStart <tauStartValue> -tauEnd <tauEndValue> -tauStep <tauStepValue>')
            sys.exit()
        elif opt in ("--tauStart"):
            tauStart = float(arg)
        elif opt in ("--tauEnd"):
            tauEnd = float(arg)
        elif opt in ("--tauStep"):
            tauStep = float(arg)
    print(f'Input values are: tauStart={tauStart}, tauEnd={tauEnd}, tauStep={tauStep}')
    return tauStart, tauEnd, tauStep


visualizeAll = True
outFilePath = 'Data_Produced/ADHopf/'+subjectName+'_52'
if __name__ == '__main__':
    # Bias = -0.01: +0.01 (steps : 0.0025)
    # Scaling in [-1, 1.05] with steps 0.025
    print("\n\n########################################")
    print(f"Processing: {subjectName}")
    print(f"(To folder: {outFilePath})")
    print("########################################\n\n")
    tauStart, tauEnd, tauStep = processRangeValues(sys.argv[1:])

    plt.rcParams.update({'font.size': 22})

    # Set G to 1.11, obtained in the prepro stage (?)
    neuronalModel.setParms({'we': 1.11})

    # tauStart = 0.5
    # tauWs = np.arange(tauStart, tauEnd + tauStep, tauStep)
    optimal = fittingPipelineL2L(all_HC_fMRI, distanceSettings, selectedObservable)
    # ------- Save result
    sio.savemat(outFilePath + f'/fittingResult-scaling-{mode}-{selectedObservable}.mat', optimal)

    # =======  Only for quick load'n plot test...
    plotFitting.loadAndPlot(outFilePath+'/fitting_scaling{}_L2L.mat', distanceSettings, weName='scaling', title=subjectName)

    print("====================================================")
    print(f"Last info: Optimal in the CONSIDERED INTERVAL only: {tauStart}, {tauEnd}, {tauStep} (not in the whole set of results!!!)")
    print(f" - Optimal {selectedObservable}({optimal['parms']})={optimal['value']}\n")
    print("====================================================")

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
