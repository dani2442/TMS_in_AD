# --------------------------------------------------------------------------------------
#  Hopf code: Pre-processing (finding G)
#  By gustavo Patow
#  Based on the code by Xenia Koblebeva
# --------------------------------------------------------------------------------------
from ADHopf_setup import *
import matplotlib.pyplot as plt
import functions.Utils.plotFitting as plotFitting


# def plot_cc_empSC_empFC(subjects):
#     results = []
#     for subject in subjects:
#         empSCnorm, abeta, tau, fMRI = AD_Auxiliar.loadSubjectData(subject)
#         empFC = FC.from_fMRI(fMRI)
#         corr_SC_FCemp = FC.pearson_r(empFC, empSCnorm)
#         print("{} -> Pearson_r(SCnorm, empFC) = {}".format(subject, corr_SC_FCemp))
#         results.append(corr_SC_FCemp)
#
#     plt.figure()
#     n, bins, patches = plt.hist(results, bins=6, color='#0504aa', alpha=0.7) #, histtype='step')  #, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('SC weights')
#     plt.ylabel('Counts')
#     plt.title("SC histogram", fontweight="bold", fontsize="18")
#     plt.show()


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
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects=20, #len(all_fMRI),  #10,
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
    wStart = 0.; wEnd = 6.0; wStep = 0.05
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
outFilePath = 'Data_Produced/ADHopf/'+subjectName+'_52'
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
    plotFitting.loadAndPlot(outFilePath+'/fitting_we{}.mat', distanceSettings, WEs=np.arange(wStart, wEnd+wStep, wStep),
                            empFilePath=outFilePath+'/fNeuro_emp.mat')

    print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    print("".join(f" - Optimal {k}({optimal[k][1]})={optimal[k][0]}\n" for k in optimal))

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
