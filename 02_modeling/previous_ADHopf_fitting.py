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
                    wmWs):
    print("\n\n###################################################################")
    print("# Fitting with ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with

    wmParms = [{'a': base_a_value + wmW * wmBurden} for wmW in wmWs]
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI,
                                            wmWs, wmParms,
                                            NumSimSubjects=len(all_fMRI),
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
outFilePath = 'Data_Produced/ADHopf/'+subjectName+'_78'
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
    # wmWs = np.arange(wmStart, wmEnd + wmStep, wmStep)
    optimal = fittingPipeline(all_HC_fMRI, distanceSettings, selectedObservable)
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
