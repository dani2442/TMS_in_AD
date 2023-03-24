#%% Hopf code: Pre-processing (finding G)
#  -------------------------------------------------------------------------------------
from petTOAD_setupXCP import *
import matplotlib.pyplot as plt
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
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects=2, #len(all_fMRI),  #10,
                                            distanceSettings=distanceSettings,
                                            distanceSettings_emp = distanceSettings_emp,
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
    wStart = 0.; wEnd = 6.0; wStep = 0.01
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
HC_DIR = OUT_XCP_DIR / f'{subjectName}'

if not Path.is_dir(HC_DIR):
    Path.mkdir(HC_DIR)

outFilePath = str(HC_DIR)

#%%
ts = np.load('/home/riccardo/petTOAD/results/subject_list_timeseries_HC.npy')
all_HC_fMRI = {subjs[0]:ts}
ts.shape
#%%
if __name__ == '__main__':
    wStart, wEnd, wStep = processRangeValues(sys.argv[1:])
    # Overwrite filters for intrinsic frequencies
    plt.rcParams.update({'font.size': 22})

    # ----------- Plot whatever results we have collected ------------
    # quite useful to peep at intermediate results
    # G_optim.loadAndPlot(outFilePath='Data_Produced/AD/'+subjectName+'-temp', distanceSettings=distanceSettings)

    wes = np.arange(wStart, wEnd + wStep, wStep)
    optimal = preprocessingPipeline(all_HC_fMRI,
                                    distanceSettings,
                                    wes)
    # # =======  Only for quick load'n plot test...
    # plotFitting.loadAndPlot(outFilePath+'/fitting_we{}.mat', distanceSettings, WEs=np.arange(wStart, wEnd+wStep, wStep),
    #                         empFilePath=outFilePath+'/fNeuro_emp.mat')

    # print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    # print("".join(f" - Optimal {k}({optimal[k][1]})={optimal[k][0]}\n" for k in optimal))

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF

# %%
