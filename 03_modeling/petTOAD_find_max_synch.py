#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Find the best G coupling parameter based on healthy controls -- Version 1.0
Last edit:  2023/03/20
Authors:    Leone, Riccardo (RL)
Notes:      - Script for finding the best G on HC
            - Release notes:
                * Initial release
To do:      - 
Comments: 

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""

#%% Hopf code: Pre-processing (finding G)
#  -------------------------------------------------------------------------------------
from petTOAD_setup import *
import matplotlib.pyplot as plt
import WholeBrain.Utils.plotFitting as plotFitting

# =====================================================================
# =====================================================================
#                      Single Subject Pipeline
# =====================================================================
# =====================================================================
def preprocessingPipeline(all_fMRI,  #, abeta,
                          distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                          wes, a, val):
    print("\n\n###################################################################")
    print("# Compute ParmSeep")
    print("###################################################################\n")
    # Now, optimize all we (G) values: determine optimal G to work with
    balancedParms = [{'we': we} for we in wes]
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects=5, #len(all_fMRI),
                                            distanceSettings=distanceSettings,
                                            parmLabel=f'a-{a}_synch-{val}_we',
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
    wStart = 0.; wEnd = 6.0; wStep = 0.5
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
subjectName = 'Best_synch'

if not Path.is_dir(OUT_DIR):
    Path.mkdir(OUT_DIR)

HC_DIR = OUT_DIR / f'{subjectName}'

if not Path.is_dir(HC_DIR):
    Path.mkdir(HC_DIR)

outFilePath = str(HC_DIR)

#%%
# if __name__ == '__main__':
#     wStart, wEnd, wStep = processRangeValues(sys.argv[1:])
    # ----------- Plot whatever results we have collected ------------
    # quite useful to peep at intermediate results
    # G_optim.loadAndPlot(outFilePath='Data_Produced/AD/'+subjectName+'-temp', distanceSettings=distanceSettings)

a_s = np.round(np.arange(-0.020, 0.000, 0.001), 3)
synch_vals = np.arange(0.05,0.21, 0.01)

res_dict = {}
for a in a_s:
    res_dict[a] = {}
    Hopf.setParms({'a': a})    
for val in synch_vals:

    sc_norm = sc * val / sc.max()      
    Hopf.setParms({"SC": sc_norm})
    wes = np.arange(0, 6, .5)
    all_HC_fMRI = {k:v for k,v in all_HC_fMRI.items() if k in list(all_HC_fMRI.keys())[:5]}
    print(f'Processing {a}, {val}')
    optimal = preprocessingPipeline(all_HC_fMRI,
                                        distanceSettings,
                                        wes, a, val)

    res_dict[a][val] = optimal
    import pickle
    f = open(outFilePath + f"/evaluate_synchronicity.pkl","wb")
    # write json object to file
    pickle.dump(optimal, f)
    # close file
    f.close()
    # # =======  Only for quick load'n plot test...
    # plotFitting.loadAndPlot(outFilePath+'/fitting_we{}.mat', distanceSettings, WEs=np.arange(wStart, wEnd+wStep, wStep),
    #                         empFilePath=outFilePath+'/fNeuro_emp.mat')

    # print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    # print("".join(f" - Optimal {k}({optimal[k][1]})={optimal[k][0]}\n" for k in optimal))

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF

# %%
