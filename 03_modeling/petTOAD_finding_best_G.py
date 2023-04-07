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
import sys 
import pickle

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
    fitting = ParmSeep.distanceForAll_Parms(all_fMRI, wes, balancedParms, NumSimSubjects= len(all_fMRI),
                                            distanceSettings=distanceSettings,
                                            parmLabel='we',
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal, fitting


# =====================================================================
# =====================================================================
#                            main
# =====================================================================
# =====================================================================
# def processRangeValues(argv):
#     import getopt
#     try:
#         opts, args = getopt.getopt(argv,'',["wStart=","wEnd=","wStep="])
#     except getopt.GetoptError:
#         print('AD_Prepro.py --wStart <wStartValue> --wEnd <wEndValue> --wStep <wStepValue>')
#         sys.exit(2)
#     wStart = 0.; wEnd = 3.0; wStep = 1
#     for opt, arg in opts:
#         if opt == '-h':
#             print('AD_Prepro.py -wStart <wStartValue> -wEnd <wEndValue> -wStep <wStepValue>')
#             sys.exit()
#         elif opt in ("--wStart"):
#             wStart = float(arg)
#         elif opt in ("--wEnd"):
#             wEnd = float(arg)
#         elif opt in ("--wStep"):
#             wStep = float(arg)
#     print(f'Input values are: wStart={wStart}, wEnd={wEnd}, wStep={wStep}')
#     return wStart, wEnd, wStep


visualizeAll = True
subjectName = 'AvgHC'

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

wes = np.arange(0, 3., 0.01)
optimal, fitting = preprocessingPipeline(all_HC_fMRI,
                                    distanceSettings,
                                    wes)


f = open(outFilePath + f"/optimal_dict_synch.pkl","wb")
# write json object to file
pickle.dump(optimal, f)
# close file
f.close()

g = open(outFilePath + f"/fitting_dict_synch.pkl","wb")
# write json object to file
pickle.dump(fitting, g)
# close file
g.close()

    # # =======  Only for quick load'n plot test...
    # plotFitting.loadAndPlot(outFilePath+'/fitting_we{}.mat', distanceSettings, WEs=np.arange(wStart, wEnd+wStep, wStep),
    #                         empFilePath=outFilePath+'/fNeuro_emp.mat')

    # print (f"Last info: Optimal in the CONSIDERED INTERVAL only: {wStart}, {wEnd}, {wStep} (not in the whole set of results!!!)")
    # print("".join(f" - Optimal {k}({optimal[k][1]})={optimal[k][0]}\n" for k in optimal))

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF

# %%
