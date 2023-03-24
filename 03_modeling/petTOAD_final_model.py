#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Homogeneous and heterogeneous Hopf models -- Version 1.0
Last edit:  2023/03/20
Authors:    Leone, Riccardo (RL)
Notes:      - Script for finding the best G on HC
            - Release notes:
                * Initial release
To do:      - 
Comments: 

Sources:  Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain) 
"""

# Set up Hopf as our model 
from petTOAD_setup import *
import pickle

# ------------------------------------------------
# Retrieve the data for all subjects 
# ------------------------------------------------
conditionToStudy = 'hc' # one of 'hc', 'mci', 'all'
mode = 'heterogeneous' # one of 'homogeneous', 'heterogeneous_sc', 'heterogeneous_node'
random = True # set to True if you want to shuffle the wmh weights

if conditionToStudy == 'hc':

    # Use all_HC_fMRI used to calculate the intrinsic frequencies
    all_fMRI = all_HC_fMRI
    nsubjects = len(all_fMRI) 

    if mode == 'homogeneous':
        wmBurden_dict = {k:v['total_WMH_load'] for k, v in all_dictionary.items() if k in HC}
        wmBurden = np.array([v for v in wmBurden_dict.values()])

    elif mode == 'heterogeneous_sc':
        sc_dict = {}
        for subj in subjs:
            sc_norm = get_sc_wmh_weighted(subj, sc_norm)
            sc_dict[subj] = sc_norm

    elif mode == 'heterogeneous_node':
        wmBurden_list = []
        for subj in subjs:
            node_damage_subj = get_node_damage(subj)
            wmBurden_list.append(node_damage_subj)
        wmBurden = np.array(wmBurden_list)

        
elif conditionToStudy == 'mci':

    all_fMRI = {k: v for k, v in all_fMRI.items() if k in MCI}
    nsubjects = len(all_fMRI) 

    if mode == 'homogeneous':
        wmBurden_dict = {k:v['total_WMH_load'] for k, v in all_dictionary.items() if k in MCI} # Need to write the function to obtain this.
        wmBurden = np.array([v for v in wmBurden_dict.values()])

    elif mode == 'heterogeneous_sc':
        sc_dict = {}
        for subj in subjs:
            sc_norm = get_sc_wmh_weighted(subj, sc_norm)
            sc_dict[subj] = sc_norm

    elif mode == 'heterogeneous_node':
        wmBurden_dict = {}
        for subj in subjs:
            node_damage_subj = get_node_damage(subj)
            wmBurden_dict[subj] = (node_damage_subj)

# Still to implement...
if random:
    np.random.shuffle(wmBurden)


# Change the file to where you want to save results
if not random:
    outFilePath = str(RES_DIR / f'{mode}_model') 
else:
    outFilePath = str(RES_DIR / f'{mode}random_model') 


def fittingPipeline_homogeneous(subj_fMRI,
                    distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
                    wmWs, wmBurden, subjectName):
    print("\n\n###################################################################")
    print("# Fitting with ParmSeep")
    print("###################################################################\n")
    # Now, evaluate different bifurcation parameters depending on WMH burden
    wmParms = [{'a': base_a_value + (wmW * wmBurden)} for wmW in wmWs]
    fitting = ParmSeep.distanceForAll_Parms(subj_fMRI,
                                            wmWs, 
                                            wmParms,
                                            NumSimSubjects=len(all_fMRI),
                                            distanceSettings=distanceSettings,
                                            parmLabel=f'a_{mode}_{conditionToStudy}_random_{random}',
                                            fileNameSuffix='_'+subjectName,
                                            outFilePath=outFilePath)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    return optimal, fitting


# Set the we (G coupling parm) to the best obtained in previous script
Hopf.setParms({'we': 2.9})

# Set the weights for all simulations:
wmWs = np.round(np.arange(-0.08,0.0501,0.001), 4)

def fittingPipeline_heterogeneous(all_fMRI, wmBurden_dict, wmWs):

    best_parameters_dict = {}
    fitting_parameters_dict = {}

    for subjectName, subj_ts in all_fMRI.items():

        subj_fMRI = {subjectName:subj_ts}
        wmBurden_subj = wmBurden_dict[subjectName]
        best_parameters, fitting_parameters = fittingPipeline_homogeneous(subj_fMRI=subj_fMRI, distanceSettings=distanceSettings, subjectName=subjectName, wms=wmWs, wmBurden = wmBurden_subj)
        best_parameters_dict[subjectName] = best_parameters
        fitting_parameters_dict[subjectName] = fitting_parameters

    return best_parameters_dict, fitting_parameters_dict

if not mode == 'heterogeneous_sc':
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