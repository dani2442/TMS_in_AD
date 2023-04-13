# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Homogeneous and heterogeneous Hopf models -- Version 1.2
Last edit:  2023/04/13
Authors:    Leone, Riccardo (RL)
Notes:      - Script for running the model
            - Release notes:
                *Added correct G and intercept
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
conditionToStudy = "hc"  # one of 'hc', 'mci', 'all'
mode = "homogeneous"  # one of 'homogeneous', 'heterogeneous_sc', 'heterogeneous_node'
random = False  # set to True if you want to shuffle the wmh weights

wmh_dict = get_wmh_load_homogeneous(subjs)

if conditionToStudy == "hc":
    all_fMRI = {k: v for k, v in all_fMRI.items() if k in HC_WMH}
    nsubjects = len(all_fMRI)

    if mode == "homogeneous":
        wmh_burden_dict = {k: v for k, v in wmh_dict.items() if k in HC_WMH}
        wmh_burden = np.array([wmh_burden_dict.values()])
    elif mode == "heterogeneous_sc":
        sc_dict = {}
        for subj in subjs:
            sc_norm = get_sc_wmh_weighted(subj, sc_norm)
            sc_dict[subj] = sc_norm

    elif mode == "heterogeneous_node":
        wmh_burden_list = []
        for subj in subjs:
            node_damage_subj = get_node_damage(subj)
            wmh_burden_list.append(node_damage_subj)
        wmh_burden = np.array(wmh_burden_list)


elif conditionToStudy == "mci":
    all_fMRI = {k: v for k, v in all_fMRI.items() if k in MCI}
    nsubjects = len(all_fMRI)

    if mode == "homogeneous":
        wmh_burden_dict = {k: v for k, v in wmh_dict.items() if k in MCI}
        wmh_burden = np.array([wmh_burden_dict.values()])

    elif mode == "heterogeneous_sc":
        sc_dict = {}
        for subj in subjs:
            sc_norm = get_sc_wmh_weighted(subj, sc_norm)
            sc_dict[subj] = sc_norm

    elif mode == "heterogeneous_node":
        wmh_burden_dict = {}
        for subj in subjs:
            node_damage_subj = get_node_damage(subj)
            wmh_burden_dict[subj] = node_damage_subj
    elif mode == "delay":
        pass
# Still to implement...
if random:
    if mode == "homogeneous":
        wmh_burden = np.random.shuffle(wmh_burden)
    elif mode == "heterogeneous_sc":
        pass
    elif mode == "heterogeneous_node":
        pass
    elif mode == "delay":
        pass

# Change the file to where you want to save results
if not random:
    OUT_DIR = RES_DIR / f"{mode}_model"
    if not Path.exists(OUT_DIR):
        Path.mkdir(OUT_DIR)
    outFilePath = str(OUT_DIR)
else:
    OUT_DIR = RES_DIR / f"random_{mode}_model"
    if not Path.exists(OUT_DIR):
        Path.mkdir(OUT_DIR)
    outFilePath = str(OUT_DIR)


def fittingPipeline_homogeneous(
    subj_fMRI,
    distanceSettings,  # This is a dictionary of {name: (distance module, apply filters bool)}
    wmWs,
    wmh_burden,
    subjectName,
):
    print("\n\n###################################################################")
    print("# Fitting with ParmSeep")
    print("###################################################################\n")
    # Now, evaluate different bifurcation parameters depending on WMH burden
    wmParms = [
        {"a": base_a_value + (wmW * wmh_burden) + 0.002} for wmW in wmWs
    ]  
    fitting = ParmSeep.distanceForAll_Parms(
        subj_fMRI,
        wmWs,
        wmParms,
        NumSimSubjects=len(all_fMRI),
        distanceSettings=distanceSettings,
        parmLabel=f"a_{mode}_{conditionToStudy}_random_{random}",
        fileNameSuffix="_" + subjectName,
        outFilePath=outFilePath,
    )

    optimal = {
        sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings
    }
    return optimal, fitting


# Set the we (G coupling parm) to the best obtained in previous script
Hopf.setParms({"we": 0.33})

# Set the weights for all simulations:
wmWs = np.round(np.arange(-0.100, 0.100, 0.0025), 3)

def fittingPipeline_heterogeneous(all_fMRI, wmh_burden_dict, wmWs):
    best_parameters_dict = {}
    fitting_parameters_dict = {}

    for subjectName, subj_ts in all_fMRI.items():
        subj_fMRI = {subjectName: subj_ts}
        wmh_burden_subj = wmh_burden_dict[subjectName]
        best_parameters, fitting_parameters = fittingPipeline_homogeneous(
            subj_fMRI, distanceSettings, wmWs, wmh_burden_subj, subjectName
        )
        best_parameters_dict[subjectName] = best_parameters
        fitting_parameters_dict[subjectName] = fitting_parameters

    return best_parameters_dict, fitting_parameters_dict

# %%
if not mode == "heterogeneous_sc":
    best_parms_dict, fitting_parms_dict = fittingPipeline_heterogeneous(
        all_fMRI, wmh_burden_dict, wmWs
    )

    if not random:
        # open file for writing, "w"
        f = open(
            outFilePath
            + f"/{mode}_model_best_parameters_dictionary_{conditionToStudy}.pkl",
            "wb",
        )
        # write json object to file
        pickle.dump(best_parms_dict, f)
        # close file
        f.close()
        # open file for writing, "w"
        g = open(
            outFilePath
            + f"/{mode}_model_fitting_parameters_dictionary_{conditionToStudy}.pkl",
            "wb",
        )
        # write json object to file
        pickle.dump(fitting_parms_dict, g)
        # close file
        g.close()

    else:
        # open file for writing, "w"
        f = open(
            outFilePath
            + f"/random_{mode}_model_best_parameters_dictionary_{conditionToStudy}.pkl",
            "wb",
        )
        # write json object to file
        pickle.dump(best_parms_dict, f)
        # close file
        f.close()
        # open file for writing, "w"
        g = open(
            outFilePath
            + f"/random_{mode}_model_fitting_parameters_dictionary_{conditionToStudy}.pkl",
            "wb",
        )
        # write json object to file
        pickle.dump(fitting_parms_dict, g)
        # close file
        g.close()

# %%
