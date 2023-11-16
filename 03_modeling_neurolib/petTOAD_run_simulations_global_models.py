#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Model simulation with neurolib   -- Version 1.1
Last edit:  2023/08/08
Authors:    Leone, Riccardo (RL)
Notes:      - Model simulation of the phenomenological Hopf model with Neurolib
            - Release notes:
                * Updated to run on HPC
To do:      - Model with delay
Comments:   

Sources: 
"""
# %% Initial imports
from petTOAD_run_simulations import *
from petTOAD_parameter_setup_global_models import *

SIM_DIR_A = SIM_DIR / "homogeneous_a_global_model"

if not Path.exists(SIM_DIR_A):
    Path.mkdir(SIM_DIR_A)

SIM_DIR_G = SIM_DIR / "G-weight_global_model"

if not Path.exists(SIM_DIR_G):
    Path.mkdir(SIM_DIR_G)
# %%

if __name__ == "__main__":
    random_value = False
    # %% Get the frequencies for each group
    f_diff_CN_no_wmh = get_f_diff_group(CN_no_WMH)
    f_diff_CN_WMH = get_f_diff_group(CN_WMH)
    f_diff_MCI_no_WMH = get_f_diff_group(MCI_no_WMH)
    f_diff_MCI_WMH = get_f_diff_group(MCI_WMH)
    wmh_dict = get_wmh_load_homogeneous(subjs)
    # Get the subject ID
    id_subj = int(sys.argv[1]) - 1
    subj = subjs_to_sim[id_subj]
    n_sim = 20
    best_G = 1.9

    print(f"SLURM ARRAY TASK: {id_subj} corresponds to subject {subj}")
    print(f"We are going to do {n_sim} simulations for subject {subj}...")

    if subj in CN_WMH:
        f_diff = f_diff_CN_WMH
        simulate_homogeneous_model_a(
            subj=subj,
            f_diff=f_diff,
            best_G=best_G,
            wmh_dict=wmh_dict,
            ws=ws_a_cu_wmh,
            bs=bs_a_cu_wmh,
            random_cond=random_value,
            sim_dir=SIM_DIR_A,
            nsim=n_sim,
        )
        simulate_homogeneous_model_G(
            subj=subj,
            f_diff=f_diff,
            wmh_dict=wmh_dict,
            ws=ws_G_cu_wmh,
            bs=bs_G_cu_wmh,
            random_cond=random_value,
            sim_dir=SIM_DIR_G,
            nsim=n_sim,
        )

    elif subj in MCI_WMH:
        f_diff = f_diff_MCI_WMH
        simulate_homogeneous_model_a(
            subj=subj,
            f_diff=f_diff,
            best_G=best_G,
            wmh_dict=wmh_dict,
            ws=ws_a_mci_wmh,
            bs=bs_a_mci_wmh,
            random_cond=random_value,
            sim_dir=SIM_DIR_A,
            nsim=n_sim,
        )
        simulate_homogeneous_model_G(
            subj=subj,
            f_diff=f_diff,
            wmh_dict=wmh_dict,
            ws=ws_G_mci_wmh,
            bs=bs_G_mci_wmh,
            random_cond=random_value,
            sim_dir=SIM_DIR_G,
            nsim=n_sim,
        )
