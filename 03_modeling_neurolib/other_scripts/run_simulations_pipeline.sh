#!/bin/bash
############################# NON-RANDOM ###################################
# Perform all simulations without random conditions and gather results
python petTOAD_exploratory_analysis_WMH_groups_a.py False
python petTOAD_exploratory_analysis_WMH_groups_G.py False
python petTOAD_exploratory_analysis_WMH_groups_heterogeneous.py False
# # Gather data for the non-random simulations
python petTOAD_exploratory_analysis_gather_data_a.py False
python petTOAD_exploratory_analysis_gather_data_G.py False
python petTOAD_exploratory_analysis_gather_data_heterogeneous.py False
############################# RANDOM ###################################
# Perform all simulations with random conditions and gather results
python petTOAD_exploratory_analysis_WMH_groups_a.py True
python petTOAD_exploratory_analysis_WMH_groups_G.py True
python petTOAD_exploratory_analysis_WMH_groups_heterogeneous.py True
# Gather data for the random simulations
python petTOAD_exploratory_analysis_gather_data_a.py True
python petTOAD_exploratory_analysis_gather_data_G.py True
python petTOAD_exploratory_analysis_gather_data_heterogeneous.py True
############################# DELAY ###################################
# Also perform simulations for the delay model.. for last since still testing
python petTOAD_exploratory_analysis_WMH_groups_delay_matrix.py
python petTOAD_exploratory_analysis_gather_data_delay.py
