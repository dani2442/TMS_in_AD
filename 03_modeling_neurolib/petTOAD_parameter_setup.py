#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Parameter setup for subjectwise simulations  -- Version 1.0
Last edit:  2023/08/08
Authors:    Leone, Riccardo (RL)
Notes:      - Parameter setup
            - Release notes:
                * Initial release
To do:      
Comments:   

Sources: 
"""
import numpy as np
#%%#############################################################
# Parameter setup for subject-wise simulations
################################################################

#  Set the exploration values for the homogeneous model of a
# Set the minimum and maximum values of w and b you want to explore for the bifurcation parameters 
# homogeneous model and random model.
# VERY IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Be sure to include w = 0 and b = 0, because this would be the baseline model!!!
ws_min_a = -0.1
ws_max_a = 0.1
bs_min_a = -0.05
bs_max_a = 0.05
# Set the number of parameters you want your min-max interval to be split into (remember that the n
# of simulations is equal to n_ws_a * n_bs_a, so make it reasonable to the type of PC/HPC you are running
# your simulations in)
n_ws_a = 21
n_bs_a = 11
# Create the final array with all the ws and bs you want to explore
ws_a = np.linspace(ws_min_a, ws_max_a, n_ws_a)
bs_a = np.linspace(bs_min_a, bs_max_a, n_bs_a)

#  Set the exploration values for the homogeneous model of G
# Set the minimum and maximum values of w and b you want to explore for the coupling parameter
# homogeneous model and random model
ws_min_G = -1.
ws_max_G = 1.
bs_min_G = -0.5
bs_max_G = 0.5
# Set the number of parameters you want your min-max interval to be split into (remember that the n
# of simulations is equal to n_ws_a * n_bs_a, so make it reasonable to the type of PC/HPC you are running
# your simulations in)
n_ws_G = 21
n_bs_G = 11
# Create the final array with all the ws and bs you want to explore
ws_G = np.linspace(ws_min_G, ws_max_G, n_ws_G)
bs_G = np.linspace(bs_min_G, bs_max_G, n_bs_G)

#  Set the exploration values for the heterogeneous model
# Set the minimum and maximum values of w and b you want to explore for the coupling parameter
# heterogeneous model and random model (should be the same as the homogenous a-weighted model)
ws_min_het = -0.1
ws_max_het = 0.1
bs_min_het = -0.05
bs_max_het = 0.05
# Set the number of parameters you want your min-max interval to be split into 
n_ws_het = 21
n_bs_het = 11
# Create the final array with all the ws and bs you want to explore
ws_het = np.linspace(ws_min_het, ws_max_het, n_ws_het)
bs_het = np.linspace(bs_min_het, bs_max_het, n_bs_het)

#  Set the exploration values for the delay model in terms of number of dt
ws_min_del = 1
ws_max_del = 2
# Set the number of parameters you want your min-max interval to be split into 
n_ws_del = 2
# Create the final array with all the ws and bs you want to explore
ws_het_del = np.linspace(ws_min_del, ws_max_del, n_ws_del)