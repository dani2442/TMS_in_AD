#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     Evaluate differences in FC between CN subjects and MCI patients -- Version 1
Last edit:  2023/19/04
Authors:    Leone, Riccardo (RL)
Notes:      - .
            - Release notes:
                * Initial release
To do:      - Evaluate differences between HC with/without WMH and MCI with/without WMH
Comments:   

Sources: Gustavo Patow's WholeBrain Code (https://github.com/dagush/WholeBrain), Mana et al, 2023, Cerebral Cortex,
"""
# %% Imports
import matplotlib.pyplot as plt
import neurolib.utils.functions as func
from scipy.stats import ranksums
from petTOAD_setup import *

plt.rcParams["image.cmap"] = "plasma"


# Define functions
def calculate_avg_fc(group):
    n_subj = len(group)
    N = group[next(iter(group))].shape[0]
    fc_group = np.zeros([N, N])
    for _, ts in group.items():
        fc_group += func.fc(ts)
    fc_avg = fc_group / n_subj

    return fc_avg


def plot_avg_fc(name_1, fc1, name_2, fc2):
    plt.figure()
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(fc1)
    axs[0].set_title(f"{name_1}")
    axs[1].imshow(fc2)
    axs[1].set_title(f"{name_2}")
    plt.show()


def process_group_avg_fc(fc_avg_1, fc_avg_2):
    # Calculate pairwise differences between conditions
    pairwise_diff = fc_avg_1 - fc_avg_2

    # Calculate global FC as the mean value of the FC matrix across all pairs of areas
    N = fc_avg_1.shape[0]
    glob_fc_1 = (np.sum(np.abs(fc_avg_1), axis=1) - 1) / N
    glob_fc_2 = (np.sum(np.abs(fc_avg_2), axis=1) - 1) / N

    # Calculate node strength as the sum over columns of the sFC matrix
    node_strength_1 = np.sum(fc_avg_1, axis=0)
    node_strength_2 = np.sum(fc_avg_2, axis=0)

    # Calculate node diversity as the standard deviation over columns of the sFC matrix
    node_diversity_1 = np.std(fc_avg_1, axis=0)
    node_diversity_2 = np.std(fc_avg_2, axis=0)

    # Test differences in pairwise, node, and global metrics across conditions using a Wilcoxon ranksum test
    pairwise_pvalue = ranksums(pairwise_diff.flatten(), np.zeros(pairwise_diff.size))[1]
    node_strength_pvalue = ranksums(node_strength_1, node_strength_2)[1]
    node_diversity_pvalue = ranksums(node_diversity_1, node_diversity_2)[1]
    global_fc_pvalue = ranksums(glob_fc_1.flatten(), glob_fc_2.flatten())[1]

    # Apply Bonferroni correction for multiple comparisons
    n_pairs = N * (N - 1) / 2
    pairwise_alpha = 0.05 / n_pairs
    node_alpha = 0.05 / fc_avg_1.shape[0]
    global_alpha = 0.05 / fc_avg_1.shape[0]

    if pairwise_pvalue < pairwise_alpha:
        print(
            f"Pairwise differences are significant, p-value: {round(pairwise_pvalue, 5)} < {pairwise_alpha}"
        )
    else:
        print(
            f"Pairwise differences are not significant, p-value: {round(pairwise_pvalue, 5)} > {pairwise_alpha}"
        )

    if node_strength_pvalue < node_alpha:
        print(
            f"Node strength differences are significant, p-value: {round(node_strength_pvalue, 5)} < {node_alpha}"
        )
    else:
        print(
            f"Node strength differences are not significant, p-value: {round(node_strength_pvalue, 5)} > {node_alpha} "
        )

    if node_diversity_pvalue < node_alpha:
        print(
            f"Node diversity differences are significant. p-value: {round(node_diversity_pvalue, 5)} < {node_alpha}"
        )
    else:
        print(
            f"Node diversity differences are not significant, p-value: {round(node_diversity_pvalue, 5)} > {node_alpha}"
        )

    if global_fc_pvalue < global_alpha:
        print(
            f"Global FC differences are significant, p-value: {round(global_fc_pvalue, 5)} < {global_alpha}"
        )
    else:
        print(
            f"Global FC differences are not significant, p-value: {round(global_fc_pvalue, 5)} > {global_alpha}"
        )

    return [glob_fc_1, node_strength_1, node_diversity_1], [
        glob_fc_2,
        node_strength_2,
        node_diversity_2,
    ]


# %%
# Comparisons between HC and MCI total
HC_dict = {k: v for k, v in all_fMRI.items() if k in HC}
MCI_dict = {k: v for k, v in all_fMRI.items() if k in MCI}
# Calculate averages amongst groups
fc_avg_hc = calculate_avg_fc(HC_dict)
fc_avg_mci = calculate_avg_fc(MCI_dict)
# Plot average fcs
plot_avg_fc("HC", fc_avg_hc, "MCI", fc_avg_mci)
# Perform group comparison
list_results1, list_results2 = process_group_avg_fc(fc_avg_hc, fc_avg_mci)

# %%
# Comparisons between HC with/without WMH
HC_no_WMH_dict = {k: v for k, v in all_fMRI.items() if k in HC_no_WMH}
HC_WMH_dict = {k: v for k, v in all_fMRI.items() if k in HC_WMH}
# Calculate averages amongst groups
fc_avg_hc_no_wmh = calculate_avg_fc(HC_no_WMH_dict)
fc_avg_hc_wmh = calculate_avg_fc(HC_WMH_dict)
# Plot average fcs
plot_avg_fc("HC no WMH", fc_avg_hc_no_wmh, "HC WMH", fc_avg_hc_wmh)
# Perform group comparison
list_results1, list_results2 = process_group_avg_fc(fc_avg_hc_no_wmh, fc_avg_hc_wmh)

# %%
# Comparisons between MCI with/without WMH
MCI_no_WMH_dict = {k: v for k, v in all_fMRI.items() if k in MCI_no_WMH}
MCI_WMH_dict = {k: v for k, v in all_fMRI.items() if k in MCI_WMH}
# Calculate averages amongst groups
fc_avg_mci_no_wmh = calculate_avg_fc(MCI_no_WMH_dict)
fc_avg_mci_wmh = calculate_avg_fc(MCI_WMH_dict)
# Plot average fcs
plot_avg_fc("MCI no WMH", fc_avg_mci_no_wmh, "MCI WMH", fc_avg_mci_wmh)
# Perform group comparison
list_results1, list_results2 = process_group_avg_fc(fc_avg_mci_no_wmh, fc_avg_mci_wmh)
# %%
list_results1, list_results2 = process_group_avg_fc(fc_avg_hc_no_wmh, fc_avg_mci_wmh)

# %%
