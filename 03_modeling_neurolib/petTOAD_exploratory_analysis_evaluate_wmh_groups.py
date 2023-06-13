# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   Gather results of the repeated simulations   -- Version 1.0
Last edit:  2023/06/12
Authors:    Leone, Riccardo (RL)
Notes:      - Evaluate the different combinations of b and 2 
            - Release notes:
                * Initial release
To do:      - 
Comments:   

Sources: 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from petTOAD_setup import *

#%%
EXPL_DIR = RES_DIR / "exploratory"
EXPL_FIG_DIR = EXPL_DIR / "Figures"
if not Path.exists(EXPL_FIG_DIR):
    Path.mkdir(EXPL_FIG_DIR)

def save_plot_results(res_df, group):
    # Convert the result df into a pivot table so to plot heatmap
    table_fc = pd.pivot_table(res_df, values='fc_pearson', index='b', columns='w')
    table_fcd = pd.pivot_table(res_df, values='fcd_ks', index='b', columns='w')
    table_phfcd = pd.pivot_table(res_df, values='phfcd_ks', index='b', columns='w')
    # Create a composite score by summing up the single model fits
    table_sum = table_fc + table_fcd + table_phfcd
    plt.figure(figsize = (18,6))
    plt.subplot(131)
    sns.heatmap(table_fc.astype(float))
    plt.title(f"FC {group}")
    plt.subplot(132)
    sns.heatmap(table_fcd.astype(float))
    plt.title(f"FCD {group}")
    plt.subplot(133)
    sns.heatmap(table_phfcd.astype(float))
    plt.title(f"phFCD {group}")
    plt.savefig(EXPL_DIR / f"{group}_results_heatmap.png")
    plt.figure(figsize = (6,6))
    sns.heatmap(table_sum.astype(float))
    plt.title(f"{group} sum of model fits")
    plt.savefig(EXPL_DIR / f"{group}_summ_heatmap.png")


# Same list as the exploratory simulations
short_subjs = HC_WMH[:30]
short_subjs = np.append(short_subjs, HC_no_WMH[:30])
short_subjs = np.append(short_subjs, MCI_no_WMH[:30])
short_subjs = np.append(short_subjs, MCI_WMH[:30])

#%%
# Load wmh dictionary
wmh_dict = get_wmh_load_homogeneous(short_subjs)
# Create a overall df and populate it with single subject results
big_df = pd.DataFrame()
for subj in short_subjs:
    res_df = pd.read_csv(EXPL_DIR / f"sub-{subj}_df_results_initial_exploration_wmh.csv", index_col=0)
    res_df['sub_name'] = subj
    res_df['wmh_load'] = wmh_dict[subj]
    big_df = pd.concat([big_df, res_df], ignore_index=True)
# Let's work with 1-fcd and 1-phfcd so to have higher numbers = better fits
big_df['fcd_ks'] = 1 - big_df['fcd_ks']
big_df['phfcd_ks'] = 1 - big_df['phfcd_ks']

#%% Best model fits and relationship with wmh load
# Get the best model fits for fc, fcd, phfcd for each subject and create one single df
res_df_best = pd.DataFrame({'fc_pearson' : big_df.groupby(["sub_name"])["fc_pearson"].max()}).reset_index()
best_fcd = pd.DataFrame({'fcd_ks' : big_df.groupby(["sub_name"])["fcd_ks"].max()}).reset_index()
best_phfcd = pd.DataFrame({'phfcd_ks' : big_df.groupby(["sub_name"])["phfcd_ks"].max()}).reset_index()
res_df_best = pd.concat([res_df_best, best_fcd])
res_df_best = pd.concat([res_df_best, best_phfcd])
res_df_best["wmh_load"] = [wmh_dict[subj] for subj in res_df_best['sub_name']]
# Plot relationship between best model fits and wmh load 
# Here, all together, with regression
plt.figure()
ax1 = sns.regplot(res_df_best, y = 'fc_pearson', x = 'wmh_load', order = 2, scatter_kws={'alpha':0.3}, label = 'fc')
ax2 = sns.regplot(res_df_best, y = 'fcd_ks', x = 'wmh_load', order = 2, scatter_kws={'alpha':0.3}, label = 'fcd')
ax3 = sns.regplot(res_df_best, y = 'phfcd_ks', x = 'wmh_load', order = 2, scatter_kws={'alpha':0.3}, label = 'phfcd')
ax3.set(ylabel = 'PCC / KS distance', xlabel = 'wmh load')
plt.legend()
plt.savefig(EXPL_FIG_DIR / "summary_best_values_regression.png")
plt.close()
# Here, separate, only with points
fig, axs = plt.subplots(ncols= 1, nrows =3, figsize = (4, 12), sharex = True)
axs[0].plot(res_df_best['wmh_load'], res_df_best['fc_pearson'], 'bo', alpha = 0.3)
axs[0].set_ylabel('PCC')
axs[0].set_title('FC')
axs[1].plot(res_df_best['wmh_load'], res_df_best['fcd_ks'], 'go', alpha = 0.3)
axs[1].set_ylabel('1 - KS distance')
axs[1].set_title('FCD')
axs[2].plot(res_df_best['wmh_load'], res_df_best['phfcd_ks'], 'ko', alpha = 0.3)
axs[2].set_ylabel('1 - KS distance')
axs[2].set_title('phFCD')
fig.text(0.5, 0.04, 'Normalized WMH load', ha='center')
plt.savefig(EXPL_FIG_DIR / "summary_best_values_points.png")


#%% Evaluate different classification types (based on Fazekas etc.)
# This classification is based on Fazekas <=2
hc_no_wmh_df = big_df[big_df['sub_name'].isin(HC_no_WMH)]
hc_no_wmh_grouped = hc_no_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(hc_no_wmh_grouped, "hc_no_wmh_Fazekas_2")

hc_wmh_df = big_df[big_df['sub_name'].isin(HC_WMH)]
hc_wmh_grouped = hc_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(hc_no_wmh_grouped, "hc_wmh_Fazekas_2")

mci_no_wmh_df = big_df[big_df['sub_name'].isin(MCI_no_WMH)]
mci_no_wmh_grouped = mci_no_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(mci_no_wmh_grouped, "mci_no_wmh_Fazekas_2")

mci_wmh_df = big_df[big_df['sub_name'].isin(MCI_WMH)]
mci_wmh_grouped = mci_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(mci_wmh_grouped, "mci_wmh_Fazekas_2")

#%% This is based on wmh quartile in hc
adnimerge = pd.read_csv(RES_DIR / "petTOAD_dataframe.csv")
adnimerge["PTID"] = adnimerge["PTID"].str.replace("_", "")

HC_no_WMH_1q = adnimerge[
    (adnimerge["PTID"].isin(subjs))
    & ((adnimerge["Group_bin_subj"] == "CN_no_WMH"))
]["PTID"].unique()

HC_WMH_1q = adnimerge[
    (adnimerge["PTID"].isin(subjs)) & ((adnimerge["Group_bin_subj"] == "CN_WMH"))
]["PTID"].unique()

MCI_no_WMH_1q = adnimerge[
    (adnimerge["PTID"].isin(subjs))
    & ((adnimerge["Group_bin_subj"] == "MCI_no_WMH"))
]["PTID"].unique()

MCI_WMH_1q = adnimerge[
    (adnimerge["PTID"].isin(subjs))
    & ((adnimerge["Group_bin_subj"] == "MCI_WMH"))
]["PTID"].unique()
hc_no_wmh_df = big_df[big_df['sub_name'].isin(HC_no_WMH_1q)]
hc_no_wmh_grouped = hc_no_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(hc_no_wmh_grouped, "hc_no_wmh_1q")

hc_wmh_df = big_df[big_df['sub_name'].isin(HC_WMH_1q)]
hc_wmh_grouped = hc_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(hc_no_wmh_grouped, "hc_wmh_1q")

mci_no_wmh_df = big_df[big_df['sub_name'].isin(MCI_no_WMH_1q)]
mci_no_wmh_grouped = mci_no_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(mci_no_wmh_grouped, "mci_no_wmh_1q")

mci_wmh_df = big_df[big_df['sub_name'].isin(MCI_WMH_1q)]
mci_wmh_grouped = mci_wmh_df.drop(columns=["sub_name"]).groupby(["b", "w"]).mean()
save_plot_results(mci_wmh_grouped, "mci_wmh_1q")

# %%
