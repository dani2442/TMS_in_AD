import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib

from tqdm import tqdm
from matplotlib import cm
from functools import reduce
from statannotations.Annotator import Annotator
from scipy.stats import shapiro, pearsonr, spearmanr, wilcoxon, mannwhitneyu, chi2_contingency
from pingouin import partial_corr
from statsmodels.stats.multitest import multipletests

import my_functions as my_func
from petTOAD_load import *
from petTOAD_parameter_setup import *


TBL_DIR = RES_DIR / "Tables"
FIG_DIR = RES_DIR / "Figures"

for my_dir in [TBL_DIR, FIG_DIR]:
    if not Path.exists(my_dir):
        Path.mkdir(my_dir)

# These are the folders where the results were saved
A_DIR = SIM_DIR / f"a-weight_ws_{ws_min_a}-{ws_max_a}_bs_{bs_min_a}-{bs_max_a}"
A_RAND_DIR = (
    SIM_DIR / f"a-weight_ws_{ws_min_a}-{ws_max_a}_bs_{bs_min_a}-{bs_max_a}_random"
)
G_DIR = SIM_DIR / f"G-weight_ws_{ws_min_G}-{ws_max_G}_bs_{bs_min_G}-{bs_max_G}"
G_RAND_DIR = (
    SIM_DIR / f"G-weight_ws_{ws_min_G}-{ws_max_G}_bs_{bs_min_G}-{bs_max_G}_random"
)
SC_DIR = SIM_DIR / f"sc_disconn_ws_{ws_min_disconn}-{ws_max_disconn}_bs_0"
SC_RAND_DIR = SIM_DIR / f"sc_disconn_ws_{ws_min_disconn}-{ws_max_disconn}_bs_0_random"
SC_RAND_DIR = (
    SIM_DIR / f"sc_disconn_ws_{ws_min_disconn}-{ws_max_disconn}_bs_None_random"
)
HET_DIR = (
    SIM_DIR / f"heterogeneous_ws_{ws_min_het}-{ws_max_het}_bs_{bs_min_het}-{bs_max_het}"
)
HET_RAND_DIR = (
    SIM_DIR
    / f"heterogeneous_ws_{ws_min_het}-{ws_max_het}_bs_{bs_min_het}-{bs_max_het}_random"
)
HET_RAND_DIR = (
    SIM_DIR
    / f"heterogeneous_ws_{ws_min_het}-{ws_max_het}_bs_{bs_min_het}-{bs_max_het}_random"
)


##########################################################
### Functions to help load/format results for analyses ###
##########################################################
## Helpers for Fig 3A-B
# Function to calculate phFCD for a given group
def calculate_group_phfcd(df, group):
    """
    Calculates the phase functional connectivity dynamics (phFCD) for subjects within a specified group.

    Arguments:
    df : DataFrame
        A DataFrame containing subject information, including a "WMH_bin" column to filter by group and "PTID" for subject IDs.
    group : int or str
        The specific group to filter subjects by in the "WMH_bin" column.

    Returns:
    numpy.ndarray
        An array of phFCD values for the subjects in the specified group.
    """
    list_phfcd = []

    for subj in tqdm(df[df["WMH_bin"] == group]["PTID"]):
        ts_subj = load_ts_aal(subj)  # Assuming load_ts_aal is defined elsewhere
        phfcd = my_func.phFCD(ts_subj)  # Assuming my_func.phFCD is defined elsewhere
        list_phfcd.append(phfcd)
    arr_phfcd = np.array(list_phfcd)

    return arr_phfcd


def calculate_group_avg_wmh_map(df, group=None):
    """
    Calculates the average white matter hyperintensity (WMH) map for a specified group or for all subjects if no group is specified.

    Arguments:
    df : DataFrame
        A DataFrame containing subject information, including a "Group" column to filter by group and "PTID" for subject IDs.
    group : int or str, optional
        The specific group to filter subjects by in the "Group" column. If None, includes all subjects.

    Returns:
    nibabel.Nifti1Image
        A NIfTI image containing the average WMH map across subjects.
    """
    if group is None:
        subjs = df["PTID"]
    else:
        subjs = df[df["Group"] == group]["PTID"]

    wmh = []
    for subj in subjs:
        wmh_subj = nib.load(
            WMH_DIR
            / f"sub-{subj}"
            / f"sub-{subj}_ses-M00_space-MNI152NLin6Asym_label-WMHMask.nii.gz"
        )
        wmh_data = wmh_subj.get_fdata()
        wmh.append(wmh_data)
        wmh_aff = wmh_subj.affine

    wmh_arr = np.array(wmh)
    wmh_mean = wmh_arr.mean(axis=0)
    wmh_map = nib.Nifti1Image(wmh_mean, wmh_aff)
    return wmh_map


def calculate_concatenate_wmh_map(df, group):
    """
    Concatenates the WMH maps for subjects within a specified group.

    Arguments:
    df : DataFrame
        A DataFrame containing subject information, including a "Group" column to filter by group and "PTID" for subject IDs.
    group : int or str
        The specific group to filter subjects by in the "Group" column.

    Returns:
    tuple
        A tuple containing:
        - numpy.ndarray: An array of concatenated WMH maps for the subjects in the specified group.
        - numpy.ndarray: The affine transformation matrix for the NIfTI images.
    """
    wmh = []
    for subj in df[df["Group"] == group]["PTID"]:
        wmh_subj = nib.load(
            WMH_DIR
            / f"sub-{subj}"
            / f"sub-{subj}_ses-M00_space-MNI152NLin6Asym_label-WMHMask.nii.gz"
        )
        wmh_data = wmh_subj.get_fdata()
        wmh.append(wmh_data)
    aff = wmh_subj.affine
    return np.array(wmh), aff


##################################################################################################################################################


# Helpers for Fig 3C-E
def load_model_results(folder, parm_type, model_type):
    """
    Loads model results from a specified folder and appends parameter and model type information (e.g. homogeneous node disconnectivity).

    Arguments:
    folder : pathlib.Path
        The folder containing the model result files.
    parm_type : str
        The type of parameter used in the model (e.g., "bif_parm" or "sdc").
    model_type : str
        The type of model (e.g., "homogeneous", "heterogeneous").

    Returns:
    pandas.DataFrame
        A DataFrame containing the concatenated model results with added parameter and model type information.
    """

    df_res_model_all = pd.DataFrame()

    for item in folder.iterdir():
        if item.is_file and item.name.endswith(".csv"):
            df_res_model_subj = pd.read_csv(item, index_col=0)
            df_res_model_subj["PTID"] = item.name.split("_")[0].split("-")[1]
            df_res_model_all = pd.concat([df_res_model_all, df_res_model_subj])

    df_res_model_all["parm_type"] = parm_type
    df_res_model_all["model_type"] = model_type

    return df_res_model_all


def select_best_model_group(subjs_in_group, df_res_model_all, parm_type, model_type):
    """
    Selects the model resulting in the best average performance for a group of subjects
    based on the given parameter and model type. Note that this allows not to overfit
    model parameters to specific subjects (e.g., if we chose a different w and b for each
    subject, then we would not have a general model to apply on unseen data).

    Arguments:
    subjs_in_group : list of str
        List of subject IDs to filter the results.
    df_res_model_all : pandas.DataFrame
        DataFrame containing model results.
    parm_type : str
        The type of parameter used in the model (e.g., "bif_parm" or "sdc").
    model_type : str
        The type of model (e.g., "homogeneous", "heterogeneous").

    Returns:
    tuple
        A tuple containing:
        - pandas.DataFrame: The best model results with added columns for parameter and model type.
        - pandas.DataFrame: A simplified DataFrame with only subject IDs and the best phFCD KS values.
    """
    # Filter the results for a specific group
    df_res_model_group = df_res_model_all[df_res_model_all["PTID"].isin(subjs_in_group)]

    if parm_type == "bif_parm" or (parm_type == "sdc" and model_type == "homogeneous"):
        grouped = (
            df_res_model_group.groupby(["b", "w"])["phfcd_ks"].mean().reset_index()
        )
        best_combination = grouped.loc[grouped["phfcd_ks"].idxmin()]
        best_b = best_combination["b"]
        best_w = best_combination["w"]
        df_best = df_res_model_group[
            (df_res_model_group["b"] == best_b) & (df_res_model_group["w"] == best_w)
        ].copy()

    elif parm_type == "sdc" and model_type != "homogeneous":
        grouped = df_res_model_group.groupby("w")["phfcd_ks"].mean().reset_index()
        best_combination = grouped.loc[grouped["phfcd_ks"].idxmin()]
        best_w = best_combination["w"]
        df_best = df_res_model_group[df_res_model_group["w"] == best_w].copy()

    elif "baseline" in model_type:
        df_best = df_res_model_group[
            (df_res_model_group["b"] == 0) & (df_res_model_group["w"] == 0)
        ].copy()
        df_best["model_type"] = model_type
        df_best["parm_type"] = parm_type

    df_best[f"phfcd_ks_{parm_type}_{model_type}"] = df_best["phfcd_ks"]

    return (
        df_best.drop(columns=f"phfcd_ks_{parm_type}_{model_type}"),
        df_best.loc[:, ["PTID", f"phfcd_ks_{parm_type}_{model_type}"]],
    )


def get_best_model_df_group(subjs_in_group):
    """
    Retrieves the best model DataFrame for a group of subjects across different model types and parameters.

    Arguments:
    subjs_in_group : list of str
        List of subject IDs to filter the results. (e.g., all subjects or only high wmh)

    Returns:
    tuple
        A tuple containing:
        - list of pandas.DataFrame: The best model results for different model types and parameters.
        - list of pandas.DataFrame: Simplified DataFrames with only subject IDs and the best phFCD KS values for different model types and parameters.
    """
    # Load all non random model results, hardcoded because they are always the same folder and dir
    df_a_all = load_model_results(A_DIR, "bif_parm", "homogeneous")
    df_G_all = load_model_results(G_DIR, "sdc", "homogeneous")
    df_het_all = load_model_results(HET_DIR, "bif_parm", "heterogeneous")
    df_disconn_all = load_model_results(SC_DIR, "sdc", "heterogeneous")
    # Select the best model for the group
    df_a, df_a_short = select_best_model_group(
        subjs_in_group, df_a_all, "bif_parm", "homogeneous"
    )
    df_G, df_G_short = select_best_model_group(
        subjs_in_group, df_G_all, "sdc", "homogeneous"
    )
    df_het, df_het_short = select_best_model_group(
        subjs_in_group, df_het_all, "bif_parm", "heterogeneous"
    )
    df_disconn, df_disconn_short = select_best_model_group(
        subjs_in_group, df_disconn_all, "sdc", "heterogeneous"
    )

    df_G_subjs_in_group = df_G_all[df_G_all["PTID"].isin(subjs_in_group)]
    # The baseline is the same for all models, so we select the model with standard parameters
    df_base_sdc = df_G_subjs_in_group[
        (df_G_subjs_in_group["b"] == 0) & (df_G_subjs_in_group["w"] == 0)
    ].copy()
    # Add the model type specification to the baseline sdc df
    df_base_sdc["model_type"] = "baseline"
    df_base_bif = df_base_sdc.copy()
    # Change the parm type specification for the baseline bifurcation parameter df
    df_base_bif["parm_type"] = "bif_parm"
    df_base_bif_short = df_base_bif.loc[:, ["PTID", "phfcd_ks"]].rename(
        columns={"phfcd_ks": "phfcd_ks_bif_parm_baseline"}
    )
    df_base_sdc_short = df_base_sdc.loc[:, ["PTID", "phfcd_ks"]].rename(
        columns={"phfcd_ks": "phfcd_ks_sdc_baseline"}
    )

    list_dfs = [df_base_sdc, df_base_bif, df_a, df_G, df_het, df_disconn]
    list_dfs_short = [
        df_base_sdc_short,
        df_base_bif_short,
        df_a_short,
        df_G_short,
        df_het_short,
        df_disconn_short,
    ]

    return list_dfs, list_dfs_short


def get_best_model_df_random_group(subjs_in_group):
    """
    Retrieves the best model DataFrame for a random group of subjects across different model types and parameters.

    Arguments:
    subjs_in_group : list of str
        List of subject IDs to filter the results.

    Returns:
    list of pandas.DataFrame
        The best model results for different model types and parameters for the random group.
    """
    # Load results from random models, hardcoded because they are always the same folder and dir
    df_a_rand_all = load_model_results(A_RAND_DIR, "bif_parm", "homogeneous_random")
    df_G_rand_all = load_model_results(G_RAND_DIR, "sdc", "homogeneous_random")
    df_het_rand_all = load_model_results(
        HET_RAND_DIR, "bif_parm", "heterogeneous_random"
    )
    df_disconn_rand_all = load_model_results(SC_RAND_DIR, "sdc", "heterogeneous_random")

    # Load results from non-random models, hardcoded because they are always the same folder and dir
    df_a_all = load_model_results(A_DIR, "bif_parm", "homogeneous")
    df_G_all = load_model_results(G_DIR, "sdc", "homogeneous")
    df_het_all = load_model_results(HET_DIR, "bif_parm", "heterogeneous")
    df_disconn_all = load_model_results(SC_DIR, "sdc", "heterogeneous")

    # Select the best model weight and bias for each non-random model
    df_a, _ = select_best_model_group(
        subjs_in_group, df_a_all, "bif_parm", "homogeneous"
    )
    df_G, _ = select_best_model_group(subjs_in_group, df_G_all, "sdc", "homogeneous")
    df_het, _ = select_best_model_group(
        subjs_in_group, df_het_all, "bif_parm", "heterogeneous"
    )
    df_disconn, _ = select_best_model_group(
        subjs_in_group, df_disconn_all, "sdc", "heterogeneous"
    )

    # Filter the random df for the same weight and bias to get the values of random models
    df_a_rand = df_a_rand_all[
        (df_a_rand_all["b"] == df_a["b"].unique()[0])
        & (df_a_rand_all["w"] == df_a["w"].unique()[0])
    ]
    df_G_rand = df_G_rand_all[
        (df_G_rand_all["b"] == df_G["b"].unique()[0])
        & (df_G_rand_all["w"] == df_G["w"].unique()[0])
    ]
    df_het_rand = df_het_rand_all[
        (df_het_rand_all["b"] == df_het["b"].unique()[0])
        & (df_het_rand_all["w"] == df_het["w"].unique()[0])
    ]
    df_disconn_rand = df_disconn_rand_all[
        df_disconn_rand_all["w"] == df_disconn["w"].unique()[0]
    ]

    df_a_rand_group = df_a_rand[df_a_rand["PTID"].isin(subjs_in_group)]
    df_G_rand_group = df_G_rand[df_G_rand["PTID"].isin(subjs_in_group)]
    df_het_rand_group = df_het_rand[df_het_rand["PTID"].isin(subjs_in_group)]
    df_disconn_rand_group = df_disconn_rand[
        df_disconn_rand["PTID"].isin(subjs_in_group)
    ]

    list_dfs_rand = [
        df_a_rand_group,
        df_G_rand_group,
        df_het_rand_group,
        df_disconn_rand_group,
    ]

    return list_dfs_rand


def format_best_model_df_long_and_wide(list_dfs, list_dfs_short):
    """
    Formats the best model DataFrames into long and wide formats.

    Arguments:
    list_dfs : list of pandas.DataFrame
        List of DataFrames containing the best model results.
    list_dfs_short : list of pandas.DataFrame or None
        List of simplified DataFrames with only subject IDs and the best phFCD KS values. If None, only the long format is returned.

    Returns:
    tuple or pandas.DataFrame
        If list_dfs_short is provided:
        - tuple: A tuple containing:
          - pandas.DataFrame: The long format DataFrame for plotting.
          - pandas.DataFrame: The wide format DataFrame for difference calculations.
        If list_dfs_short is None:
        - pandas.DataFrame: The long format DataFrame for plotting.
    """
    # Prepare a df in long format for plotting
    df_model_comparison_long = pd.concat(list_dfs)
    if list_dfs_short is not None:
        # Prepare a df in wide format for difference calculations
        df_model_comparison_wide = reduce(
            lambda left, right: pd.merge(left, right, on=["PTID"], how="outer"),
            list_dfs_short,
        )
        return df_model_comparison_long, df_model_comparison_wide
    return df_model_comparison_long


def prepare_group_df_for_plotting(subjs_in_group, is_random=False):
    """
    Main function to prepare DataFrames for plotting for a group of subjects.

    Arguments:
    subjs_in_group : list of str
        List of subject IDs to filter the results.
    is_random : bool
        Whether to use random models or not.

    Returns:
    tuple
        A tuple containing:
        - pandas.DataFrame: The long format DataFrame for plotting.
        - pandas.DataFrame: The wide format DataFrame for difference calculations (if is_random is False).
    """

    if not is_random:
        list_dfs, list_dfs_short = get_best_model_df_group(subjs_in_group)
        df_model_comparison_long, df_model_comparison_wide = (
            format_best_model_df_long_and_wide(list_dfs, list_dfs_short)
        )
        return df_model_comparison_long, df_model_comparison_wide

    else:
        list_dfs = get_best_model_df_random_group(subjs_in_group)
        df_model_comparison_long = format_best_model_df_long_and_wide(
            list_dfs, list_dfs_short=None
        )
        return df_model_comparison_long


def calculate_model_diffs(df_model_compare):
    """
    Calculates model differences for comparison.

    Arguments:
    df_model_compare : pandas.DataFrame
        DataFrame containing model comparison results.
    is_random : bool
        Whether to process the DataFrame for random models or not.

    Returns:
    tuple
        A tuple containing:
        - pandas.DataFrame: The long format DataFrame with calculated differences for plotting.
        - pandas.DataFrame: The wide format DataFrame with calculated differences.
    """

    df_model_compare["phfcd_ks_diff_base_a"] = (
        df_model_compare["phfcd_ks_bif_parm_baseline"]
        - df_model_compare["phfcd_ks_bif_parm_homogeneous"]
    ) / df_model_compare["phfcd_ks_bif_parm_baseline"]
    df_model_compare["phfcd_ks_diff_base_G"] = (
        df_model_compare["phfcd_ks_sdc_baseline"]
        - df_model_compare["phfcd_ks_sdc_homogeneous"]
    ) / df_model_compare["phfcd_ks_sdc_baseline"]
    df_model_compare["phfcd_ks_diff_base_het"] = (
        df_model_compare["phfcd_ks_bif_parm_baseline"]
        - df_model_compare["phfcd_ks_bif_parm_heterogeneous"]
    ) / df_model_compare["phfcd_ks_bif_parm_baseline"]
    df_model_compare["phfcd_ks_diff_base_disconn"] = (
        df_model_compare["phfcd_ks_sdc_baseline"]
        - df_model_compare["phfcd_ks_sdc_heterogeneous"]
    ) / df_model_compare["phfcd_ks_sdc_baseline"]
    df_model_compare["phfcd_ks_diff_disconn_G"] = (
        df_model_compare["phfcd_ks_sdc_baseline"]
        - df_model_compare["phfcd_ks_sdc_homogeneous"]
    ) / df_model_compare["phfcd_ks_sdc_baseline"]

    cols_to_keep = [
        "PTID",
        "phfcd_ks_diff_base_a",
        "phfcd_ks_diff_base_G",
        "phfcd_ks_diff_base_het",
        "phfcd_ks_diff_base_disconn",
        "phfcd_ks_diff_disconn_G",
    ]

    df_long = pd.melt(
        df_model_compare[cols_to_keep],
        id_vars=["PTID"],
        var_name="model_type",
        value_name="phfcd_ks",
    )

    df_long["model_type"] = df_long["model_type"].str.replace("phfcd_ks_diff_", "")
    df_wide = df_model_compare.copy()

    return df_long, df_wide


def calculate_pvals_random_comparisons(df_plotting_random):
    """
    Calculates p-values for comparisons of random vs. non random results using the Wilcoxon
    signed-rank test for dependent samples.

    Args:
        df_plotting_random (pd.DataFrame): DataFrame containing the data for comparisons.

    Returns:
        list: List of p-values for the comparisons.
    """
    homo_sdc_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "sdc_homogeneous")
        & (df_plotting_random["random"] == "random")
    ].sort_values("PTID")["phfcd_ks"]
    homo_sdc_not_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "sdc_homogeneous")
        & (df_plotting_random["random"] == "not_random")
    ].sort_values("PTID")["phfcd_ks"]

    hetero_sdc_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "sdc_heterogeneous")
        & (df_plotting_random["random"] == "random")
    ].sort_values("PTID")["phfcd_ks"]
    hetero_sdc_not_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "sdc_heterogeneous")
        & (df_plotting_random["random"] == "not_random")
    ].sort_values("PTID")["phfcd_ks"]

    homo_bif_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "bif_parm_homogeneous")
        & (df_plotting_random["random"] == "random")
    ].sort_values("PTID")["phfcd_ks"]
    homo_bif_not_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "bif_parm_homogeneous")
        & (df_plotting_random["random"] == "not_random")
    ].sort_values("PTID")["phfcd_ks"]

    hetero_bif_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "bif_parm_heterogeneous")
        & (df_plotting_random["random"] == "random")
    ].sort_values("PTID")["phfcd_ks"]
    hetero_bif_not_rand = df_plotting_random[
        (df_plotting_random["homo_het"] == "bif_parm_heterogeneous")
        & (df_plotting_random["random"] == "not_random")
    ].sort_values("PTID")["phfcd_ks"]

    _, pval_homo_sdc = wilcoxon(homo_sdc_rand, homo_sdc_not_rand)
    _, pval_hetero_sdc = wilcoxon(hetero_sdc_rand, hetero_sdc_not_rand)
    _, pval_homo_bif = wilcoxon(homo_bif_rand, homo_bif_not_rand)
    _, pval_hetero_bif = wilcoxon(hetero_bif_rand, hetero_bif_not_rand)

    pvalues = {
        "Homogeneous SDC": pval_homo_sdc,
        "Heterogeneous SDC": pval_hetero_sdc,
        "Homogeneous NDC": pval_homo_bif,
        "Heterogeneous NDC": pval_hetero_bif,
    }
    return pvalues


###################################################
################## Plotting utils #################
###################################################

def choose_limits(list_values):
    """
    Chooses the limit for color scales based on the maximum absolute value from a list of values.

    Arguments:
    list_values : list of float
        A list of numerical values.

    Returns:
    float
        The maximum absolute value from the list.
    """
    v1 = np.abs(min(list_values))
    v2 = np.abs(max(list_values))
    if v1 > v2:
        return v1
    else:
        return v2


# Function to format p-values for display
def format_pval(p):
    """
    Formats p-values for display.

    Arguments:
    p : float
        The p-value to format.

    Returns:
    str
        A formatted string representation of the p-value.
    """
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p = {str(p):.2f}"


# Function to plot a colorbar on a figure
def plot_colorbar_ax(fig, cmap, list_positions):
    """
    Adds a colorbar to a figure at specified positions.

    Arguments:
    fig : matplotlib.figure.Figure
        The figure to add the colorbar to.
    cmap : matplotlib.colors.Colormap
        The colormap for the colorbar.
    list_positions : list of float
        A list specifying the position of the colorbar in the format [left, bottom, width, height].

    Returns:
    None
    """
    cb_ax = fig.add_axes(list_positions)
    cb = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=cb_ax, orientation="horizontal")
    cb.set_label("WMH probability (%)")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")


def write_corr_pval(df, x_col, mod_diff, ax, fontsize, correct_age):
    """
    Annotates a plot with the Spearman correlation coefficient and the corresponding p-value.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Name of the column to be used as the x variable in the correlation.
        mod_diff (str): Name of the column to be used as the y variable in the correlation.
        ax (matplotlib.axes.Axes): The axis on which to draw the annotation.
        fontsize (int): Font size for the text annotation.
        correct_age (bool): If True, perform partial correlation controlling for age.

    Returns:
        None
    """

    def format_pval_corr(p):
        if 0.04 < p < 0.05:
            return f"p = {p:.3f}"
        if p < 0.01:
            return f"p < 0.01"
        else:
            return f"p = {p:.2f}"

    if not correct_age:
        r, p = spearmanr(df[x_col], df[mod_diff])
    else:
        pc = partial_corr(
            data=df, x=x_col, y=mod_diff, covar=["Age"], method="spearman"
        )
        r = pc["r"].values[0]
        p = pc["p-val"].values[0]

    ax.text(
        0.05,
        0.73,
        f"r={r:.2f}\n{format_pval_corr(p)}",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


def choose_and_compute_correlation(data, group, column1, column2):
    """
    Tests whether to use Pearson or Spearman correlation and computes it.

    :param data: pandas DataFrame containing the data
    :param column1: Name of the first column for correlation
    :param column2: Name of the second column for correlation
    :return: Correlation value and type (Pearson or Spearman)
    """

    if group != "All":
        data = data[data["Group"] == group]
    # Test for normality
    normality_col1 = (
        shapiro(data[column1])[1] > 0.05
    )  # p-value > 0.05 suggests normal distribution
    normality_col2 = (
        shapiro(data[column2])[1] > 0.05
    )  # p-value > 0.05 suggests normal distribution

    # If both columns are normally distributed, use Pearson's correlation
    if normality_col1 and normality_col2:
        corr, _ = pearsonr(data[column1], data[column2])
        corr_type = "Pearson"
    else:
        # If either column is not normally distributed, use Spearman's correlation
        corr, p = spearmanr(data[column1], data[column2])
        corr_type = "Spearman"

    return corr, p, corr_type


#######################################################
################## Plotting functions #################
#######################################################


## Fig 3A
def plot_t1_wmh_mask(axs_row, t1_ax_rev, wmh_rev):
    """
    Plots T1-weighted images with overlaid WMH masks on a row of axes.

    Arguments:
    axs_row : list of matplotlib.axes._subplots.AxesSubplot
        A list of axes to plot on.
    t1_ax_rev : numpy.ndarray
        A 3D array containing the T1-weighted image data, with axes reversed as needed.
    wmh_rev : numpy.ndarray
        A 3D array containing the WMH mask data, with axes reversed as needed.

    Returns:
    None
    """
    axs_row[0].imshow(
        np.rot90(t1_ax_rev[60, :, :]), cmap="binary_r", alpha=0.7, aspect="auto"
    )
    axs_row[1].imshow(
        np.rot90(t1_ax_rev[:, :, 97]), cmap="binary_r", alpha=0.7, aspect="auto"
    )
    axs_row[2].imshow(
        np.rot90(t1_ax_rev[:, 115, :]), cmap="binary_r", alpha=0.7, aspect="auto"
    )
    axs_row[0].imshow(
        np.rot90(np.nanmax(wmh_rev, axis=0)), cmap="RdYlBu_r", vmin=0.0, vmax=1
    )
    axs_row[2].imshow(
        np.rot90(np.nanmax(wmh_rev, axis=1)), cmap="RdYlBu_r", vmin=0.0, vmax=1
    )
    axs_row[1].imshow(
        np.rot90(np.nanmax(wmh_rev, axis=2)), cmap="RdYlBu_r", vmin=0.0, vmax=1
    )
    plt.subplots_adjust(hspace=0, wspace=0)

    axs_row[0].axis("off")
    axs_row[1].axis("off")
    axs_row[2].axis("off")


######################################################################################################


# Fig. 3C-E
def plot_comparison(df, obs, ax):

    pairs = [
        (("sdc", "heterogeneous"), ("sdc", "baseline")),
        (("sdc", "homogeneous"), ("sdc", "baseline")),
        (("bif_parm", "heterogeneous"), ("bif_parm", "baseline")),
        (("bif_parm", "homogeneous"), ("bif_parm", "baseline")),
    ]
    # This is a dictionary of arguments that are passed into the function inside map_dataframe
    kwargs = {
        "plot_params": {  # this takes what normally goes into sns.barplot etc.
            "x": "parm_type",
            "y": obs,
            "hue": "model_type",
            "hue_order": ["baseline", "homogeneous", "heterogeneous"],
            "order": ["sdc", "bif_parm"],
        },
        "annotation_func": "apply_test",
        "configuration": {
            "test": "Wilcoxon",
            "hide_non_significant": True,
            # "comparisons_correction": "BH",
            "loc": "outside",
            "line_offset": 0.0,
            "line_height": 0.0,
            "text_offset": -1,
        },
        "plot": "boxplot",
        "ax": ax,
    }

    ant = Annotator(None, pairs)
    # We create a FacetGrid and pass the dataframe that we want to use to later apply our functions (plotting the comparisons between groups)
    g1 = sns.FacetGrid(df, aspect=1.5, height=4)
    # map_dataframe accepts a function, which it then applies to the dataframe that is previously passed in the FacetGrid. It also accepts kwargs which
    # are passed inside the function
    g1.map_dataframe(ant.plot_and_annotate_facets, **kwargs)
    # ax.set_xticklabels(ordered_names, rotation=45)
    ax.set_ylabel("KSD")
    legend = ax.get_legend()
    if legend:
        legend.remove()
    plt.close()


def plot_wmh_bmrks(df, group, axs_row):
    group_dict = {"All": "All", "CN": "CU", "MCI": "MCI"}
    corr_wmh_abeta, p_wmh_abeta, type_wmh_abeta = choose_and_compute_correlation(
        df, group, "wmh_log", "ABETA_ratio"
    )
    corr_wmh_tau, p_wmh_tau, type_wmh_tau = choose_and_compute_correlation(
        df, group, "wmh_log", "TAU"
    )
    sns.scatterplot(data=df, x="wmh_log", y="ABETA_ratio", ax=axs_row[0])
    sns.scatterplot(data=df, x="wmh_log", y="TAU", ax=axs_row[1])
    axs_row[0].set_xlabel("WMH volume (log)")
    axs_row[0].set_ylabel(r"A$\beta$42/A$\beta$40 ratio")
    axs_row[0].text(
        0.05,
        0.95,
        f"{type_wmh_abeta}'s r= {corr_wmh_abeta:.2f}, p = {p_wmh_abeta:.3f}",
        verticalalignment="top",
        horizontalalignment="left",
        transform=axs_row[0].transAxes,
    )
    axs_row[0].set_title(group_dict[group])
    axs_row[1].set_xlabel("WMH volume (log)")
    axs_row[1].set_ylabel("Tau (pg/mL)")
    axs_row[1].text(
        0.05,
        0.95,
        f"{type_wmh_tau}'s r= {corr_wmh_tau:.2f}, p = {p_wmh_tau:.3f}",
        verticalalignment="top",
        horizontalalignment="left",
        transform=axs_row[1].transAxes,
    )

####################################################
################# Tables helpers ##################
####################################################

def calculate_pvalue_comparison_long(df, obs):
    # Define the pairs for comparison
    pairs = [

        (("sdc", "homogeneous"), ("sdc", "baseline")),
        (("sdc", "heterogeneous"), ("sdc", "baseline")),
        (("bif_parm", "homogeneous"), ("bif_parm", "baseline")),
        (("bif_parm", "heterogeneous"), ("bif_parm", "baseline")),
        
    ]
    
    p_values_dict = {}
    model_name_map = {
        ("sdc", "baseline"): "Baseline",
        ("sdc", "homogeneous"): "Homogeneous SDC",
        ("sdc", "heterogeneous"): "Heterogeneous SDC",
        ("bif_parm", "homogeneous"): "Homogeneous NDC",
        ("bif_parm", "heterogeneous"): "Heterogeneous NDC"
    }
    
    p_values_list = []

    for pair in pairs:
        group1 = df[(df['parm_type'] == pair[0][0]) & (df['model_type'] == pair[0][1])][obs]
        group2 = df[(df['parm_type'] == pair[1][0]) & (df['model_type'] == pair[1][1])][obs]
        
        stat, p_value = wilcoxon(group1, group2)
        
        model_name = model_name_map[pair[0]]
        p_values_dict[model_name] = p_value
        p_values_list.append(p_value)
    
    # Apply Benjamini-Hochberg correction
    corrected_p_values = multipletests(p_values_list, method='fdr_bh')

    # Use zip to pair up the pairs and the boolean values, and filter based on the boolean values
    filtered_pairs = [pair for pair, keep in zip(pairs, corrected_p_values[0]) if keep]

    # Print the filtered pairs
    for pair in filtered_pairs:
        print(pair)

    return p_values_dict, corrected_p_values

####################################################
################# Tables functions #################
####################################################

def create_table_demographics(df_petTOAD):
    def format_pval_tbl(pval):
        return "< 0.001" if pval < 0.001 else f"{pval:.3f}"

    num_wmh = df_petTOAD.groupby("WMH_bin").count()["PTID"]["WMH"]
    num_no_wmh = df_petTOAD.groupby("WMH_bin").count()["PTID"]["no_WMH"]
    num = num_wmh + num_no_wmh

    wmh_seg_n = df_petTOAD.groupby("WMH_bin")["Sex"].value_counts()["WMH"]
    no_wmh_seg_n = df_petTOAD.groupby("WMH_bin")["Sex"].value_counts()["no_WMH"]

    wmh_age_25 = df_petTOAD.groupby("WMH_bin")["Age"].describe()["25%"]["WMH"]
    wmh_age_50 = df_petTOAD.groupby("WMH_bin")["Age"].describe()["50%"]["WMH"]
    wmh_age_75 = df_petTOAD.groupby("WMH_bin")["Age"].describe()["75%"]["WMH"]

    no_wmh_age_25 = df_petTOAD.groupby("WMH_bin")["Age"].describe()["25%"]["no_WMH"]
    no_wmh_age_50 = df_petTOAD.groupby("WMH_bin")["Age"].describe()["50%"]["no_WMH"]
    no_wmh_age_75 = df_petTOAD.groupby("WMH_bin")["Age"].describe()["75%"]["no_WMH"]

    wmh_mmse_25 = df_petTOAD.groupby("WMH_bin")["MMSE"].describe()["25%"]["WMH"]
    wmh_mmse_50 = df_petTOAD.groupby("WMH_bin")["MMSE"].describe()["50%"]["WMH"]
    wmh_mmse_75 = df_petTOAD.groupby("WMH_bin")["MMSE"].describe()["75%"]["WMH"]

    no_wmh_mmse_25 = df_petTOAD.groupby("WMH_bin")["MMSE"].describe()["25%"]["no_WMH"]
    no_wmh_mmse_50 = df_petTOAD.groupby("WMH_bin")["MMSE"].describe()["50%"]["no_WMH"]
    no_wmh_mmse_75 = df_petTOAD.groupby("WMH_bin")["MMSE"].describe()["75%"]["no_WMH"]

    wmh_edu_25 = df_petTOAD.groupby("WMH_bin")["PTEDUCAT"].describe()["25%"]["WMH"]
    wmh_edu_50 = df_petTOAD.groupby("WMH_bin")["PTEDUCAT"].describe()["50%"]["WMH"]
    wmh_edu_75 = df_petTOAD.groupby("WMH_bin")["PTEDUCAT"].describe()["75%"]["WMH"]

    no_wmh_edu_25 = df_petTOAD.groupby("WMH_bin")["PTEDUCAT"].describe()["25%"]["no_WMH"]
    no_wmh_edu_50 = df_petTOAD.groupby("WMH_bin")["PTEDUCAT"].describe()["50%"]["no_WMH"]
    no_wmh_edu_75 = df_petTOAD.groupby("WMH_bin")["PTEDUCAT"].describe()["75%"]["no_WMH"]

    wmh_wmh_log_25 = df_petTOAD.groupby("WMH_bin")["wmh_log"].describe()["25%"]["WMH"]
    wmh_wmh_log_50 = df_petTOAD.groupby("WMH_bin")["wmh_log"].describe()["50%"]["WMH"]
    wmh_wmh_log_75 = df_petTOAD.groupby("WMH_bin")["wmh_log"].describe()["75%"]["WMH"]

    no_wmh_wmh_log_25 = df_petTOAD.groupby("WMH_bin")["wmh_log"].describe()["25%"]["no_WMH"]
    no_wmh_wmh_log_50 = df_petTOAD.groupby("WMH_bin")["wmh_log"].describe()["50%"]["no_WMH"]
    no_wmh_wmh_log_75 = df_petTOAD.groupby("WMH_bin")["wmh_log"].describe()["75%"]["no_WMH"]


    _, pval_age = mannwhitneyu(df_petTOAD[df_petTOAD["WMH_bin"] == "no_WMH"]["Age"], df_petTOAD[df_petTOAD["WMH_bin"] == "WMH"]["Age"])
    ct_sex = pd.crosstab(df_petTOAD["WMH_bin"], df_petTOAD["Sex"])
    _, p_sex, _, _ = chi2_contingency(ct_sex)
    _, pval_mmse = mannwhitneyu(df_petTOAD[df_petTOAD["WMH_bin"] == "no_WMH"]["MMSE"], df_petTOAD[df_petTOAD["WMH_bin"] == "WMH"]["MMSE"])
    _, pval_edu = mannwhitneyu(df_petTOAD[df_petTOAD["WMH_bin"] == "no_WMH"]["PTEDUCAT"], df_petTOAD[df_petTOAD["WMH_bin"] == "WMH"]["PTEDUCAT"])
    _, pval_wmh = mannwhitneyu(df_petTOAD[df_petTOAD["WMH_bin"] == "no_WMH"]["wmh_log"], df_petTOAD[df_petTOAD["WMH_bin"] == "WMH"]["wmh_log"])

    summary_data = {
        "Age": [f"{wmh_age_50} ({wmh_age_25} - {wmh_age_75})",
            f"{no_wmh_age_50} ({no_wmh_age_25} - {no_wmh_age_75})",
            f"{format_pval_tbl(pval_age)}",
            ],
        
        "Sex n (%)": ["", "", f"{p_sex:.2f}"],
        "Women": [
            f"{wmh_seg_n['F']} ({round(wmh_seg_n['F'] / num_wmh * 100, 2)}%)",
            f"{no_wmh_seg_n['F']} ({round(no_wmh_seg_n['F'] / num_no_wmh * 100, 2)}%)",
            " "
        ],
        "Men": [
            f"{wmh_seg_n['M']} ({round(wmh_seg_n['M'] / num_wmh * 100, 2)}%)",
            f"{no_wmh_seg_n['M']} ({round(no_wmh_seg_n['M'] / num_no_wmh * 100, 2)}%)",
            " "        
        ],
        
        "WMH vol. (log)": [f"{wmh_wmh_log_50:.2f} ({wmh_wmh_log_25:.2f} - {wmh_wmh_log_75:.2f})",
            f"{no_wmh_wmh_log_50:.2f} ({no_wmh_wmh_log_25:.2f} - {no_wmh_wmh_log_75:.2f})",
            f"{format_pval_tbl(pval_wmh)}",
            ],

        "MMSE": [f"{wmh_mmse_50} ({wmh_mmse_25} - {wmh_mmse_75})",
            f"{no_wmh_mmse_50} ({no_wmh_mmse_25} - {no_wmh_mmse_75})",
            f"{format_pval_tbl(pval_mmse)}",
            ],

        "Education (yrs.)": [f"{wmh_edu_50} ({wmh_edu_25} - {wmh_edu_75})",
            f"{no_wmh_edu_50} ({no_wmh_edu_25} - {no_wmh_edu_75})",
            f"{format_pval_tbl(pval_edu)}",
            ],        
        }

    # Create the summary DataFrame
    df_summary = pd.DataFrame(data=summary_data)

    # Print the summary DataFrame
    df_table1 = df_summary.T
    df_table1.columns = [
        f"WMH (n = {num_wmh})",
        f"no WMH (n = {num_no_wmh})",
        "p",
    ]
    reorder = [1, 0, 2]
    df_table1_reordered = df_table1.iloc[:, reorder]
    df_table1_reordered.to_csv(TBL_DIR / "suppl_tbl2_demographics.csv")

    return df_table1_reordered


# Function to format the table of random versus non-random comparison
def format_table_compare_rand_non_rand(df_rand_plot, p_values):
    df = (
        df_rand_plot.groupby(["parm_type", "model_type"])
        .describe()["phfcd_ks"]
        .iloc[:, -4:-1]
    )
    model_name_order = [
        "Homogeneous SDC",
        "Heterogeneous SDC",
        "Homogeneous NDC",
        "Heterogeneous NDC",
    ]
    data_dict = {
        "Model Name": [],
        "KSD Random": [],
        "KSD Non-random": [],
        "p-value": [],
    }

    for model_name in model_name_order:
        if "SDC" in model_name:
            parm_type = "sdc"
        else:
            parm_type = "bif_parm"

        if "Homogeneous" in model_name:
            model_random = "homogeneous_random"
            model_non_random = "homogeneous"
        else:
            model_random = "heterogeneous_random"
            model_non_random = "heterogeneous"

        random_row = df.loc[(parm_type, model_random)]
        non_random_row = df.loc[(parm_type, model_non_random)]

        random_range_str = (
            f"{random_row['50%']:.2f} ({random_row['25%']:.2f}-{random_row['75%']:.2f})"
        )
        non_random_range_str = f"{non_random_row['50%']:.2f} ({non_random_row['25%']:.2f}-{non_random_row['75%']:.2f})"

        p_value = p_values.get(model_name, np.nan)
        if not np.isnan(p_value) and p_value < 0.05:
            p_value_str = f"**{p_value:.3f}**"
        else:
            p_value_str = f"{p_value:.3f}" if not np.isnan(p_value) else "NaN"

        data_dict["Model Name"].append(model_name)
        data_dict["KSD Random"].append(random_range_str)
        data_dict["KSD Non-random"].append(non_random_range_str)
        data_dict["p-value"].append(p_value_str)

    return pd.DataFrame(data_dict)


def create_table_comparison_long(df_long, group, p_values):
    df = (
        df_long.groupby(["parm_type", "model_type"])
        .describe()["phfcd_ks"]
        .iloc[:, -4:-1]
    )

    model_name_order = [
        "Baseline",
        "Homogeneous SDC",
        "Heterogeneous SDC",
        "Homogeneous NDC",
        "Heterogeneous NDC",
    ]
    data_dict = {"Model Name": [], "KSD": [], "p-value": []}

    for model_name in model_name_order:
        if "SDC" in model_name:
            parm_type = "sdc"
        elif "NDC" in model_name:
            parm_type = "bif_parm"
        elif "Baseline" in model_name:
            parm_type = "sdc"

        if "Homogeneous" in model_name:
            model = "homogeneous"
        elif "Heterogeneous" in model_name:
            model = "heterogeneous"
        else:
            model = "baseline"

        row = df.loc[(parm_type, model)]

        range_str = f"{row['50%']:.2f} ({row['25%']:.2f}-{row['75%']:.2f})"

        p_value = p_values.get(model_name, np.nan)
        if not np.isnan(p_value) and p_value < 0.05:
            p_value_str = f"**{p_value:.3f}**"
        elif not np.isnan(p_value) and 0.05 < p_value < 0.06:
            p_value_str = f"{p_value:.3f}"
        else:
            p_value_str = f"{p_value:.2f}" if not np.isnan(p_value) else ""

        data_dict["Model Name"].append(model_name)
        data_dict["KSD"].append(range_str)
        data_dict["p-value"].append(p_value_str)

    df = pd.DataFrame(data_dict)

    # Creating a MultiIndex for the columns
    columns = pd.MultiIndex.from_tuples(
        [("Model Name", ""), (group, "KSD"), (group, "p-value")], names=["", ""]
    )

    # Applying the MultiIndex to the DataFrame
    df.columns = columns

    return df

def create_table_corr_model_improvement(df):
    # Define the model names and their corresponding columns
    model_name_order = {
        "Homogeneous SDC": "phfcd_ks_diff_base_G",
        "Heterogeneous SDC": "phfcd_ks_diff_base_disconn",
        "Homogeneous NDC": "phfcd_ks_diff_base_a",
        "Heterogeneous NDC": "phfcd_ks_diff_base_het"
    }

    # List of demographic variables
    demographic_vars = ["wmh_log", "Age", "MMSE", "PTEDUCAT"]
    demographic_labels = ["WMH Volume (log)", "Age", "MMSE", "Education (yrs)"]

    # Dictionary to store the results
    results = {
        # "Demographic variable": [],
        ("Homogeneous SDC", "r"): [],
        ("Homogeneous SDC", "p-value"): [],
        ("Heterogeneous SDC", "r"): [],
        ("Heterogeneous SDC", "p-value"): [],
        ("Homogeneous NDC", "r"): [],
        ("Homogeneous NDC", "p-value"): [],
        ("Heterogeneous NDC", "r"): [],
        ("Heterogeneous NDC", "p-value"): []
    }

    # Calculate partial correlations and store in the results dictionary
    for x_col, x_label in zip(demographic_vars, demographic_labels):
        # results["Demographic variable"].append(x_label)
        for model_name, model_ratio in model_name_order.items():
            if x_col == "Age":
                pc = partial_corr(data=df, x=x_col, y=model_ratio, method="spearman")
            else:
                pc = partial_corr(data=df, x=x_col, y=model_ratio, covar=["Age"], method="spearman")
            r = pc["r"].values[0]
            p = pc["p-val"].values[0]
            
            results[(model_name, "r")].append(f"{r:.2f}")
            if p < 0.05:
                results[(model_name, "p-value")].append(f"**{p:.3f}**")
            elif 0.05 < p < 0.06:
                results[(model_name, "p-value")].append(f"{p:.3f}")
            else:
                results[(model_name, "p-value")].append(f"{p:.2f}")

    df = pd.DataFrame(results)
    df["Demographic variable"] = demographic_labels
    # Put demographic label as first column
    last_column = df.columns[-1]
    df = df[[last_column] + df.columns[:-1].tolist()]
    return df
