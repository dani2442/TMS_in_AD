#%%
# Import needed packages
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import my_functions as my_func
from phFCD import *
from petTOAD_setup import *


def load_df_arrays(groupname, model):
    df_group = pd.read_csv(
        SIM_GROUP_DIR / groupname / f"group-{groupname}_model-{model}_desc-df-fitting-results.csv",
        index_col=[0],
        header=[0],
    )
    arr_fc_phfcd = np.load(
        SIM_GROUP_DIR
        / groupname
        / f"group-{groupname}_data-simulated_model-{model}_desc-fc-phfcd-arr.npz"
    )
    return df_group, arr_fc_phfcd

########################################################


def get_group_empirical_observables(groupname, grouplist):
    print(f"Calculating the group empirical phfcd for {groupname}...")
    dict_group = {subj: all_fMRI_clean[subj] for subj in grouplist}
    empirical_fc = np.array([my_func.fc(ts) for ts in dict_group.values()])
    empirical_phfcd = np.array([phFCD(ts) for ts in dict_group.values()])
    return empirical_fc, empirical_phfcd

########################################################

def phfcd_parallel_iteration(n_repeats, emp_phfcd, sim_phfcd, n_random_samples):
    random_sample_indices_emp = np.random.RandomState().choice(
        emp_phfcd.shape[0], n_random_samples, replace=True
    )
    emp_phfcd_array_random_sample = emp_phfcd[random_sample_indices_emp].flatten()
    sim_random_sample_indices = np.random.RandomState().choice(
        sim_phfcd.shape[0], n_random_samples, replace=True
    )
    sim_phfcd_array_random_sample = sim_phfcd[sim_random_sample_indices]
    phfcd_ks = [
        my_func.matrix_kolmogorov(
            sim_phfcd_array_random_sample[:, n].flatten(), emp_phfcd_array_random_sample
        )
        for n in range(sim_phfcd_array_random_sample.shape[1])
    ]
    return phfcd_ks

########################################################


def montecarlo_phfcd_parallel(emp_phfcd, sim_phfcd, n_samples=25, repeats=5):
    """
    This function performs Montecarlo simulations to evaluate the difference between empirical and simulated phase functional connectivity dynamics in parallel.

    Args:
        emp_phfcd (arr): the array with all the phase functional connectivity dynamics for the empirical data from patient in the specific group
        sim_phfcd (arr): the array with all the phase functional connectivity dynamics for the simulated data in the group
        n_samples (int): the number of individuals you want in your sub-sample for each iteration (default = 25)
        repeats (int): the number of times you want to repeat the simulations (default = 5 for testing)
        n_workers (int): the number of parallel workers to use (default is None, which means using the maximum available)

    Returns:
        arr_results_ksd (arr): an array with the Kolmogorov-Smirnov distance between the cumulative distribution of the empirical and simulated data
                               out of the n_samples subjects.

    """
    n_repeats = range(repeats)
    with ProcessPoolExecutor(max_workers=150) as executor:
        results = list(
            executor.map(
                partial(
                    phfcd_parallel_iteration,
                    emp_phfcd=emp_phfcd,
                    sim_phfcd=sim_phfcd,
                    n_random_samples=n_samples,
                ),
                n_repeats,
            )
        )
    arr_results_ksd = np.array(results)
    return arr_results_ksd

########################################################


def run_montecarlo_phfcd(groupname, model, empirical_phfcd, n_repeats):
    df, arr_fc_phfcd = load_df_arrays(groupname, model)
    simulated_phfcd = arr_fc_phfcd["phfcd"]
    print(
        f"Now performing Montecarlo simulations of KSD between empirical and simulated data for {groupname} {model}..."
    )
    start_time = time.time()
    arr_res = montecarlo_phfcd_parallel(
        empirical_phfcd, simulated_phfcd, n_samples=25, repeats=n_repeats
    )
    end_time = time.time()
    print(
        f"Done performing Montecarlo for {groupname} {model}! It took {round(end_time - start_time, 3)} seconds"
    )
    return df, arr_res

########################################################

def run_montecarlo_model_comparisons_wmh(groupname, grouplist, n_rep, no_wmh_best_a=None):
    _, emp_phfcd = get_group_empirical_observables(groupname, grouplist)
    df_a, arr_phfcd_ks_a = run_montecarlo_phfcd(
        groupname, "homogeneous_a", emp_phfcd, n_repeats=n_rep
    )
    df_a["a"] = [float(df_a["a"][i].split(" ")[1]) for i in range(len(df_a["a"]))]
    df_G, arr_phfcd_ks_G = run_montecarlo_phfcd(
        groupname, "homogeneous_G", emp_phfcd, n_repeats=n_rep
    )

    data_to_save = {
        "a": arr_phfcd_ks_a,
        "G": arr_phfcd_ks_G,
    }

    np.savez_compressed(
        SIM_GROUP_DIR
        / groupname
        / f"group-{groupname}_data-simulated_model-all_desc-ksd-comparison-with-empirical-montecarlo-{n_rep}-repeats.npz",
        **data_to_save,
    )

    # When modeling CN with WMH, the last value of the homogeneous_a model represents the best a found in the CN no WMH group by definition because we start from there
    # and we move toward more negative values (since we have negative values, the first simulated will be the most negative and then we move on to the ones closer to 0).
    # So if we want to see how the baseline model (the best one fitted on CN without WMH) performs in CN WMH, the index is the last one.
    if groupname == "CN_WMH":
        base_idx = arr_phfcd_ks_a.shape[1] - 1
    # When modeling MCI with WMH it is a bit different, because we start our range of explorations still with the best a found in CN without WMH, but this might not be
    # the best in MCI without WMH (and we want to compare MCI with to MCI without WMH). So no_wmh_best_a represents the best a that is found in MCI without WMH, then we
    # get the corresponding index from the df (and this will be the same as the best index in the phfcd array)
    elif groupname == "MCI_WMH":
        base_idx = df_a[df_a["a"] == no_wmh_best_a].index.values[0]

    best_idx_a = arr_phfcd_ks_a.mean(
        axis=0
    ).argmin()  # Get the a where the mean phfcd is the best
    best_idx_G = arr_phfcd_ks_G.mean(
        axis=0
    ).argmin()  # Get the G where the mean phfcd is the best
    dict_best_values = {
        groupname: {
            "best_a-idx_model-homogeneous_a": best_idx_a,
            "best_G-idx_model-homogeneous_G": best_idx_G,
            "best_a_model-homogeneous_a": df_a["a"][best_idx_a],
            "best_G_model-homogeneous_G": df_G["K_gl"][best_idx_G],
        }
    }

    df_best_values = pd.DataFrame(dict_best_values)
    df_best_values.to_csv(
        SIM_GROUP_DIR
        / groupname
        / f"group-{groupname}_data-empirical_model-all_desc-df-best-values-and-indices.csv"
    )

    base_box = arr_phfcd_ks_a[
        :, base_idx
    ]  # Only get the distribution of values at the best index for the specific model for plotting as boxplots
    best_a_box = arr_phfcd_ks_a[:, best_idx_a]
    best_G_box = arr_phfcd_ks_G[:, best_idx_G]
    df_compare_boxplot = pd.DataFrame(
        {
            "Base": base_box,
            "Homog. a": best_a_box,
            "Homog. G": best_G_box,
        }
    )
    df_compare_boxplot_melted = df_compare_boxplot.melt().rename(
        columns={"variable": "model", "value": "phfcd_ks"}
    )
    df_compare_boxplot_melted.to_csv(
        SIM_GROUP_DIR
        / groupname
        / f"group-{groupname}_data-simulated_model-all_desc-df-long-for-boxplots.csv"
    )
    return (
        df_compare_boxplot_melted,
        best_idx_a,
        best_idx_G,
    )

########################################################################################################################################################################

def run_montecarlo_model_comparisons_mci_no_wmh(groupname, grouplist, n_rep, no_wmh_best_a=None):
    _, emp_phfcd = get_group_empirical_observables(groupname, grouplist)
    df_a, arr_phfcd_ks_a = run_montecarlo_phfcd(
        groupname, "homogeneous_a", emp_phfcd, n_repeats=n_rep
    )
    df_a["a"] = [float(df_a["a"][i].split(" ")[1]) for i in range(len(df_a["a"]))]
    data_to_save = {
        "a": arr_phfcd_ks_a,
    }

    np.savez_compressed(
        SIM_GROUP_DIR
        / groupname
        / f"group-{groupname}_data-simulated_model-all_desc-ksd-comparison-with-empirical-montecarlo-{n_rep}-repeats.npz",
        **data_to_save,
    )

    # When modeling CN with WMH, the last value of the homogeneous_a model represents the best a found in the CN no WMH group by definition because we start from there
    # and we move toward more negative values (since we have negative values, the first simulated will be the most negative and then we move on to the ones closer to 0).
    # So if we want to see how the baseline model (the best one fitted on CN without WMH) performs in CN WMH, the index is the last one.
    if groupname == "CN_WMH":
        base_idx = arr_phfcd_ks_a.shape[1] - 1
    # When modeling MCI with WMH it is a bit different, because we start our range of explorations still with the best a found in CN without WMH, but this might not be
    # the best in MCI without WMH (and we want to compare MCI with to MCI without WMH). So no_wmh_best_a represents the best a that is found in MCI without WMH, then we
    # get the corresponding index from the df (and this will be the same as the best index in the phfcd array)
    elif groupname == "MCI_WMH":
        base_idx = df_a[df_a["a"] == no_wmh_best_a].index.values[0]

    best_idx_a = arr_phfcd_ks_a.mean(
        axis=0
    ).argmin()  # Get the a where the mean phfcd is the best
    dict_best_values = {
        groupname: {
            "best_a-idx_model-homogeneous_a": best_idx_a,
            "best_a_model-homogeneous_a": df_a["a"][best_idx_a],
        }
    }

########################################################

def get_best_G_cn_no_wmh():
    df_cn_no_wmh_G, _ = load_df_arrays("CN_no_WMH", "homogeneous_G")
    best_G_idx = df_cn_no_wmh_G["phfcd_ks"].argmin()
    best_G = df_cn_no_wmh_G["K_gl"][best_G_idx]
    return best_G, best_G_idx

########################################################

def get_best_a_mci_no_wmh(n_rep):
    _, emp_phfcd = get_group_empirical_observables("MCI_no_WMH", MCI_no_WMH)
    df_a, arr_phfcd_ks_a = run_montecarlo_phfcd(
        "MCI_no_WMH", "homogeneous_a", emp_phfcd, n_repeats=n_rep
    )
    df_a["a"] = [float(df_a["a"][i].split(" ")[1]) for i in range(len(df_a["a"]))]
    data_to_save = {
        "a": arr_phfcd_ks_a,
    }
    np.savez_compressed(
        SIM_GROUP_DIR
        / "MCI_no_WMH"
        / f"group-MCI-no-WMH_data-simulated_model-homogeneous-a_desc-ksd-comparison-with-empirical-montecarlo-{n_rep}-repeats.npz",
        **data_to_save,
    )
    best_idx = np.argmin(arr_phfcd_ks_a.mean(axis=0))
    values_best = arr_phfcd_ks_a[:, best_idx]
    stat, pval = mannwhitneyu(values_best, arr_phfcd_ks_a[:,-1])
    if pval < 0.05:
        best_idx_no_wmh = best_idx
        best_a_no_wmh = df_a.loc[best_idx, "a"]
    else:
        best_idx_no_wmh = df_a.shape[0]
        best_a_no_wmh = -0.02

    return best_a_no_wmh, best_idx_no_wmh


def save_dict_best_values(group1_best_G_idx,
                          group2_best_a_idx,
                          group1_best_idx_a,
                          group1_best_idx_G,
                          group2_best_idx_a,
                          group2_best_idx_G,
                          ):
    dict_best = {
        "cn_no_wmh_G_idx": group1_best_G_idx,
        "mci_no_wmh_a_idx": group2_best_a_idx,
        "idx_a_cn_wmh": group1_best_idx_a,
        "idx_G_cn_wmh": group1_best_idx_G,
        "idx_a_mci_wmh": group2_best_idx_a,
        "idx_G_mci_wmh": group2_best_idx_G,
    }

    df_best = pd.DataFrame(dict_best, index=[0])
    df_best.to_csv(SIM_GROUP_DIR / "group-all_data-simulated_model-all_df-best_indices.csv")

########################################################

def main():
    cn_no_wmh_best_G, cn_no_wmh_best_G_idx = get_best_G_cn_no_wmh()
    mci_no_wmh_best_a, mci_no_wmh_best_a_idx = get_best_a_mci_no_wmh(n_rep = n_simulations)
    (
        df_all_models_cn_wmh,
        cn_wmh_best_idx_a,
        cn_wmh_best_idx_G,
    ) = run_montecarlo_model_comparisons_wmh("CN_WMH", CN_WMH, n_rep=n_simulations)
    (
        df_all_models_mci_wmh,
        mci_wmh_best_idx_a,
        mci_wmh_best_idx_G,
    ) = run_montecarlo_model_comparisons_wmh(
        "MCI_WMH", MCI_WMH, n_rep=n_simulations, no_wmh_best_a=mci_no_wmh_best_a
    )
    
    save_dict_best_values(cn_no_wmh_best_G_idx,
                          mci_no_wmh_best_a_idx,
                          cn_wmh_best_idx_a,
                          cn_wmh_best_idx_G,
                          mci_wmh_best_idx_a,
                          mci_wmh_best_idx_G,
                          )


n_simulations = 1000

if __name__ == "__main__":
    main()