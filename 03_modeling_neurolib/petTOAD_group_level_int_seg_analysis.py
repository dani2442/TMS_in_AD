#%%
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import my_functions as my_func
from phFCD import *
from petTOAD_setup import *
from integration import IntegrationFromFC_Fast
from segregation import computeSegregation

########################################################################################################################################################################

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


########################################################################################################################################################################

def get_group_empirical_observables(groupname, grouplist):
    print(f"Calculating the group empirical phfcd for {groupname}...")
    dict_group = {subj: all_fMRI_clean[subj] for subj in grouplist}
    empirical_fc = np.array([my_func.fc(ts) for ts in dict_group.values()])
    empirical_phfcd = np.array([phFCD(ts) for ts in dict_group.values()])
    return empirical_fc, empirical_phfcd

########################################################################################################################################################################

def integration_parallel(n_repeats: int, arr_fc: np.ndarray, n_random_samples: int):
    """
    Calculate integration from a subset of n_random_samples out of the whole array of functional connectivities for a group. To be used with run_int_seg_parallel()

    Parameters:
    - n_repeats (int): Number of Monte-Carlo runs to perform.
    - arr_fc (np.ndarray): 3D array of functional connectivity data for subjects in the chosen group.
    - n_random_samples (int): Number of subjects to randomly sample in each simulation.

    Returns:
    - integration: Integration measure
    """
    # Randomly sample n subjects
    random_sample_indices = np.random.RandomState().choice(
        arr_fc.shape[0], n_random_samples, replace=False
    )
    random_sample_data = arr_fc[random_sample_indices]
    # Calculate average connectivity for the random sample
    average_connectivity = np.abs(random_sample_data.mean(axis=0))
    integration = IntegrationFromFC_Fast(average_connectivity)
    return integration

########################################################################################################################################################################


def segregation_parallel(n_repeats: int, arr_fc: np.ndarray, n_random_samples: int):
    """
    Calculate segregation from a subset of n_random_samples out of the whole array of functional connectivities for a group. To be used with run_int_seg_parallel()

    Parameters:
    - n_repeats (int): Number of Monte Carlo simulations to perform.
    - arr_fc (np.ndarray): 3D array of functional connectivity data for subjects.
    - n_random_samples (int): Number of subjects to randomly sample in each simulation.

    Returns:
    - segregation (float): Segregation measure.
    """
    # Randomly sample n subjects
    random_sample_indices = np.random.RandomState().choice(
        arr_fc.shape[0], n_random_samples, replace=False
    )
    random_sample_data = arr_fc[random_sample_indices]
    # Calculate average connectivity for the random sample
    average_connectivity = np.abs(random_sample_data.mean(axis=0))
    segregation = computeSegregation(average_connectivity)[0]
    return segregation

########################################################


def run_int_seg_parallel(
    groupname: str, arr_fc: np.ndarray, repeats: int, n_samples: int = 25
):
    """
    Run parallel Monte-Carlo simulations for integration and segregation.

    Parameters:
    - groupname (str): Name of the group for which simulations are performed.
    - arr_fc (np.ndarray): 3D array of functional connectivity data for subjects.
    - repeats (int): Number of Monte Carlo simulations to perform.
    - n_samples (int): Number of subjects to randomly sample in each simulation. Default is 25.

    Returns:
    - (integrations, segregations) (tuple): Tuple of lists containing integration and segregation measures.
    """

    print(
        f"Calculating Montecarlo simulations for integration and segregation for group {groupname} for {repeats} times..."
    )
    n_repeats = range(repeats)
    with ProcessPoolExecutor(max_workers=350) as executor:
        integrations = list(
            executor.map(
                partial(
                    integration_parallel, arr_fc=arr_fc, n_random_samples=n_samples
                ),
                n_repeats,
            )
        )
        segregations = list(
            executor.map(
                partial(
                    segregation_parallel, arr_fc=arr_fc, n_random_samples=n_samples
                ),
                n_repeats,
            )
        )
    return integrations, segregations

########################################################

def create_and_save_int_seg_df(integrations, segregations, n_repeats):
    """
    Create DataFrames from integration and segregation results and save them to CSV files.

    Parameters:
    - integrations (Dict[str, List[float]]): Dictionary containing integration results for different groups.
    - segregations (Dict[str, List[float]]): Dictionary containing segregation results for different groups.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing integration and segregation data.
    """
    df_int = pd.DataFrame(integrations).melt(var_name="group", value_name="Integration")
    df_seg = pd.DataFrame(segregations).melt(var_name="group", value_name="Segregation")
    df_int.to_csv(
        SIM_GROUP_DIR
        / f"group-all_data-empirical_desc-df-integration-montecarlo-{n_repeats}-repeats.csv"
    )
    df_seg.to_csv(
        SIM_GROUP_DIR
        / f"group-all_data-empirical_desc-df-segregation-montecarlo-{n_repeats}-repeats.csv"
    )
    return df_int, df_seg

########################################################

def run_montecarlo_int_seg_empirical(n_rep: int):
    """
    Run Monte-Carlo simulations for integration and segregation for different groups, save results, and plot comparisons.

    Parameters:
    - n_repeats (int): Number of Monte Carlo simulations to perform.

    Returns:
    - None
    """
    dict_groups = {
        "CN_no_WMH": CN_no_WMH,
        "CN_WMH": CN_WMH,
        "MCI_no_WMH": MCI_no_WMH,
        "MCI_WMH": MCI_WMH,
    }
    integrations = {}
    segregations = {}
    for groupname, grouplist in dict_groups.items():
        group_emp_fc, _ = get_group_empirical_observables(groupname, grouplist)
        integrations[groupname], segregations[groupname] = run_int_seg_parallel(
            groupname, group_emp_fc, repeats=n_rep
        )
    df_int, df_seg = create_and_save_int_seg_df(integrations, segregations, n_rep)

########################################################

def prepare_df_for_plotting_int_seg(dict_int_or_seg, obs_name, n_sim):
    df = pd.DataFrame.from_dict(dict_int_or_seg)
    df["CN_no_WMH"]["homogeneous_a"] = np.repeat(np.nan, n_sim)
    df["MCI_no_WMH"]["homogeneous_G"] = np.repeat(np.nan, n_sim)
    df_exploded = df.explode(["CN_no_WMH", "MCI_no_WMH", "CN_WMH", "MCI_WMH"])
    df_long = pd.melt(
        df_exploded.reset_index(),
        id_vars=["index"],
        var_name="group",
        value_name=obs_name,
    ).dropna()
    df_long = df_long.rename(columns={"index": "model_type"})
    df_long["group_model"] = df_long["group"] + "_" + df_long["model_type"]
    df_long[obs_name] = df_long[obs_name].astype(
        float
    )  # needs to reconvert to float otherwise statannotations complains
    return df_long

########################################################

def run_montecarlo_int_seg_simulated(
    n_rep: int,
    cn_no_wmh_G_idx: int,
    mci_no_wmh_a_idx: int,
    idx_a_cn_wmh: int,
    idx_G_cn_wmh: int,
    idx_a_mci_wmh: int,
    idx_G_mci_wmh: int,
):
    """
    Run Monte-Carlo simulations for integration and segregation for different groups of simulated data, save results, and plot comparisons.

    Parameters:
    - n_repeats (int): Number of Monte Carlo simulations to perform.

    Returns:
    - None
    """
    dict_groups = {
        "CN_no_WMH": ["homogeneous_G"],
        "MCI_no_WMH": ["homogeneous_a"],
        "CN_WMH": ["homogeneous_a", "homogeneous_G"], 
        "MCI_WMH": ["homogeneous_a", "homogeneous_G"] 
    }

    integrations = {}
    segregations = {}
    for groupname, list_model in dict_groups.items():
        integrations[groupname] = {}
        segregations[groupname] = {}
        for model in list_model:
            _, arr_fc_phfcd = load_df_arrays(groupname, model)
            group_sim_fc = arr_fc_phfcd["fc"]
            if groupname == "CN_no_WMH":
                group_sim_fc_best = group_sim_fc[:, cn_no_wmh_G_idx]
            elif groupname == "MCI_no_WMH":
                group_sim_fc_best = group_sim_fc[:, mci_no_wmh_a_idx]
            elif groupname == "CN_WMH":
                if model == "homogeneous_a":
                    group_sim_fc_best = group_sim_fc[:, idx_a_cn_wmh]
                if model == "homogeneous_G":
                    group_sim_fc_best = group_sim_fc[:, idx_G_cn_wmh]
            elif groupname == "MCI_WMH":
                if model == "homogeneous_a":
                    group_sim_fc_best = group_sim_fc[:, idx_a_mci_wmh]
                if model == "homogeneous_G":
                    group_sim_fc_best = group_sim_fc[:, idx_G_mci_wmh]
            (
                integrations[groupname][model],
                segregations[groupname][model],
            ) = run_int_seg_parallel(groupname, group_sim_fc_best, repeats=n_rep)
    df_int_long = prepare_df_for_plotting_int_seg(
        integrations, "Integration", n_sim=n_rep
    )
    df_seg_long = prepare_df_for_plotting_int_seg(
        segregations, "Segregation", n_sim=n_rep
    )
    df_int_long.to_csv(
        SIM_GROUP_DIR
        / f"group-all_data-simulated_model-all_desc-df-integration-montecarlo-{n_rep}-repeats.csv"
    )
    df_seg_long.to_csv(
        SIM_GROUP_DIR
        / f"group-all_data-simulated_model-all_desc-df-segregation-montecarlo-{n_rep}-repeats.csv"
    )
    # plot_all_int_seg_comparisons_simulated(df_int_long, df_seg_long)
    return df_int_long, df_seg_long

def get_best_indices():
    df_best = pd.read_csv(SIM_GROUP_DIR / "group-all_data-simulated_model-all_df-best_indices.csv", index_col = 0)
    cn_no_wmh_best_G_idx = df_best.loc[0, "cn_no_wmh_G_idx"]
    mci_no_wmh_best_a_idx = df_best.loc[0, "mci_no_wmh_a_idx"]
    cn_wmh_best_idx_a = df_best.loc[0, "idx_a_cn_wmh"]
    cn_wmh_best_idx_G = df_best.loc[0, "idx_G_cn_wmh"]
    mci_wmh_best_idx_a = df_best.loc[0, "idx_a_mci_wmh"]
    mci_wmh_best_idx_G = df_best.loc[0, "idx_G_mci_wmh"]
    return cn_no_wmh_best_G_idx, mci_no_wmh_best_a_idx, cn_wmh_best_idx_a, cn_wmh_best_idx_G, mci_wmh_best_idx_a, mci_wmh_best_idx_G


def main():
    cn_no_wmh_best_G_idx, mci_no_wmh_best_a_idx, cn_wmh_best_idx_a, cn_wmh_best_idx_G, mci_wmh_best_idx_a, mci_wmh_best_idx_G = get_best_indices()
    run_montecarlo_int_seg_empirical(n_rep=n_simulations)
    run_montecarlo_int_seg_simulated(
        n_rep=n_simulations,
        cn_no_wmh_G_idx=cn_no_wmh_best_G_idx,
        mci_no_wmh_a_idx=mci_no_wmh_best_a_idx,
        idx_a_cn_wmh=cn_wmh_best_idx_a,
        idx_G_cn_wmh=cn_wmh_best_idx_G,
        idx_a_mci_wmh=mci_wmh_best_idx_a,
        idx_G_mci_wmh=mci_wmh_best_idx_G,
    )

n_simulations = 1000
if __name__ == "__main__":
    main()
