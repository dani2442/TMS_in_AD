# %% Initial imports

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statannotations.Annotator import Annotator
from neurolib.utils import paths
from neurolib.utils import pypetUtils as pu
from neurolib.utils.parameterSpace import ParameterSpace

import my_functions as my_func
from phFCD import *
from petTOAD_parameter_setup import *
from petTOAD_setup import *
from segregation import computeSegregation
from integration import IntegrationFromFC_Fast


# %%
SIM_DIR = RES_DIR / "final_simulations_log_2023-11-23"
A_DIR = SIM_DIR / "a-weight_ws_-0.1-0.0_bs_-0.05-0.0"
A_RAND_DIR = SIM_DIR / "a-weight_ws_-0.1-0.0_bs_-0.05-0.0_random"
G_DIR = SIM_DIR / "G-weight_ws_-1.0-0.0_bs_-0.5-0.0"
G_RAND_DIR = SIM_DIR / "G-weight_ws_-1.0-0.0_bs_-0.5-0.0_random"
SC_DIR = SIM_DIR / "sc_disconn"
SC_RAND_DIR = SIM_DIR / "sc_disconn_random"
HET_DIR = SIM_DIR / "heterogeneous_ws_-0.1-0.0_bs_-0.05-0.0"
HET_RAND_DIR = SIM_DIR / "heterogeneous_ws_-0.1-0.0_bs_-0.05-0.0_random"

# %%
def calculate_results_from_bolds_subject(bold_arr, n_sim, n_parms, n_nodes):
    # Create a new array to store the FC and phFCD values with the same shape as bold array
    fc_array = np.zeros([n_sim, n_parms, n_nodes, n_nodes])
    phfcd_array = np.zeros([n_sim, n_parms, 18336])

    # Iterate over each element in the bold array
    for i in range(n_sim):
        for j in range(n_parms):
            print(
                f"Now calculating results from the {i} simulation for parameter {j}..."
            )
            # Get the current timeseries
            timeseries = bold_arr[i, j].squeeze()

            # Perform FC and phFCD analysis
            if np.isnan(timeseries).any():
                print("Simulation has some nans, aborting!")
                continue
            else:
                print("Calculating FC..")
                fc_value = my_func.fc(timeseries)
                print("Calculating phFCD")
                phfcd_value = phFCD(timeseries)
                # Store the FC and phFCD value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
                phfcd_array[i, j] = phfcd_value
    return fc_array, phfcd_array


def load_or_gather_results_from_repeated_simulations(
    subj, res_dir, filename, model_type, ws=None, bs=None, disconn=False
):
    try:
        sim_res = np.load(f"{res_dir}/group-single-subj-{subj}_data-simulated_model-{model_type}_desc-fc-phfcd-arr.npz")
        emp_res = np.load(f"{res_dir}/group-single-subj-{subj}_data-empirical_model-{model_type}_desc-fc-phfcd-arr.npz")
        sim_fc = sim_res["sim_fc"]
        emp_fc = emp_res["emp_fc"]
    except:
        trajs = pu.getTrajectorynamesInFile(f"{res_dir}/{filename}")
        big_list = []
        for traj in trajs:
            traj_list = []
            tr = pu.loadPypetTrajectory(f"{res_dir}/{filename}", traj)
            run_names = tr.f_get_run_names()
            n_run = len(run_names)
            ns = range(n_run)
            for i in ns:
                r = pu.getRun(i, tr)
                traj_list.append(r["BOLD"])
            big_list.append(traj_list)
        bold_arr = np.array(big_list)
        n_sim = len(trajs)
        if disconn:
            n_parms = len(ws)
        else:
            n_parms = len(ws) * len(bs)
        fc_array, phfcd_array = calculate_results_from_bolds_subject(
            bold_arr, n_sim, n_parms, n_nodes
        )
        timeseries = all_fMRI_clean[subj]
        emp_fc = my_func.fc(timeseries)
        print("Calculating phFCD")
        emp_phfcd = phFCD(timeseries)
        # Get the average fc across the n simulations
        sim_fc = fc_array.mean(axis=0)
        np.savez_compressed(f"{res_dir}/group-single-subj-{subj}_data-simulated_model-{model_type}_desc-fc-phfcd-arr.npz", sim_fc = sim_fc, sim_phfcd = phfcd_array)
        np.savez_compressed(f"{res_dir}/group-single-subj-{subj}_data-empirical_model-{model_type}_desc-fc-phfcd-arr.npz", emp_fc = emp_fc, emp_phfcd = emp_phfcd)
    return sim_fc, emp_fc #fc_pearson, phfcd_ks

def process_subject(subj, model, df_best_combo, bs, ws):
    if subj in CN_WMH:
        w = df_best_combo.loc[model, "best_w_cn"]
    elif subj in MCI_WMH:
        w = df_best_combo.loc[model, "best_w_mci"]

    if model != "disconnectivity":
        b = df_best_combo.loc[model, "best_b_all"]
        b_w = [(round(b, 3), round(w, 3)) for w in ws for b in bs]
        df = pd.DataFrame(data=b_w, columns=["b", "w"])
        best_idx = df[(df["b"] == b) & (df["w"] == w)].index.values[0]

    else:
        df = pd.DataFrame(data=ws, columns=["w"])
        best_idx = df[df["w"] == w].index.values[0]
    return best_idx

def compute_int_seg(best_index, empirical_fc, simulated_fc, also_sim):
    
    abs_emp_fc = np.abs(empirical_fc)
    emp_segregation = computeSegregation(abs_emp_fc)[0]
    emp_integration = IntegrationFromFC_Fast(abs_emp_fc)
    if also_sim:
        abs_sim_fc = np.abs(simulated_fc[best_index])
        sim_segregation = computeSegregation(abs_sim_fc)[0]
        sim_integration = IntegrationFromFC_Fast(abs_sim_fc)
        return sim_integration, emp_integration, sim_segregation, emp_segregation
    return emp_integration, emp_segregation

def lin_reg(df_int_seg, x, y):
    X = sm.add_constant(df_int_seg[x])
    Y = df_int_seg[y]
    # Fit the model
    model = sm.OLS(Y, X).fit()
    return model.params[x], model.params['const'], model.pvalues[x]

def plot_sim_int_subj_level(df_int_seg, axs, nrow, group):
    nrow_str = str(nrow)
    colors = {"0": "tab:orange",
              "1": "tab:blue",
              "2": "tab:green",}
    axs_row = axs[nrow, :]
    sns.regplot(data = df_int_seg, x = "sim_int", y = "emp_int", ax = axs_row[2], color = colors[nrow_str])
    sns.regplot(data = df_int_seg, x = "sim_seg", y = "emp_seg", ax = axs_row[3], label = group, color = colors[nrow_str])
    x_int, const_int, pval_int = lin_reg(df_int_seg, x = "sim_int", y = "emp_int")
    x_seg, const_seg, pval_seg = lin_reg(df_int_seg, x = "sim_seg", y = "emp_seg")
    title_int = f"y = {x_int:.2f}*sim_int + {const_int:.2f}, pval = {pval_int:.3f}"
    title_seg = f"y = {x_seg:.2f}*sim_seg + {const_seg:.2f}, pval = {pval_seg:.3f}"
    axs_row[2].set_title(title_int)
    axs_row[3].set_title(title_seg)
    axs_row[2].set_xlabel("Simulated integration")
    axs_row[2].set_ylabel("Empirical integration")
    axs_row[3].set_xlabel("Simulated segregation")
    axs_row[3].set_ylabel("Empirical segregation")
    axs_row[3].legend()

def lin_reg_parm(df, group, x, y):
    X = pd.to_numeric(df[x])
    Y = pd.to_numeric(df[y])
    # Use statsmodels to calculate p-values for the coefficients
    X_with_intercept = sm.add_constant(
        X
    )  # Add a constant (intercept) term to the features
    model = sm.OLS(
        Y, X_with_intercept
    ).fit()  # Fit an OLS (ordinary least squares) model
    # Get the coefficients (w) and intercept (b)
    lr_slope = model.params[x]
    lr_int = model.params["const"]
    r2 = model.rsquared
    p_slope = model.pvalues[x]

    return group, lr_slope, lr_int, r2, p_slope

def regplot_with_lr(group, df, x, y, ax, color=None):
    if group != "All":
        df = df[df["Group_bin_Fazekas"] == group].copy()
    _, lr_slope, lr_int, r2, p_slope = lin_reg_parm(df, group, x, y)
    dict_names = {"wmh_load": "WMH log",
                  "WMH_load_subj_space": "WMH (mm^3)",
                  "emp_int": "Empirical integration",
                  "sim_int": "Simulated integration",
                  "emp_seg": "Empirical segregation",
                  "sim_seg": "Simulated segregation",
                  "MMSE": "MMSE",
                  "PTEDUCAT": "Education (yrs)",
                  "Age": "Age",
                  }
    if group == "All":
        sns.regplot(data=df, x=x, y=y, ax=ax, color = color)
        groupname = group
    else:
        groupname = f"{group.split('_')[0]} {group.split('_')[1]}"
        sns.regplot(
            data=df[df["Group_bin_Fazekas"] == group],
            x=x,
            y=y,
            ax=ax,
            color=color,
        )
    ax.set_ylabel(f"{dict_names[y]}")
    ax.set_xlabel(f"{dict_names[x]}")
    ax.set_title(f"{groupname}, {y} = {lr_slope:.2f} * {x} + {lr_int:.2f}, p = {p_slope:.3f}", fontsize = 8)

def plot_int_or_seg(df: pd.DataFrame, obs: str, model_type: str, ax: plt.Axes):
    """
    Plot boxplots for integration or segregation measures with statistical comparisons among Fazekas groups

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data for plotting.
    - obs (str): Name of the observation variable (e.g., "Integration" or "Segregation").
    - ax (plt.Axes): Matplotlib Axes object to draw the plot onto.

    Returns:
    - None
    """
    dict_axes = {"emp_int": "Empirical integration",
                 "emp_seg": "Empirical segregation"}
    order = ["CN_no_WMH", "CN_WMH", "MCI_no_WMH", "MCI_WMH"]
    # Choose the combinations that you want to test..
    pairs = [
        ("CN_no_WMH", "CN_WMH"),
        ("MCI_no_WMH", "MCI_WMH"),
    ]
    f = sns.boxplot(data=df, x="Group_bin_Fazekas", y=obs, ax=ax, order=order)
    annotator = Annotator(f, pairs, data=df, x="Group_bin_Fazekas", y=obs, order=order)
    # Choose the type of statistical test to perform
    annotator.configure(
        test="Mann-Whitney", text_format="star", loc="inside", verbose=1
    )
    _, results = annotator.apply_and_annotate()
    ax.set_xticks([0, 1, 2, 3], order)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(["CU no WMH", "CU WMH", "MCI no WMH", "MCI WMH"])
    ax.set_xlabel("Group")
    ax.set_ylabel(dict_axes[obs])
    
def process_int_seg_subjectwise(df, dict_model, model_type, bad_subj, good_subjs):
        if model_type == "disconnectivity":
            disconn = True
        else:
            disconn = False
        try:
            df_int_seg_all = pd.read_csv(SIM_DIR / f"group-all-single-subjects_data-both_model-{model_type}_desc-sim-vs-emp-int-seg.csv", index_col=0)
        except:
            # I use this dictionary just in case we also want to test integration and segregation for other models
            
            list_sim_int = []
            list_emp_int = []
            list_sim_seg = []
            list_emp_seg = []
            for i, subj in enumerate(df_petTOAD["PTID"]):
                print(f"{i+1}/{len(df_petTOAD['PTID'])}")
                if subj in bad_subj:
                    continue
                elif subj in good_subjs:
                    bs = dict_model_bs_ws[model_type]["bs"]
                    ws = dict_model_bs_ws[model_type]["ws"]
                    res_dir = dict_model_bs_ws[model_type]["res_dir"]
                    filename = subj + dict_model_bs_ws[model_type]["suffix"]
                    sim_fc, emp_fc = load_or_gather_results_from_repeated_simulations(subj, res_dir, filename, model_type, ws=ws, bs=bs, disconn=disconn)
                    best_idx = process_subject(subj, model_type, df_best_combinations, bs, ws)
                    sim_int, emp_int, sim_seg, emp_seg = compute_int_seg(best_idx, emp_fc, sim_fc, also_sim = True)
                else:
                    sim_int = np.nan
                    sim_seg = np.nan
                    emp_fc = my_func.fc(all_fMRI_clean[subj])
                    emp_int, emp_seg = compute_int_seg(best_index=None, empirical_fc=emp_fc, simulated_fc=None, also_sim = False)
            
                list_sim_int.append(sim_int)
                list_sim_seg.append(sim_seg)
                list_emp_int.append(emp_int)
                list_emp_seg.append(emp_seg)

            df_int_seg_all = pd.DataFrame({"PTID": df_petTOAD["PTID"],
                                "sim_int": list_sim_int,
                                "emp_int": list_emp_int,
                                "sim_seg": list_sim_seg,
                                "emp_seg": list_emp_seg}
                                )
            df_int_seg_all.to_csv(SIM_DIR / f"group-all-single-subjects_data-both_model-{model_type}_desc-sim-vs-emp-int-seg.csv")
        return df_int_seg_all
#%%
df_petTOAD = pd.read_csv(RES_DIR / "df_petTOAD.csv", index_col=0)
df_best_combinations = pd.read_csv(SIM_DIR / "group-all_data-simulated_model-all_desc-best-b-w-combinations.csv", index_col=0)
bad_subj = pd.read_csv(SIM_DIR / "outlier_bad_fitting.csv", index_col=0)
good_subjs = [subj for subj in subjs_to_sim if subj not in bad_subj["PTID"].values]
dict_model_bs_ws = {"homogeneous_G": {"res_dir": G_DIR, "suffix": "_homogeneous_G-weight_model.hdf","bs": bs_G, "ws": ws_G}}

for model_type in ["homogeneous_G", "disconnectivity"]:
    df_int_seg_all = process_int_seg_subjectwise(df_petTOAD, dict_model_bs_ws, model_type, bad_subj, good_subjs)


df_int_seg_petTOAD = pd.merge(df_petTOAD, df_int_seg_all[["PTID", "emp_int", "emp_seg", "sim_int", "sim_seg"]], on = "PTID")
df_int_seg_petTOAD_sim = df_int_seg_petTOAD[df_int_seg_petTOAD["PTID"].isin(good_subjs)].copy()
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 10))
regplot_with_lr("All", df_int_seg_petTOAD_sim, "sim_int", "emp_int", axs[0, 0], color="tab:orange")
regplot_with_lr("All", df_int_seg_petTOAD_sim, "sim_seg", "emp_seg", axs[1, 0], color="tab:orange")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD_sim, "sim_int", "emp_int", axs[0, 1], color="tab:blue")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD_sim, "sim_seg", "emp_seg", axs[1, 1], color="tab:blue")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD_sim, "sim_int", "emp_int", axs[0, 2], color="tab:green")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD_sim, "sim_seg", "emp_seg", axs[1, 2], color="tab:green")
fig.tight_layout()
fig.savefig(FIG_DIR / f"group-all-single-subjects_data-both_model-{model_type}_desc-sim-vs-emp-int-seg.png")

fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 10))
regplot_with_lr("All", df_int_seg_petTOAD_sim, "WMH_load_subj_space", "emp_int", axs[0, 0], color="tab:orange")
regplot_with_lr("All", df_int_seg_petTOAD_sim, "WMH_load_subj_space", "emp_seg", axs[1, 0], color="tab:orange")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD_sim, "WMH_load_subj_space", "emp_int", axs[0, 1], color="tab:blue")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD_sim, "WMH_load_subj_space", "emp_seg", axs[1, 1], color="tab:blue")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD_sim, "WMH_load_subj_space", "emp_int", axs[0, 2], color="tab:green")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD_sim, "WMH_load_subj_space", "emp_seg", axs[1, 2], color="tab:green")
fig.tight_layout()
fig.savefig(FIG_DIR / f"group-all-single-subjects_data-empirical-model-None_desc-wmh-vs-emp-int-seg.png")
#%%
fig, axs = plt.subplots(nrows=1, ncols = 2, figsize = (10, 5))
plot_int_or_seg(df_int_seg_petTOAD, "emp_int", "empirical", axs[0])
plot_int_or_seg(df_int_seg_petTOAD, "emp_seg", "empirical", axs[1])
fig.tight_layout()

fig, axs = plt.subplots(nrows=2, ncols = 3, figsize = (15, 10))
regplot_with_lr("All", df_int_seg_petTOAD, "emp_int", "MMSE", axs[0, 0], "tab:orange")
regplot_with_lr("All", df_int_seg_petTOAD, "emp_seg", "MMSE", axs[1, 0], "tab:orange")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD, "emp_int", "MMSE", axs[0, 1], "tab:blue")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD, "emp_seg", "MMSE", axs[1, 1], "tab:blue")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD, "emp_int", "MMSE", axs[0, 2], "tab:green")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD, "emp_seg", "MMSE", axs[1, 2], "tab:green")
fig.tight_layout()

#%%
fig, axs = plt.subplots(nrows=2, ncols = 3, figsize = (15, 10))
regplot_with_lr("All", df_int_seg_petTOAD, "emp_int", "PTEDUCAT", axs[0, 0], "tab:orange")
regplot_with_lr("All", df_int_seg_petTOAD, "emp_seg", "PTEDUCAT", axs[1, 0], "tab:orange")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD, "emp_int", "PTEDUCAT", axs[0, 1], "tab:blue")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD, "emp_seg", "PTEDUCAT", axs[1, 1], "tab:blue")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD, "emp_int", "PTEDUCAT", axs[0, 2], "tab:green")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD, "emp_seg", "PTEDUCAT", axs[1, 2], "tab:green")
fig.tight_layout()

#%%
fig, axs = plt.subplots(nrows=2, ncols = 3, figsize = (15, 10))
regplot_with_lr("All", df_int_seg_petTOAD, "emp_int", "Age", axs[0, 0], "tab:orange")
regplot_with_lr("All", df_int_seg_petTOAD, "emp_seg", "Age", axs[1, 0], "tab:orange")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD, "emp_int", "Age", axs[0, 1], "tab:blue")
regplot_with_lr("CN_WMH", df_int_seg_petTOAD, "emp_seg", "Age", axs[1, 1], "tab:blue")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD, "emp_int", "Age", axs[0, 2], "tab:green")
regplot_with_lr("MCI_WMH", df_int_seg_petTOAD, "emp_seg", "Age", axs[1, 2], "tab:green")
fig.tight_layout()
