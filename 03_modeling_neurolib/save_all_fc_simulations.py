# %% Initial imports
from neurolib.utils import paths
from neurolib.utils import pypetUtils as pu
from neurolib.utils.parameterSpace import ParameterSpace

import my_functions as my_func
from petTOAD_parameter_setup import *
from petTOAD_setup import *

RES_DIR = Path("/home/leoner/Backup_petTOAD/petTOAD/results")

SIM_DIR = RES_DIR / "final_simulations_log_2023-11-23"
SIM_GROUP_DIR = RES_DIR / "final_simulations"
A_DIR = SIM_DIR / "a-weight_ws_-0.1-0.0_bs_-0.05-0.0"
A_RAND_DIR = SIM_DIR / "a-weight_ws_-0.1-0.0_bs_-0.05-0.0_random"
G_DIR = SIM_DIR / "G-weight_ws_-1.0-0.0_bs_-0.5-0.0"
G_RAND_DIR = SIM_DIR / "G-weight_ws_-1.-0.0_bs_-0.5-0.0_random"
SC_DIR = SIM_DIR / "sc_disconn_ws_-0.5-0.0_bs_0"
SC_RAND_DIR = SIM_DIR / "sc_disconn_ws_-0.5-0.0_bs_0_random"
HET_DIR = SIM_DIR / "heterogeneous_ws_-0.1-0.0_bs_-0.05-0.0"
HET_RAND_DIR = SIM_DIR / "heterogeneous_ws_-0.1-0.0_bs_-0.05-0.0_random"


def calculate_results_from_bolds_subject(bold_arr, n_sim, n_parms, n_nodes):
    # Create a new array to store the FC values with the same shape as bold array
    fc_array = np.zeros([n_sim, n_parms, n_nodes, n_nodes])

    # Iterate over each element in the bold array
    for i in range(n_sim):
        for j in range(n_parms):
            print(
                f"Now calculating results from the {i} simulation for parameter {j}..."
            )
            # Get the current timeseries
            timeseries = bold_arr[i, j].squeeze()

            # Perform FC analysis
            if np.isnan(timeseries).any():
                print("Simulation has some nans, aborting!")
                continue
            else:
                print("Calculating FC..")
                fc_value = my_func.fc(timeseries)
                # Store the FC value in the corresponding position in the arrays
                fc_array[i, j] = fc_value
    return fc_array

def load_or_gather_results_from_repeated_simulations(
    subj, res_dir, filename, model_type, ws=None, bs=None, disconn=False
):
    try:
        emp_res = np.load(f"{res_dir}/group-single-subj-{subj}_data-empirical_desc-fc-phfcd-arr.npz")
        emp_fc = emp_res["emp_fc"]
    except:
        timeseries = all_fMRI_clean[subj]
        emp_fc = my_func.fc(timeseries)
        np.savez_compressed(f"{res_dir}/group-single-subj-{subj}_data-empirical_desc-fc-phfcd-arr.npz", emp_fc = emp_fc)
    try:
       if subj in subjs_to_sim:
            sim_res = np.load(f"{res_dir}/group-single-subj-{subj}_data-simulated_model-{model_type}_desc-fc-phfcd-arr.npz")
            sim_fc = sim_res["sim_fc"]
       else:
           return np.nan, emp_fc
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
        fc_array = calculate_results_from_bolds_subject(
            bold_arr, n_sim, n_parms, n_nodes
        )
        # Get the average fc across the n simulations
        sim_fc = fc_array.mean(axis=0)
        np.savez_compressed(f"{res_dir}/group-single-subj-{subj}_data-simulated_model-{model_type}_desc-fc-arr.npz", sim_fc = sim_fc)
    return sim_fc, emp_fc

def find_best_b_w(subj, model, df_best_combo, bs, ws):
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
        df["w"] = round(df["w"], 3)
        best_idx = df[df["w"] == w].index.values[0]
    return best_idx

def calculate_fc_model(sim_subjs, model_type, dict_model, df_best_combo):
    if model_type != "disconnectivity":
        disconn = False
    else:
        disconn = True
    
    bs = dict_model[model_type]["bs"]
    ws = dict_model[model_type]["ws"]
    res_dir = dict_model[model_type]["res_dir"]
    
    list_sim_fc_cn_wmh = []
    list_sim_fc_mci_wmh = []

    list_emp_fc_cn_wmh = []
    list_emp_fc_mci_wmh = []
    
    for _, subj in enumerate(sim_subjs):
        filename = subj + dict_model[model_type]["suffix"]
        sim_fc_arr, emp_fc = load_or_gather_results_from_repeated_simulations(subj, res_dir, filename, model_type, ws=ws, bs=bs, disconn=disconn)
        best_idx = find_best_b_w(subj, model_type, df_best_combo, bs, ws)
        sim_fc = sim_fc_arr[best_idx]
        if subj in CN_WMH:
            list_sim_fc_cn_wmh.append(sim_fc)
            list_emp_fc_cn_wmh.append(emp_fc)
        elif subj in MCI_WMH:
            list_sim_fc_mci_wmh.append(sim_fc)
            list_emp_fc_mci_wmh.append(emp_fc)
    
    arr_sim_fc_cn_wmh = np.array(list_sim_fc_cn_wmh)
    arr_sim_fc_mci_wmh = np.array(list_sim_fc_mci_wmh)
    arr_emp_fc_cn_wmh = np.array(list_emp_fc_cn_wmh)
    arr_emp_fc_mci_wmh = np.array(list_emp_fc_mci_wmh)

    return arr_sim_fc_cn_wmh, arr_sim_fc_mci_wmh, arr_emp_fc_cn_wmh, arr_emp_fc_mci_wmh

if __name__ == "__main__":
    df_best_combinations = pd.read_csv(SIM_DIR / "group-all_data-simulated_model-all_desc-best-b-w-combinations.csv", index_col=0)
    dict_model_bs_ws = {
        "homogeneous_G": {"res_dir": G_DIR, "suffix": "_homogeneous_G-weight_model.hdf","bs": bs_G, "ws": ws_G},
        "disconnectivity": {"res_dir": SC_DIR, "suffix": "_sc_disconn_model.hdf", "bs": None, "ws": ws_disconn},
        }

    arr_sim_fc_cn_wmh_G, arr_sim_fc_mci_wmh_G, arr_emp_fc_cn_wmh_G, arr_emp_fc_mci_wmh_G = calculate_fc_model(subjs_to_sim, "homogeneous_G", dict_model_bs_ws, df_best_combinations)
    arr_sim_fc_cn_wmh_disconn, arr_sim_fc_mci_wmh_disconn, arr_emp_fc_cn_wmh_disconn, arr_emp_fc_mci_wmh_disconn = calculate_fc_model(subjs_to_sim, "disconnectivity", dict_model_bs_ws, df_best_combinations)
