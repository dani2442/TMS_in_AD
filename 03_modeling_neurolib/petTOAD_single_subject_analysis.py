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
from petTOAD_group_level_int_seg_analysis import n_simulations, integration_parallel, segregation_parallel, run_int_seg_parallel
# %%

SIM_DIR = RES_DIR / "final_simulations_log_2023-11-23"
SIM_GROUP_DIR = RES_DIR / "final_simulations"
A_DIR = SIM_DIR / "a-weight_ws_-0.1-0.0_bs_-0.05-0.0"
A_RAND_DIR = SIM_DIR / "a-weight_ws_-0.1-0.0_bs_-0.05-0.0_random"
G_DIR = SIM_DIR / "G-weight_ws_-1.0-0.0_bs_-0.5-0.0"
G_RAND_DIR = SIM_DIR / "G-weight_ws_-1.o-0.0_bs_-0.5-0.0_random"
SC_DIR = SIM_DIR / "sc_disconn_ws_-0.5-0.0_bs_0"
SC_RAND_DIR = SIM_DIR / "sc_disconn_ws_-0.5-0.0_bs_0_random"
HET_DIR = SIM_DIR / "heterogeneous_ws_-0.1-0.0_bs_-0.05-0.0"
HET_RAND_DIR = SIM_DIR / "heterogeneous_ws_-0.1-0.0_bs_-0.05-0.0_random"

# %% Load stuff
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
        emp_res = np.load(f"{res_dir}/group-single-subj-{subj}_data-empirical_desc-fc-phfcd-arr.npz")
        emp_fc = emp_res["emp_fc"]
    except:
        timeseries = all_fMRI_clean[subj]
        emp_fc = my_func.fc(timeseries)
        emp_phfcd = phFCD(timeseries)
        np.savez_compressed(f"{res_dir}/group-single-subj-{subj}_data-empirical_desc-fc-phfcd-arr.npz", emp_fc = emp_fc, emp_phfcd = emp_phfcd)
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
        fc_array, phfcd_array = calculate_results_from_bolds_subject(
            bold_arr, n_sim, n_parms, n_nodes
        )
        # Get the average fc across the n simulations
        sim_fc = fc_array.mean(axis=0)
        np.savez_compressed(f"{res_dir}/group-single-subj-{subj}_data-simulated_model-{model_type}_desc-fc-phfcd-arr.npz", sim_fc = sim_fc, sim_phfcd = phfcd_array)
    return sim_fc, emp_fc #fc_pearson, phfcd_ks

# Int-seg functions
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

def compute_int_seg(fc, sim=False, best_index=None):
    if sim:
        fc = fc[best_index]
    abs_fc = np.abs(fc)
    segregation = computeSegregation(abs_fc)[0]
    integration = IntegrationFromFC_Fast(abs_fc)
    return integration, segregation

def process_empirical_int_seg(all_subjs, model_type, dict_model, disconn):
    savename = SIM_DIR / f"group-single-subjects_data_empirical_desc-int-seg.csv"
    try:
        df_int_seg = pd.read_csv(savename, index_col = 0)
    except:
            
        bs = dict_model[model_type]["bs"]
        ws = dict_model[model_type]["ws"]
        res_dir = dict_model[model_type]["res_dir"]

        list_emp_int = []
        list_emp_seg = []
        
        for i, subj in enumerate(all_subjs):
            filename = subj + dict_model[model_type]["suffix"]
            _, emp_fc = load_or_gather_results_from_repeated_simulations(subj, res_dir, filename, model_type, ws=ws, bs=bs, disconn=disconn)
            emp_int, emp_seg = compute_int_seg(emp_fc, best_index=None, sim=False)
            list_emp_int.append(emp_int)
            list_emp_seg.append(emp_seg)
        
        df_int_seg = pd.DataFrame({"PTID": all_subjs,
                                    "emp_int": list_emp_int,
                                    "emp_seg": list_emp_seg}
                                    )
        df_int_seg.to_csv(savename)
    return df_int_seg
    
def process_simulated_int_seg(sim_subjs, model_type, dict_model, df_best_combo, disconn):
    savename = SIM_DIR / f"group-single-subjects_data_simulated_model-{model_type}_desc-int-seg.csv"
    try:
        df_int_seg = pd.read_csv(savename, index_col = 0)            
    except:     
        bs = dict_model[model_type]["bs"]
        ws = dict_model[model_type]["ws"]
        res_dir = dict_model[model_type]["res_dir"]
        list_sim_int = []
        list_sim_seg = []
        for i, subj in enumerate(sim_subjs):
            filename = subj + dict_model[model_type]["suffix"]
            sim_fc_arr, _ = load_or_gather_results_from_repeated_simulations(subj, res_dir, filename, model_type, ws=ws, bs=bs, disconn=disconn)
            best_idx = find_best_b_w(subj, model_type, df_best_combo, bs, ws)
            sim_int, sim_seg = compute_int_seg(sim_fc_arr, sim=True, best_index=best_idx)
            list_sim_int.append(sim_int)
            list_sim_seg.append(sim_seg)

        df_int_seg = pd.DataFrame({"PTID": sim_subjs,
                            "sim_int": list_sim_int,
                            "sim_seg": list_sim_seg}
                            )
        df_int_seg.to_csv(savename)
    return df_int_seg

def process_int_seg_subjectwise(all_subjs, sim_subjs, model_type, dict_model, df_best_combo):
    if model_type != "disconnectivity":
        disconn = False
    else:
        disconn = True
    df_int_seg_emp = process_empirical_int_seg(all_subjs, model_type, dict_model, disconn)
    df_int_seg_sim = process_simulated_int_seg(sim_subjs, model_type, dict_model, df_best_combo, disconn)
    df_int_seg_emp_sim = pd.merge(df_int_seg_emp, df_int_seg_sim, on = "PTID")
    df_int_seg_emp_sim.to_csv(SIM_DIR / f"group-single-subjects_data_both_model-{model_type}_desc-emp-sim-int-seg.csv")
    return df_int_seg_emp_sim

def run_int_seg_montecarlo_single_subject_model(sim_subjs, model_type, dict_model, df_best_combo, n_rep):
    try:
        df_int_all = pd.read_csv(SIM_DIR / f"group-all-sing-subj_data-simulated_model-{model_type}_desc-df-int-montecarlo-{n_rep}-repeats.csv", index_col=0)
        df_seg_all = pd.read_csv(SIM_DIR / f"group-all-sing-subj_data-simulated_model-{model_type}_desc-df-seg-montecarlo-{n_rep}-repeats.csv", index_col=0)
    except:
        if model_type == "disconnectivity":
            disconn = True
        else:
            disconn = False

        bs = dict_model[model_type]["bs"]
        ws = dict_model[model_type]["ws"]
        res_dir = dict_model[model_type]["res_dir"]
        list_sim_fc_cn_wmh = []
        list_sim_fc_mci_wmh = []

        for subj in sim_subjs:
            filename = subj + dict_model[model_type]["suffix"]
            sim_fc_arr, _ = load_or_gather_results_from_repeated_simulations(subj, res_dir, filename, model_type, ws=ws, bs=bs, disconn=disconn)
            best_idx = find_best_b_w(subj, model_type, df_best_combo, bs, ws)
            sim_fc = sim_fc_arr[best_idx]
            if subj in CN_WMH:
                list_sim_fc_cn_wmh.append(sim_fc)
            elif subj in MCI_WMH:
                list_sim_fc_mci_wmh.append(sim_fc)

        arr_sim_fc_cn_wmh = np.array(list_sim_fc_cn_wmh)
        arr_sim_fc_mci_wmh = np.array(list_sim_fc_mci_wmh)
        integrations_cn_wmh, segregations_cn_wmh = run_int_seg_parallel(groupname="subject-level CN WMH", arr_fc=arr_sim_fc_cn_wmh, repeats=n_rep, n_samples= 25)
        integrations_mci_wmh, segregations_mci_wmh = run_int_seg_parallel(groupname="subject-level MCI WMH", arr_fc=arr_sim_fc_mci_wmh, repeats=n_rep, n_samples= 25)

        df_int_cn_wmh = pd.DataFrame({"Integration": integrations_cn_wmh,
                                    "group": "CN_WMH",})

        df_int_mci_wmh = pd.DataFrame({"Integration": integrations_mci_wmh,
                                    "group": "MCI_WMH",})

        df_seg_cn_wmh = pd.DataFrame({"Segregation": segregations_cn_wmh,
                                    "group": "CN_WMH",})

        df_seg_mci_wmh = pd.DataFrame({"Segregation": segregations_mci_wmh,
                                    "group": "MCI_WMH",})

        df_int_all = pd.concat([df_int_cn_wmh, df_int_mci_wmh], axis = 0)
        df_seg_all = pd.concat([df_seg_cn_wmh, df_seg_mci_wmh], axis = 0)
        df_int_all.to_csv(SIM_DIR / f"group-all-sing-subj_data-simulated_model-{model_type}_desc-df-int-montecarlo-{n_rep}-repeats.csv")
        df_seg_all.to_csv(SIM_DIR / f"group-all-sing-subj_data-simulated_model-{model_type}_desc-df-seg-montecarlo-{n_rep}-repeats.csv")
    return df_int_all, df_seg_all

#%% 
n_repeats = 1000
df_best_combinations = pd.read_csv(SIM_DIR / "group-all_data-simulated_model-all_desc-best-b-w-combinations.csv", index_col=0)
#%%
dict_model_bs_ws = {
        "homogeneous_G": {"res_dir": G_DIR, "suffix": "_homogeneous_G-weight_model.hdf","bs": bs_G, "ws": ws_G},
        "disconnectivity": {"res_dir": SC_DIR, "suffix": "_sc_disconn_model.hdf", "bs": None, "ws": ws_disconn},
        }


df_int_seg_emp_sim_G = process_int_seg_subjectwise(subjs, subjs_to_sim, "homogeneous_G", dict_model_bs_ws, df_best_combinations)
df_int_seg_emp_sim_disconn = process_int_seg_subjectwise(subjs, subjs_to_sim, "disconnectivity", dict_model_bs_ws, df_best_combinations)
df_montecarlo_int_all_G, df_montecarlo_seg_all_G = run_int_seg_montecarlo_single_subject_model(subjs_to_sim, "homogeneous_G", dict_model_bs_ws, df_best_combinations, n_repeats)
df_montecarlo_int_all_disconn, df_montecarlo_seg_all_disconn = run_int_seg_montecarlo_single_subject_model(subjs_to_sim, "disconnectivity", dict_model_bs_ws, df_best_combinations, n_repeats)