# TMS_in_AD

Code accompanying the paper *"Beyond Focal Lesions: Dynamical Network Effects of White Matter Hyperintensities"*.

This repository is **not** a standalone, end-to-end runnable project: the scripts expect external (not versioned) inputs under a project “spine” directory (BIDS derivatives, WMH segmentations, SC matrices, LQT outputs). This README documents the **intended workflow** and how to execute each stage.

## How paths work in this repo (important)

Most scripts define:

- `SPINE = Path.cwd().parents[2]`

and then assume these folders exist under `SPINE`:

- `SPINE/data` (inputs)
- `SPINE/results` (outputs)
- `SPINE/scripts/TMS_in_AD` (this repo)

That means you must run scripts from a working directory like:

```bash
cd <SPINE>/scripts/TMS_in_AD/03_modeling_neurolib
python 1.0-group_level_simulations.py
```

If you run scripts from the repo root (`.../projects/TMS_in_AD`), `SPINE` will resolve to a different directory and the scripts will look for `data/` and `results/` in the wrong place.

## Pipeline overview (execution order)

1) **Build subject table + WMH load** (`01_calculate_WMH_load/`)
2) **Prepare lesion masks and run LQT** (`02_LQT/`)
3) **Run neurolib simulations** (group-level fit → single-subject simulations; `03_modeling_neurolib/`)
4) **Run analysis notebooks** (`03_modeling_neurolib/2.0-*.ipynb`, `2.1-*.ipynb`)

## Required inputs (external to this repo)

The scripts assume (minimum):

- **WMH segmentations** under `SPINE/data/preprocessed/WMH_segmentation/` as BIDS-derivatives.
	- `extract_WMH_volumes.py` looks for `label=WMHMask` and chooses the file **without** `"space"` in the filename.
- **xcp-d timeseries** under `SPINE/data/preprocessed/xcp_d/`.
	- `petTOAD_load.load_ts_aal()` expects per subject:
		- `SPINE/data/preprocessed/xcp_d/sub-<PTID>/ses-M00/func/sub-<PTID>_ses-M00_task-rest_space-MNI152NLin2009cAsym_atlas-AAL_cort_timeseries.csv`
- **Structural connectivity matrices** under `SPINE/data/utils/AAL_not_norm/`.
	- `petTOAD_load.load_norm_aal_sc()` reads `S*_rawcounts.csv`, averages across files, symmetrizes, then normalizes.
- **Clinical/selection CSVs** under `SPINE/data/utils/`:
	- `ADNI_selected_pts.csv`
	- `df_wmh_checklist.csv`
	- `df_adnimerge.csv`
- **LQT outputs** under `SPINE/results/LQT/`:
	- `SPINE/results/LQT/dataframes/parc_discon.csv` (node damage; used by heterogeneous model)
	- `SPINE/results/LQT/sub-<PTID>/pct_sdc_matrix.csv` (SC disconnection; used by disconnection model)

## Step-by-step execution

### 0) Create the expected directory layout

At minimum, you need a spine directory like:

```text
<SPINE>/
	data/
		preprocessed/
			WMH_segmentation/
			xcp_d/
		utils/
			ADNI_selected_pts.csv
			df_wmh_checklist.csv
			df_adnimerge.csv
			AAL_not_norm/
				S*_rawcounts.csv
	results/
	scripts/
		TMS_in_AD/   (this repository)
```

### 1) Compute WMH lesion load + build `df_petTOAD.csv`

```bash
cd <SPINE>/scripts/TMS_in_AD/01_calculate_WMH_load
python extract_WMH_volumes.py
python update_df_petTOAD.py
```

Outputs:

- `SPINE/results/df_petTOAD.csv`

Notes:

- `extract_WMH_volumes.py` currently **computes** a dataframe (`df_wmh`) but does **not** write it to disk.
- `update_df_petTOAD.py` does **not** merge the computed `df_wmh` dataframe; it assumes `WMH_load_subj_space` is already present in `df_wmh_checklist.csv` (or was merged upstream).

### 2) Prepare lesion masks for LQT

On Linux/macOS you can collect subject-space WMH masks into a single folder:

```bash
cd <SPINE>/scripts/TMS_in_AD/02_LQT
bash create_mask_dir.sh
```

Important: `create_mask_dir.sh` contains a hardcoded `PROJ_DIR=/home/riccardo/petTOAD`. Edit it to point to your `<SPINE>`.

Output:

- `SPINE/data/preprocessed/WMH_lesion_masks/*.nii.gz`

### 3) Run LQT (R)

`02_LQT/LQT.R` was run on Windows in the original workflow and contains hardcoded paths.

Required edits:

- Set `BASE_DIR` to your `<SPINE>`
- Fix a typo: the script calls `dir.create(LQTT_DIR)` but the variable is `LQT_DIR`

Then run from R:

```r
setwd("<SPINE>/scripts/TMS_in_AD/02_LQT")
source("LQT.R")
```

Outputs:

- `SPINE/results/LQT/dataframes/*.csv` (parcel and tract disconnection measures)
- `SPINE/data/utils/Schaefer200_sc.csv` (exported from LQT atlas connectivity; optional)

### 4) Neurolib modeling (group-level fit)

This stage uses:

- `SPINE/results/df_petTOAD.csv` (group labels + WMH)
- `SPINE/data/preprocessed/xcp_d/...` (empirical timeseries)
- `SPINE/data/utils/AAL_not_norm/S*_rawcounts.csv` (SC)

Run:

```bash
cd <SPINE>/scripts/TMS_in_AD/03_modeling_neurolib
python petTOAD_setup.py
python 1.0-group_level_simulations.py
```

Outputs:

- `SPINE/results/subjs_to_sim.csv` (written by `petTOAD_setup.py`)
- `SPINE/results/group_simulations/group-CN-no-WMH_desc-best-G.csv` (written by `1.0-group_level_simulations.py`)

Optional: open and run `1.1-find_plot_best_G.ipynb` to plot/inspect the fit.

### 5) Neurolib modeling (single-subject simulations)

Local run (argument is **1-based index** into `subjs_to_sim`):

```bash
cd <SPINE>/scripts/TMS_in_AD/03_modeling_neurolib
python 1.2-single_subjects_simulations.py 1
```

Default outputs:

- `SPINE/results/simulations_2024_06_18/heterogeneous_*_random/sub-<PTID>_df_results_heterogeneous.csv`
- `SPINE/results/simulations_2024_06_18/sc_disconn_*_random/sub-<PTID>_df_results_disconn.csv`

Notes:

- As currently committed, `1.2-single_subjects_simulations.py` runs **only** the heterogeneous and SC-disconnection models and only for `random_conditions = [True]`. The homogeneous `a` and homogeneous `G` model calls are commented out.
- The output folder name `simulations_2024_06_18` is hardcoded.

HPC/SLURM run:

- `1.3-run_single_subjs_sim.sh` is an example SLURM array job using Singularity.
- It contains hardcoded paths (`/home/leoner/...`) and points to a script name that is not in this repo (`petTOAD_single_subjects_simulations.py`).

To use it, you must update at least:

- `#SBATCH --array=1-<N>` to match the number of subjects in `SPINE/results/subjs_to_sim.csv`
- `SINGULARITY_CONTAINER=...` to your container
- `SCRIPT=...` to `<SPINE>/scripts/TMS_in_AD/03_modeling_neurolib/1.2-single_subjects_simulations.py`

### 6) Analyses / figures / tables (notebooks)

Run from the same working directory as above (`<SPINE>/scripts/TMS_in_AD/03_modeling_neurolib`).

- `2.0-analyze_empirical_data.ipynb`: empirical-only figures (WMH maps, phFCD distributions)
- `2.1-analyze_single_subj_simulations.ipynb`: simulation results figures/tables

## File reference (what each file does)

### `01_calculate_WMH_load/`

- `extract_WMH_volumes.py`: computes per-subject `WMH_load_subj_space` from BIDS-derivatives WMH masks.
- `update_df_petTOAD.py`: builds `results/df_petTOAD.csv` (groups + Fazekas-based bins) from CSVs in `data/utils/`.

### `02_LQT/`

- `create_mask_dir.sh`: copies lesion masks from `data/preprocessed/WMH_segmentation/` into `data/preprocessed/WMH_lesion_masks/`.
- `LQT.R`: runs LQT damage/disconnection computations and writes analysis-ready CSVs + exports an SC matrix.
- `preprocess_aal.py`: one-off AAL atlas conversion to MNI6Asym and region-center CSV; contains a hardcoded Windows path for copying outputs.
- `get_aal_mni6.py`: debug/inspection script printing a JSON file from `data/utils/`.

### `03_modeling_neurolib/`

- `petTOAD_load.py`: central loader + path definitions; loads AAL SC, xcp-d AAL timeseries, group labels, LQT disconnection matrices.
- `petTOAD_setup.py`: filters timeseries and writes `results/subjs_to_sim.csv`.
- `petTOAD_parameter_setup.py`: defines parameter grids (`ws_*`, `bs_*`) for subject-wise searches.
- `1.0-group_level_simulations.py`: group-level Hopf model fitting (currently configured for CN_no_WMH, homogeneous G).
- `1.1-find_plot_best_G.ipynb`: plots/selects best `G` from the group-level run.
- `1.2-single_subjects_simulations.py`: runs per-subject simulations for multiple model variants (currently heterogeneous + disconnection, random condition).
- `1.3-run_single_subjs_sim.sh`: SLURM/Singularity wrapper (paths must be edited).
- `2.0-analyze_empirical_data.ipynb`, `2.1-analyze_single_subj_simulations.ipynb`: analysis notebooks.
- `my_functions.py`: metrics/utilities (FC, correlations, KS distance, helpers used by simulation + analysis).
- `petTOAD_analyses_helpers.py`: plotting/stats utilities used by notebooks (expects specific simulation output folder names).
- `BOLDFilters.py`, `demean.py`, `filteredPowerSpectralDensity.py`, `phFCD.py`: signal processing utilities (adapted from WholeBrain).

## Known sharp edges / things you may need to edit

- **Hardcoded paths**: `02_LQT/create_mask_dir.sh`, `02_LQT/LQT.R`, `02_LQT/preprocess_aal.py`, and `03_modeling_neurolib/1.3-run_single_subjs_sim.sh` contain user/machine-specific absolute paths.
- **Working directory requirement**: scripts rely on `Path.cwd().parents[2]`.
- **WMH load merge**: `extract_WMH_volumes.py` doesn’t persist outputs; `update_df_petTOAD.py` assumes WMH load is already available in the utils CSV.

If you want, I can also refactor the codebase to derive `SPINE` from `__file__` (repo root) and/or support an explicit `SPINE` environment variable so you can run everything from anywhere.




## Fit Hopf on a single subject from a `.mat`

If you have a MATLAB file like `ts_young_TR0.72.mat` containing `timeseries_all` (per-subject BOLD), you can fit a Hopf model to one subject and evaluate it with the same metrics used in this repo (FC correlation + phFCD KS).

Script: `fit_hopf_single_subject_from_mat.py`

Example:

```bash
python 03_modeling_neurolib/fit_hopf_single_subject_from_mat.py \
    --mat ts_young_TR0.72.mat \
    --subject-index 0 \
    --tr 0.72 \
    --sc /path/to/your_sc.csv \
    --out results/hopf_single_subj
```

Outputs:
- `subj-<idx>_hopf_fit_results.csv` (one row per `K_gl` value)
- `subj-<idx>_hopf_fit_summary.json` (best `K_gl` by min phFCD KS)