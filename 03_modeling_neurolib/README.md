This folder contains the main code for simulation and analyses of Leone R, Geysen S, Deco G, Kobeleva X; Alzheimer's Disease Neuroimaging Initiative. Beyond Focal Lesions: Dynamical Network Effects of White Matter Hyperintensities. Hum Brain Mapp. 2024 Dec 1;45(17):e70081. doi: 10.1002/hbm.70081. PMID: 39624946; PMCID: PMC11612665.

### Simulation pipeline
The main scripts to run the simulations are (to be run in the following order):

- 1.0-group_level_simulations.py --> runs the simulation at a group level to find the best G, first to run
- 1.1-find_plot_best_G.ipynb --> finds the best G tuned to healthy controls
- 1.2-single_subjects_simulations.py --> starts from the previously found best G and runs the simulation at a single-subjec level for all models. The weights and biases can be set in the helper script petTOAD_parameter_setup.py (note that this script needs to be run with 1.3-run_single_subjs_sim.sh at the moment, but could easily be changed to accept a subject name)
- 1.3-run_single_subjs_sim.sh --> launces slurm processing of the single subject simulations


### Analyses and plots
The following scripts are for analyses, plots and tables:

- 2.0-analyze_empirical_data.ipynb --> creates:
    Figures:
    - Projections of wmh frequency;
    - Suppl. Fig. 4 (correlation between wmh and tau and amyloid)
    Tables:
    - Suppl. Table 1 (table of demographics)

- 2.1-analyze_single_subj_simulations.ipynb --> creates: 

    Figures:
    - Fig. 3A (distribution of WMH across the brain)
    - Fig. 3B (histograms of phFCD distribution in WMH/no WMH)
    - Fig. 3C-D (boxplots of model performances + correlation plots of % improvement and wmh log);
    - Fig. 4 (correlation plots of % improvement with clinical data with SDC models);
    - Suppl. Fig. 4 (comparison with random models);
    - Suppl. Fig. 5 (correlation plots of % improvement with clinical data with NDC models)
    
    Tables:
    - Table 1 (table of model performance in all wmh and in high wmh only subjects)
    - Table 2 (correlations between improvement in model performance compared to the baseline and clinical/demographic data)
    - Suppl. Table 3 (model performance of random vs. non random models)


### Helpers script
We have several scripts helping with the simulations:

- BOLDFilters.py (from https://github.com/dagush/WholeBrain)
- demean.py (from https://github.com/dagush/WholeBrain)
- filteredPowerSpectralDensity.py (from https://github.com/dagush/WholeBrain)
- phFCD.py (from https://github.com/dagush/WholeBrain)
- my_functions.py 
- petTOAD_load.py
- petTOAD_setup.py
- petTOAD_parameter_setup.py

And one containing all the functions helping with the analyses:

- petTOAD_analyses_helpers.py
