TO-DO: Expand on this!


The project is structured into:
- load.py: contains a lot of different functions to load stuff we are going to use in the analyses;
- setup.py: contains some initial setup things like filtering timeseries and loading the structural connectivity;
- group_level_simulations.py: runs the simulations at the group level (here you can modify the models that you want)
- group_level_analysis.py: call group_level_simulations.py and then performs the comparisons between the different groups, as well as the evaluation of the group-level integration and segregation with Monte Carlo resampling 
- group_level_plotting.ipynb: jupyter notebook for plotting
