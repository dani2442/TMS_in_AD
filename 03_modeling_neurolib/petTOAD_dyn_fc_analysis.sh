#!/bin/bash
# First script
script_main_FC_states="/home/riccardo/brain_dynamics/main_FC_states.py"
cn_no_wmh_dir="/home/riccardo/petTOAD/data/fcd/cn_no_wmh"
cn_wmh_dir="/home/riccardo/petTOAD/data/fcd/cn_wmh"
mci_no_wmh_dir="/home/riccardo/petTOAD/data/fcd/mci_no_wmh"
mci_wmh_dir="/home/riccardo/petTOAD/data/fcd/mci_wmh"
res_dir="/home/riccardo/petTOAD/results/fcd_pca"

n_areas=90
tr_value=3

# Second script
script_main_states_features="/home/riccardo/brain_dynamics/main_states_features.py"

# Array of cluster values
clusters=(3) #(4 5 6 7 8 9 10)

# Loop through the cluster values
for n_cluster in "${clusters[@]}"; do

    # Create output directory with the cluster value
    clust_dir="$res_dir/fcd_cluster_autoen_$n_cluster"
    mkdir -p $clust_dir    

    # Run the Python script with the current cluster value
    command1="python $script_main_FC_states \
        --input $cn_no_wmh_dir $cn_wmh_dir $mci_no_wmh_dir $mci_wmh_dir \
        --output $clust_dir \
        --areas $n_areas \
        --autoen \
        --clusters $n_cluster \
        --tr $tr_value \
        --imb"     

    command2="python $script_main_states_features \
    --input $clust_dir/concatentated_matrix_clusters.npz \
    --output $clust_dir/states \
    --n_clusters $n_cluster \
    --starts $clust_dir/arrays_starts.json \
    --separate \
    --clusters $clust_dir/clustered_matrix.npz \
    --tr $tr_value \
    --sub_t $clust_dir/subjects_times_dict.json"

    echo "Running command: $command1"
    $command1

    echo "Running command: $command2"
    $command2

done
