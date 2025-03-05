#!/bin/bash
#SBATCH --job-name="neurolib"
#SBATCH --output=/home/leoner/Projects/petTOAD/slurm/outputs/output_%A_%a.out
#SBATCH --error=/home/leoner/Projects/petTOAD/slurm/errors/error_%A_%a.err
#SBATCH --time=2-12:00:00
#SBATCH --array=1-87
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=HPC-default

# Load the required module for Singularity
module load singularity

# Set the paths to the Singularity container and Python script
SINGULARITY_CONTAINER="/home/leoner/Projects/petTOAD/singims/neurolib.sif"
SCRIPT="/home/leoner/Projects/petTOAD/scripts/TMS_in_AD/03_modeling_neurolib/petTOAD_single_subjects_simulations.py"

# Run the Python script inside the Singularity container assignin the slurm_task_id which is used by the script to iterate over subjs
singularity exec $SINGULARITY_CONTAINER python $SCRIPT $SLURM_ARRAY_TASK_ID
