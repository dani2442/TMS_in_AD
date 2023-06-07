#!/bin/bash
#SBATCH --job-name="neurolib"
#SBATCH --mail-user=riccardoleone1991@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/home/leoner/petTOAD/slurm/outputs/output_%A_%a.out
#SBATCH --error=/home/leoner/petTOAD/slurm/errors/error_%A_%a.err
#SBATCH --time=2-00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G

module load singularity 
# Change to the directory where the Singularity image is located
cd /home/leoner/petTOAD/singims/neurolib.sif

# Run the Python script using the Singularity image
singularity run --bind /home/leoner/petTOAD/scripts/TMS_in_AD/03_modeling_neurolib:/work /home/leoner/singims/neurolib.sif python /work/petTOAD_homogeneous_model.py






