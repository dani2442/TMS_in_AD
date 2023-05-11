#!/bin/bash
#SBATCH --job-name="py"
#SBATCH --mail-user=xkobeleva@gmail.com
#SBATCH --mem-per-cpu=45G
#SBATCH --output=/home/xkobeleva/spatial_scales/basic_scripts/output/output_%A_%a.out
#SBATCH --error=/home/xkobeleva/spatial_scales/basic_scripts/errors/error_%A_%a.err
#SBATCH --array 1-1%1


# 65 G f�r 800 und 1000, 40G f�r 400 und 600, 15G f�r 100-300
#Load Python 3.6.4
#ml Python

#Load Python 2.7.15
#Python/2.7.15-foss-2018a

#export PYTHONUSERBASE=/usr/bin/env/python

#python flow.py 
#python mou_ec.py
#
#conda install git pip
#    Install git in the virtual environment: (conda install git)
#    Clone the project. (git clone [URL])
#    Install the package (cd to package directory that contains setup.py.
#    Then run "python3 setup.py install").

# Activate Anaconda work environment for OpenDrift
#source /home/${USER}/.bashrc
#export PATH=/home/xkobeleva/python/miniconda/:$PATH.
#/home/xkobeleva/python/miniconda/bin/conda create --name flow python=3 pandas numpy scipy matplotlib pip 
/home/xkobeleva/python/miniconda/bin/activate flow
#conda activate flow
#if packages need to be installed from github
# e.g. pip install git+https://github.com/mb-BCA/NetDynFlow.git@master

#if package needs to be installed: pip install oder conda install

# cd /home/xkobeleva/spatial_scales/basic_scripts
#/home/xkobeleva/python/miniconda/envs/mou_ec/bin/python3 mou_ec.py

# /home/xkobeleva/python/miniconda/envs/flow/bin/python3 /home/xkobeleva/spatial_scales/basic_scripts/mou_ec.py
/home/xkobeleva/python/miniconda/envs/flow/bin/python3 /home/xkobeleva/spatial_scales/basic_scripts/flow.py



