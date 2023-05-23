#!/bin/bash

PROJ_DIR=/home/riccardo/petTOAD
DATA_DIR=$PROJ_DIR/data
PREPRO_DIR=$DATA_DIR/preprocessed
LES_DIR=$PREPRO_DIR/WMH_lesion_masks
mkdir $LES_DIR

WMH_DIR=$PREPRO_DIR/WMH_segmentation

for subj_dir in $WMH_DIR/*; do
    echo $subj_dir
    cd $subj_dir
    cp *fromSubjSpace.nii.gz $LES_DIR
done