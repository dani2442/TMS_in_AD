#!/bin/bash

LES_DIR=../../../data/preprocessed/WMH_lesion_masks
mkdir $LES_DIR

WMH_DIR=../../../data/preprocessed/WMH_segmentation

for subj_dir in WMH_DIR/*; do
    cp *fromSubjSpace.nii.gz $LES_DIR
done