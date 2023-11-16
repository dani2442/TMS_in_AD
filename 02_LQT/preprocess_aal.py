#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""     AAL atlas conversion to MNI6Asym -- Version 1.0
Last edit:  2023/05/17
Authors:    Leone, Riccardo (RL)
Notes:      - Converts the AAL atlas from MNI2009cAsym to MNI6Asym
            - Remember that LQT runs on Windows because I can't make it run on the cluster! 
            - Release notes:
                * Initial release
To do:      
Comments:   
Sources:  
"""

# %%
# Imports
import templateflow.api as tflow
import nilearn.image as nimg
import nibabel as nib
import pandas as pd
import numpy as np
import ants
from nilearn import datasets
from pathlib import Path

# Directories
SPINE = Path.cwd().parents[2]
DATA_DIR = SPINE / "data"
PREP_DIR = DATA_DIR / "preprocessed"
UTL_DIR = DATA_DIR / "utils"

# Define functions
def create_df():
    df = pd.DataFrame([unique_labels, region_names, x, y, z, netw_ID]).T
    df.columns = ['RegionID', 'RegionName', 'X', 'Y', 'Z', 'NetworkID']
    return df


# Get the template files in MNI152NLin2009cAsym space
t1_2009 = nib.load(
    tflow.get(
        "MNI152NLin2009cAsym", desc=None, resolution=1, suffix="T1w", extension="nii.gz"
    )
)

t1_2009_mask = nib.load(
    tflow.get(
        "MNI152NLin2009cAsym",
        desc="brain",
        suffix="mask",
        resolution=1,
        extension="nii.gz",
    )
)

# Get the template files in MNI152NLin6Asym space
t1_6 = nib.load(
    tflow.get(
        "MNI152NLin6Asym", desc=None, resolution=1, suffix="T1w", extension="nii.gz"
    )
)

t1_6_mask = nib.load(
    tflow.get(
        "MNI152NLin6Asym",
        desc="brain",
        suffix="mask",
        resolution=1,
        extension="nii.gz",
    )
)

# Mask both MNI
t1_2009_masked = nimg.math_img("a * b", a=t1_2009, b=t1_2009_mask)
t1_6_masked = nimg.math_img("a * b", a=t1_6, b=t1_6_mask)
# Get the AAL atlas in mni 2009
aal_atlas = nib.load(UTL_DIR / "AAL_space-MNI152NLin2009cAsym_res-2.nii.gz")
# Transform from nibabel to Ants
fixed = ants.from_nibabel(t1_6_masked)
moving = ants.from_nibabel(t1_2009_masked)
# Perform registration between t1s
mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")
# Apply registration to AAL
aal_mni6 = ants.apply_transforms(
    fixed=fixed,
    moving=ants.from_nibabel(aal_atlas),
    transformlist=mytx["fwdtransforms"],
    interpolator="genericLabel",
)
# Convert to nibabel format and save
aal_mni6_nib = ants.to_nibabel(aal_mni6)
nib.save(aal_mni6_nib, UTL_DIR / "AAL_space-MNI152NLin6Asym_res-1.nii.gz")

# Now, let's get the coordinates of the centers of the brain regions
# Transform into array
aal = aal_mni6_nib.get_fdata()
unique_labels = np.unique(aal)
unique_labels = np.delete(unique_labels, 0) # zero is not a region
sz = unique_labels.shape[0]
centers = []
# Loop through the different labels (e.g., 2001...) and get the mean X,Y,Z coordinate for that brain region
for i, n in enumerate(unique_labels):
    ind_N = np.where(aal == n)
    centers.append(np.array(ind_N).mean(axis=1))
cent_arr = np.array(centers).squeeze()
#Put into columns
x = cent_arr[:, 0]
y = cent_arr[:, 1]
z = cent_arr[:, 2]
netw_ID = ['netw'] * 90
# Get the region names 
aal_files = datasets.fetch_atlas_aal(version="SPM12")
region_names = aal_files.labels[:90]

df = create_df()
df.to_csv(UTL_DIR / 'AAL_space-MNI152NLin6Asym_coordinates.csv')

# Save a copy to Windows Path
windows_path = '/mnt/c/Users/leo_r/Documenti/R/win-library/4.1/LQT/extdata/Other_Atlases'
nib.save(aal_mni6_nib, windows_path + '/AAL_space-MNI152NLin6Asym_res-1.nii.gz')
df.to_csv(windows_path + '/AAL_space-MNI152NLin6Asym_res-1.csv')

# %%
