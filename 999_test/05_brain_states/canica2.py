#%%
import numpy as np
import nibabel as nib
from nilearn import image,datasets, plotting
from nilearn.maskers import NiftiMasker
from nilearn.decomposition import CanICA
from bids import BIDSLayout
from pathlib import Path

# Directories
SPINE = Path.cwd().parents[2]
DATA_DIR = SPINE / "data"
PREP_DIR = DATA_DIR / "preprocessed"
FPRP_DIR = PREP_DIR / "fmriprep"
UTL_DIR = DATA_DIR / "utils"
WMH_DIR = PREP_DIR / "WMH_segmentation"
XCP_DIR = PREP_DIR / "xcp_d"

REL_SES = "M00"

# %% ~~ Load data ~~ %%#
print("Getting the layout...")
layout = BIDSLayout(XCP_DIR, validate=False, config=["bids", "derivatives"])
subjs = layout.get_subjects()
print("Done with the layout...")

masker = NiftiMasker(mask_strategy='epi')

# Preprocess the functional MRI data using standard techniques such as slice timing correction, motion correction, etc.
func_preprocessed = []
for subj in subjs:
    func_denoised = layout.get(subject=subj, desc="denoised", return_type="file", extension = ".nii.gz")
    func_img = func_denoised[0]
    func_masked = masker.fit_transform(func_img)
    func_preprocessed.append(func_masked)

#%%
# Concatenate the preprocessed functional MRI data from multiple subjects
func_concatenated = np.concatenate(func_preprocessed, axis=0)

# Perform CanICA on the concatenated data to extract independent components
canica = CanICA(n_components=20, smoothing_fwhm=6., memory="nilearn_cache", memory_level=2, random_state=0, threshold=3., verbose=10)
canica.fit(func_concatenated)

# Get the independent components and the corresponding time courses
components = canica.components_
timecourses = canica.transform(func_concatenated)

# Calculate the explained variance for each number of components
expl_var = []
for n_components in range(1, 21):
    canica = CanICA(n_components=n_components, smoothing_fwhm=6., memory="nilearn_cache", memory_level=2, random_state=0, threshold=3., verbose=10)
    canica.fit(func_concatenated)
    expl_var.append(canica.explained_variance_)

# Plot the explained variance as a function of the number of components
plotting.plot_prob_atlas(components_img, view_type='filled_contours')
plotting.show()

# Find the optimal number of components that maximize the observed variance
optimal_n_components = np.argmax(expl_var) + 1
print(f'The optimal number of components is {optimal_n_components}')


# %%
