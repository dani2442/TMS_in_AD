import numpy as np
import nibabel as nib
from nilearn import image, input_data, datasets, plotting
from nilearn.decomposition import CanICA

# Load the functional MRI data for multiple subjects
func_filenames = ['subject1.nii.gz', 'subject2.nii.gz', 'subject3.nii.gz']

# Create a masker object to extract the brain from the functional images
masker = input_data.NiftiMasker(mask_strategy='epi')

# Preprocess the functional MRI data using standard techniques such as slice timing correction, motion correction, etc.
func_preprocessed = []
for func_filename in func_filenames:
    func_img = nib.load(func_filename)
    func_masked = masker.fit_transform(func_img)
    func_preprocessed.append(func_masked)

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

