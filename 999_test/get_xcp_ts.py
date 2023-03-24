#%% 

from nilearn import datasets
# Import the needed packages
import numpy as np
import pandas as pd

from bids import BIDSLayout

from nilearn.interfaces import fmriprep
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image as nimg
from nilearn.signal import clean 

import nibabel as nib

import scipy.io as sio

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
atlas_filename = atlas.maps
use_filter = True
filt_low = 0.08
filt_high = 0.008
TR = 3.0
func_image = '/mnt/e/petTOAD/data/preprocessed/fmriprep/sub-ADNI109S6213/ses-M00/func/sub-ADNI109S6213_ses-M00_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'


masker = NiftiLabelsMasker(labels_img=atlas_filename)
signal = masker.fit_transform(func_image)

# Here we are applying a 36P confound regression strategy with motion parameters, derivatives and power of parameters and derivatives. We are excluding the first 4 timepoints. No scrubbing.
confounds, mask = fmriprep.load_confounds(func_image, strategy=('motion', 'wm_csf'), 
                                                    motion='full', wm_csf='full')
confounds_selected = confounds.iloc[4:,:]
signal_selected = signal[4:,:]

if use_filter == True:
    cleaned_signal = clean(signal_selected, detrend=True, standardize='zscore', confounds=confounds_selected, standardize_confounds=True, filter='butterworth', low_pass=filt_low, high_pass=filt_high, t_r=TR, ensure_finite=False)
else:
    cleaned_signal = clean(signal_selected, detrend=True, standardize='zscore', confounds=confounds_selected, standardize_confounds=True, ensure_finite=False, t_r = TR)
    
ts = cleaned_signal.T
np.save('/home/riccardo/petTOAD/results/subject_list_timeseries_HC.npy', np.array(ts))

# %%
