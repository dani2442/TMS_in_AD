#%%# Imports
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn.image as nimg
from pathlib import Path
from bids import BIDSLayout

# Directories
SPINE = Path.cwd().parents[2]
DATA_DIR = SPINE / "data"
PREP_DIR = DATA_DIR / "preprocessed"
FPRP_DIR = PREP_DIR / "fmriprep"
UTL_DIR = DATA_DIR / "utils"
WMH_DIR = PREP_DIR / "WMH_segmentation"
RES_DIR = SPINE / "results"

REL_SES = "M00"


def get_WMH_mni_space(subj):
    wmh_mask_mni = (
        WMH_DIR
        / f"sub-{subj}"
        / f"sub-{subj}_ses-M00_space-MNI152NLin6Asym_desc-fromSubjSpace.nii.gz"
    )
    try:
        wmh_mni = nib.load(wmh_mask_mni)
    except:
        print(f"MNI mask not found for subj-{subj}")

    return wmh_mni

print('Loading df..')
pts_df = pd.read_csv(RES_DIR / "petTOAD_dataframe.csv", index_col=[0])

print('Selecting patients..')
cn_list = pts_df[pts_df["Group_bin"] == "CN"]["PTID"]
mci_list = pts_df[pts_df["Group_bin"] == "MCI"]["PTID"]
cn_wmh_masks = [get_WMH_mni_space(cn) for cn in cn_list]
mci_wmh_masks = [get_WMH_mni_space(mci) for mci in mci_list]
cn_mask = np.zeros(
    [cn_wmh_masks[0].shape[0], cn_wmh_masks[0].shape[1], cn_wmh_masks[0].shape[2]]
)

#%%
cn_100 = nib.load(WMH_DIR / "cn_mean_wmh_mask_100.nii.gz")
# cn_100 = cn_100.get_fdata().astype(int)

# print('Processing CN...')
# for i, cn_wmh in enumerate(cn_wmh_masks[100:]):
#     print(f'Processing {i+1} / {len(cn_wmh_masks)}...')
#     cn_wmh_arr = cn_wmh.get_fdata().astype(int)
#     cn_100 += cn_wmh_arr

# cn_mask_mean = cn_100 / len(cn_wmh_masks[100:])
# cn_mask_img = nimg.new_img_like(cn_wmh_masks[0], cn_mask_mean)
# nib.save(cn_mask_img, WMH_DIR / "cn_mean_wmh_mask_all.nii.gz")

mci_mask = np.zeros(
    [mci_wmh_masks[0].shape[0], mci_wmh_masks[0].shape[1], mci_wmh_masks[0].shape[2]]
)

mci_100 = nib.load(WMH_DIR / "mci_mean_wmh_mask_100.nii.gz")
mci_100 = mci_100.get_fdata().astype(int)
print('Processing MCI...')
for i, mci_wmh in enumerate(mci_wmh_masks[100:]):
    print(f'Processing {i+1} / {len(mci_wmh_masks)}...')
    mci_wmh_arr = mci_wmh.get_fdata().astype(int)
    mci_mask += mci_wmh_arr

mci_mask_mean = mci_mask / len(mci_wmh_masks[100:])
mci_mask_img = nimg.new_img_like(mci_wmh_masks[0], mci_mask_mean)

nib.save(mci_mask_img, WMH_DIR / "mci_mean_wmh_mask_all.nii.gz")

# %%
cn_100 = nib.load(WMH_DIR / "cn_mean_wmh_mask_100.nii.gz")
mci_100 = nib.load(WMH_DIR / "mci_mean_wmh_mask_100.nii.gz")

a = nimg.math_img("a - b", a=mci_100, b=cn_100)
nib.save(a, WMH_DIR / "diff_wmh_mask.nii.gz")

# %%
