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

#%%
import templateflow.api as tflow
import nilearn.image as nimg
import nibabel as nib
import pandas as pd
import numpy as np
import ants
from nilearn import datasets
from pathlib import Path
import json

# Directories
SPINE = Path.cwd().parents[2]
DATA_DIR = SPINE / "data"
PREP_DIR = DATA_DIR / "preprocessed"
UTL_DIR = DATA_DIR / "utils"

# Opening JSON file
f = open(UTL_DIR / 'AAL_space-MNI152NLin6_res-1x1x1.json')

  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
for i in data['emp_details']:
    print(i)
  
# Closing file
f.close()