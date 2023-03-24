#%%
import matplotlib.pyplot as plt    
import numpy as np

# Let's import the hopf model
from neurolib.models.hopf import HopfModel

# Some useful functions are provided here
import neurolib.utils.functions as func

# a nice color map
plt.rcParams['image.cmap'] = 'plasma'

from petTOAD_loadXCP import *

#%%
sc_pre = get_sc()
sc_pre[sc_pre < 0] = 0
# Prevent full synchronization of the model
sc_norm = sc_pre * 0.2 / sc_pre.max()
model = HopfModel(Cmat = sc_norm, Dmat = np.zeros([len(sc_norm), len(sc_norm)]))

model.params['w'] = 1.0
model.params['signalV'] = 0
model.params['duration'] = 20 * 1000 
model.params['sigma_ou'] = 0.07
model.params['K_gl'] = 4

model.run(chunkwise=True)
#%%
plt.plot(model.t, model.x[::1, :].T, alpha=0.8);
plt.xlim(0, 200)
plt.xlabel("t [ms]")


fig, axs = plt.subplots(1, 2, figsize=(8, 2))
axs[0].imshow(func.fc(model.x[:, -10000:]))
axs[1].plot(model.t, model.x[::5, :].T, alpha=0.8);

# scores = [func.matrix_correlation(func.fc(model.x[:, -int(5000/model.params['dt']):]), fcemp) for fcemp in ds.FCs]
# print("Correlation per subject:", [f"{s:.2}" for s in scores])
# print("Mean FC/FC correlation: {:.2f}".format(np.mean(scores)))
# # %%

# %%
from neurolib.utils.signal import BOLDSignal
# %%
bold = BOLDSignal.from_model_output(model, group="", time_in_ms=True)
bold.description = "Simulated BOLD of the Hopf model with Schaefer200 SC atlas"
# %%
bold.sampling_frequency
bold.start_time
bold.end_time
# %%
