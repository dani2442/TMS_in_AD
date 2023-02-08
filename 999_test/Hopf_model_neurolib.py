import numpy as np
import matplotlib.pyplot as plt

from neurolib.utils.functions import fc, matrix_correlation, ts_kolmogorov
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.loadData import Dataset
from neurolib.models.hopf import HopfModel

ds = Dataset('hcp')
model = HopfModel(Cmat = ds.Cmat, Dmat = ds.Dmat)
model.params['duration'] = 5 * 60 * 1000

def fmri_fit(model):
    t_bold = model.BOLD.t_BOLD > 10000
    sim_BOLD = model.BOLD.BOLD[:, t_bold]
    fits = {}
    fits['fc'] = np.mean([matrix_correlation(fc(sim_BOLD), f) for f in ds.FCs])
    fits['fcd'] = np.mean([ts_kolmogorov(sim_BOLD, b) for b in ds.BOLDs])
    return fits

def evaluate(traj):
    model = search.getModelFromTraj(traj)
    model.run(chunkwise=True, bold = True)
    fits = fmri_fit(model)
    search.saveToPypet(fits, traj)

parameters = ParameterSpace({'K_gl': np.linspace(0, 500.0, 5), 'sigma_ou': [0.2, 0.5]})
search = BoxSearch(model, parameters, evaluate)
search.run()