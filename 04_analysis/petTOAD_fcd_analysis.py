#%%
import neurolib.utils.functions as func
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from numpy import linalg as LA
from petTOAD_setup import *


#%%
eigvecs = []
for pt, ts in all_HC_fMRI.items():
    ph_M = func.phase_int_matrix(ts)
    for i in range(ph_M.shape[0]):
        eigval, eigvec = LA.eig(ph_M[i])
        eigvecs.append(eigvec[eigval.argmax()])
real_eigvecs = np.real(np.array(eigvecs))
#%%
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(real_eigvecs)
kmeans.predict(real_eigvecs)
