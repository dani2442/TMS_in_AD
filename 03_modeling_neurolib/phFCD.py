# %%
import demean
import warnings
import numpy as np

from scipy import signal
from numba import jit 

import BOLDFilters


# ===================================================================
discardOffset = 0
        
def tril_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.tril(np.ones(N), k=k).T
    )  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (
        col_i,
        row_i,
    )  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


def triu_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.triu(np.ones(N), k=k).T
    )  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (
        col_i,
        row_i,
    )  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


# ==================================================================
# Computes the mean of the matrix
# ==================================================================
@jit(nopython=True)
def mean(x, axis=None):
    if axis == None:
        return np.sum(x, axis) / np.prod(x.shape)
    else:
        return np.sum(x, axis) / x.shape[axis]


@jit(nopython=True)
def adif(a, b):
    if np.abs(a - b) > np.pi:
        c = 2 * np.pi - np.abs(a - b)
    else:
        c = np.abs(a - b)
    return c


@jit(nopython=True)
def numba_PIM(phases, N, Tmax, dFC, PhIntMatr):
    T = np.arange(discardOffset, Tmax - discardOffset + 1)
    for t in T:
        for i in range(N):
            for j in range(i + 1):
                dFC[i, j] = np.cos(adif(phases[i, t - 1], phases[j, t - 1]))
                dFC[j, i] = dFC[i, j]
        PhIntMatr[t - discardOffset] = dFC
    return PhIntMatr


def calc_PIM(
    ts, applyFilters=False, removeStrongArtefacts=False
):  # Compute the Phase-Interaction Matrix of an input BOLD signal
    if not np.isnan(ts).any():  # No problems, go ahead!!!
        (N, Tmax) = ts.shape
        npattmax = Tmax - (2 * discardOffset - 1)  # calculates the size of phfcd matrix
        # Data structures we are going to need...
        phases = np.zeros((N, Tmax))
        dFC = np.zeros((N, N))
        # PhIntMatr = np.zeros((npattmax, int(N * (N - 1) / 2)))  # The int() is not needed, but... (see above)
        PhIntMatr = np.zeros((npattmax, N, N))
        # syncdata = np.zeros(npattmax)

        # Filters seem to be always applied...
        if applyFilters:
            ts_filt = BOLDFilters.BandPassFilter(
                ts, removeStrongArtefacts=removeStrongArtefacts
            )  # zero phase filter the data
        else:
            ts_filt = ts

        for n in range(N):
            Xanalytic = signal.hilbert(demean.demean(ts_filt[n, :]))
            phases[n, :] = np.angle(Xanalytic)

        PhIntMatr = numba_PIM(phases, N, Tmax, dFC, PhIntMatr)

    else:
        warnings.warn(
            "############ Warning!!! PhaseInteractionMatrix.from_fMRI: NAN found ############"
        )
        PhIntMatr = np.array([np.nan])
    # ======== sometimes we need to plot the matrix. To simplify the code, we save it here if needed...
    # if saveMatrix:
    #     import scipy.io as sio
    #     sio.savemat(save_file + '.mat', {name: PhIntMatr})
    return PhIntMatr


# ==================================================================
# numba_phFCD: convenience function to accelerate computations
# ==================================================================
@jit(nopython=True)
def numba_phFCD(phIntMatr_upTri, npattmax, size_kk3):
    phfcd = np.zeros((size_kk3))
    kk3 = 0

    for t in range(npattmax - 2):
        p1_sum = np.sum(phIntMatr_upTri[t : t + 3, :], axis=0)
        p1_norm = np.linalg.norm(p1_sum)
        for t2 in range(t + 1, npattmax - 2):
            p2_sum = np.sum(phIntMatr_upTri[t2 : t2 + 3, :], axis=0)
            p2_norm = np.linalg.norm(p2_sum)

            dot_product = np.dot(p1_sum, p2_sum)
            phfcd[kk3] = dot_product / (p1_norm * p2_norm)
            kk3 += 1
    return phfcd


def phFCD(
    ts, applyFilters=False, removeStrongArtefacts=False
):  # Compute the FCD of an input BOLD signal
    phIntMatr = calc_PIM(
        ts, applyFilters=applyFilters, removeStrongArtefacts=removeStrongArtefacts
    )  # Compute the Phase-Interaction Matrix
    if not np.isnan(phIntMatr).any():  # No problems, go ahead!!!
        (N, Tmax) = ts.shape
        npattmax = Tmax - (2 * discardOffset - 1)  # calculates the size of phfcd vector
        size_kk3 = int(
            (npattmax - 3) * (npattmax - 2) / 2
        )  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...
        Isubdiag = tril_indices_column(
            N, k=-1
        )  # Indices of triangular lower part of matrix
        phIntMatr_upTri = np.zeros(
            (npattmax, int(N * (N - 1) / 2))
        )  # The int() is not needed, but... (see above)
        for t in range(npattmax):
            phIntMatr_upTri[t, :] = phIntMatr[t][Isubdiag]
        phfcd = numba_phFCD(
            phIntMatr_upTri,
            npattmax,
            size_kk3,
        )
    else:
        warnings.warn("############ Warning!!! phFCD.from_fMRI: NAN found ############")
        phfcd = np.array([np.nan])
    # if saveMatrix:
    #     buildMatrixToSave(phfcd, npattmax - 2)
    return phfcd