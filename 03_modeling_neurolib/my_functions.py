import logging
import numpy as np
import scipy.signal 
import scipy.stats
import numba
import warnings
import scipy.io as sio



"""Collection of useful functions for data processing.
"""


def kuramoto(traces, smoothing=0.0, distance=10, prominence=5):
    """
    Computes the Kuramoto order parameter of a timeseries which is a measure for synchrony.
    Can smooth timeseries if there is noise.
    Peaks are then detected using a peakfinder. From these peaks a phase is derived and then
    the amount of phase synchrony (the Kuramoto order parameter) is computed.

    :param traces: Multidimensional timeseries array
    :type traces: numpy.ndarray
    :param smoothing: Gaussian smoothing strength
    :type smoothing: float, optional
    :param distance: minimum distance between peaks in samples
    :type distance: int, optional
    :param prominence: vertical distance between the peak and its lowest contour line
    :type prominence: int, optional

    :return: Timeseries of Kuramoto order paramter
    :rtype: numpy.ndarray
    """
    #@numba.njit
    # def _estimate_phase(maximalist, n_times):
    #     lastMax = 0
    #     phases = np.empty((n_times), dtype=np.float64)
    #     n = 0
    #     for m in maximalist:
    #         for t in range(lastMax, m):
    #             # compute instantaneous phase
    #             phi = 2 * np.pi * float(t - lastMax) / float(m - lastMax)
    #             phases[n] = phi
    #             n += 1
    #         lastMax = m
    #     phases[-1] = 2 * np.pi
    #     return phases
    def my_estimate_phase(ts):
        N, Tmax = ts.shape
        # Data structures we are going to need...
        phases_emp = np.zeros([N, Tmax])
        # Time-series of the phases
        for n in range(N):
            Xanalytic = scipy.signal.hilbert(ts[n, :])  # demean.demean
            phases_emp[n, :] = np.angle(Xanalytic)
        return phases_emp
    
    #@numba.njit
    def _estimate_r(ntraces, times, phases):
        kuramoto = np.empty((times), dtype=np.float64)
        for t in range(times):
            R = 1j*0
            for n in range(ntraces):
                R += np.exp(1j * phases[n, t])
            R /= ntraces
            kuramoto[t] = np.absolute(R)
        return kuramoto

    nTraces, nTimes = traces.shape
    phases = my_estimate_phase(traces)
    
    # for n in range(nTraces):
    #     a = traces[n]
    #     # find peaks
    #     if smoothing > 0:
    #         # smooth data
    #         a = scipy.ndimage.filters.gaussian_filter(traces[n], smoothing)
    #     maximalist = scipy.signal.find_peaks(a, distance=distance,
    #                                          prominence=prominence)[0]
    #     maximalist = np.append(maximalist, len(traces[n])-1).astype(int)

    #     if len(maximalist) > 1:
    #         phases[n, :] = _estimate_phase(maximalist, nTimes)
    #     else:
    #         logging.warning("Kuramoto: No peaks found, returning 0.")
    #         return 0
    # determine kuramoto order paramter
    kuramoto = _estimate_r(nTraces, nTimes, phases)
    return kuramoto


def matrix_correlation(M1, M2):
    """Pearson correlation of the lower triagonal of two matrices.
    The triangular matrix is offset by k = 1 in order to ignore the diagonal line

    :param M1: First matrix
    :type M1: numpy.ndarray
    :param M2: Second matrix
    :type M2: numpy.ndarray
    :return: Correlation coefficient
    :rtype: float
    """
    cc = np.corrcoef(M1[np.triu_indices_from(M1, k=1)], M2[np.triu_indices_from(M2, k=1)])[0, 1]
    return cc


def weighted_correlation(x, y, w):
    """Weighted Pearson correlation of two series.

    :param x: Timeseries 1
    :type x: list, np.array
    :param y: Timeseries 2, must have same length as x
    :type y: list, np.array
    :param w: Weight vector, must have same length as x and y
    :type w: list, np.array
    :return: Weighted correlation coefficient
    :rtype: float
    """

    def weighted_mean(x, w):
        """Weighted Mean"""
        return np.sum(x * w) / np.sum(w)

    def weighted_cov(x, y, w):
        """Weighted Covariance"""
        return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))


def fc(ts):
    """Functional connectivity matrix of timeseries multidimensional `ts` (Nxt).
    Pearson correlation (from `np.corrcoef()` is used).

    :param ts: Nxt timeseries
    :type ts: numpy.ndarray
    :return: N x N functional connectivity matrix
    :rtype: numpy.ndarray
    """
    fc = np.corrcoef(ts)
    fc = np.nan_to_num(fc)  # remove NaNs
    return fc


# Calculate and save average FC for the group
def group_fc(group):
    """Calculates and saves the average functional connectivity matrix of 
    a group of multidimensional timeserieses `ts` (Nxt).

    :param group: dictionary containing timeseries
    :type group: dict
    :param save_dir: path dir where to save the avg FC
    :type path object
    :return: N x N average functional connectivity matrix
    :rtype: numpy.ndarray
    """
    fcs = []
    for ts in group.values():
        fcs.append(fc(ts))
    avg_fc = np.array(fcs).mean(axis=0)
    return avg_fc




def fcd(ts, windowsize=30, stepsize=5):
    """Computes FCD (functional connectivity dynamics) matrix, as described in Deco's whole-brain model papers.
    Default paramters are suited for computing FCS matrices of BOLD timeseries:
    A windowsize of 30 at the BOLD sampling rate of 0.5 Hz equals 60s and stepsize = 5 equals 10s.

    :param ts: Nxt timeseries
    :type ts: numpy.ndarray
    :param windowsize: Size of each rolling window in timesteps, defaults to 30
    :type windowsize: int, optional
    :param stepsize: Stepsize between each rolling window, defaults to 5
    :type stepsize: int, optional
    :return: T x T FCD matrix
    :rtype: numpy.ndarray
    """
    t_window_width = int(windowsize)  # int(windowsize * 30) # x minutes
    stepsize = stepsize  # ts.shape[1]/N
    corrFCs = []
    try:
        counter = range(0, ts.shape[1] - t_window_width, stepsize)

        for t in counter:
            ts_slice = ts[:, t : t + t_window_width]
            corrFCs.append(np.corrcoef(ts_slice))

        FCd = np.empty([len(corrFCs), len(corrFCs)])
        f1i = 0
        for f1 in corrFCs:
            f2i = 0
            for f2 in corrFCs:
                FCd[f1i, f2i] = np.corrcoef(f1.reshape((1, f1.size)), f2.reshape((1, f2.size)))[0, 1]
                f2i += 1
            f1i += 1

        return FCd
    except:
        return 0

def phase_int_matrix(ts): 
    """Computes the Phase-Interaction Matrix explained as defined in: 
    Lopez-Gonzalez [2020] (https://doi.org/10.1038/s42003-021-02537-9)
        Translated to Python by Xenia Kobeleva
        Revised by Gustavo Patow
        Refactored by Gustavo Patow
        Modified for neurolib by Riccardo Leone

    :param ts: N-dimensional timeseries
    :type ts: np.ndarray

    :return: phase interaction matrix
    :rtype: np.ndarray

    """

    def demean(x,dim=0):
        import numpy.matlib as mtlib
        dims = x.size
        return x - mtlib.tile(np.mean(x,dim), dims)  # repmat(np.mean(x,dim),dimrep)


    (N, Tmax) = ts.shape
    npattmax = Tmax

    if not np.isnan(ts).any():  # No problems, go ahead!!!
        # Data structures we are going to need...
        phases = np.zeros((N, Tmax))
        dFC = np.zeros((N, N))
        PhIntMatr = np.zeros((npattmax, N, N))
        for n in range(N):
            
            Xanalytic = scipy.signal.hilbert(demean(ts[n, :]))
            phases[n, :] = np.angle(Xanalytic)

        # Isubdiag = tril_indices_column(N, k=-1)  # Indices of triangular lower part of matrix
        T = np.arange(0, Tmax + 1)
        for t in T:
            dFC = np.cos(np.subtract.outer(phases[:, t - 1], phases[:, t - 1]))
            PhIntMatr[t-1] = dFC
    else:
        warnings.warn('### Warning!!! PhaseInteractionMatrix: NAN found ############')
        PhIntMatr = np.array([np.nan])
    # if we need to save the matrix for some later use...
    # if saveMatrix:
    #     import scipy.io as sio
    #     sio.savemat(save_file + '.mat', {name: PhIntMatr})
    return PhIntMatr



# From [Deco2019]: Comparing empirical and simulated FCD.
# We measure KS distance between the upper triangular elements of the empirical and simulated FCD matrices
# (accumulated over all participants). The KS distance quantifies the maximal difference between the cumulative
# distribution WholeBrain of the 2 samples.

def phFCD(ts, windowsize_phase = 3):  # Compute the ohFCD of an input BOLD signal
    """ 
    Computes the Phase Functional Connectivity Dynamics (phFCD) as defined in 
    Lopez-Gonzalez [2020] (https://doi.org/10.1038/s42003-021-02537-9)

        - Translated to Python by Xenia Kobeleva
        - Revised by Gustavo Patow
        - Translated to neurolib by Riccardo Leone

    This function computes the phase Functional Connectivity Dynamics (phFCD) of a given input BOLD 
    signal using the cosine similarity between two matrices. It takes two inputs, the BOLD signal 
    "ts" and the window length "window_length" which represents the number of TR. 
    At first, it calculates the size of th phFCD vector based on the size of the input signal while 
    taking the window length into consideration. Then it uses an auxiliary function to compute the 
    Phase-Interaction Matrix, which is stored in the "phIntMatr" variable. Next, the function checks 
    if there are NAN values in the input Phase-Interaction Matrix. If there aren't, for each time step, 
    it computes the cosine similarity between the mean of the upper triangular parts of two corresponding
    dFC matrices The output is stored in the "phfcd" variable. 
    If there are NAN values in the input Phase-Interaction Matrix, the function outputs a warning message 
    and only a single NAN value is stored in the "phfcd" variable. 
    It then returns the "phfcd" variable. 
    Currently tested only for window_length = 3 (3 TRs)

    :param ts: N-dimensional timeseries
    :type ts: np.ndarray
    :param window_length: the window_length on which to calculate the phase functional connectivity dynamics
    :type ts: int
    

    :return: the phFCD of an input BOLD signal.
    :rtype: np.ndarray
    """
   
    (N, Tmax) = ts.shape
    npattmax = Tmax  # calculates the size of phfcd vector
    size_kk3 = int((npattmax - windowsize_phase) * (npattmax - (windowsize_phase-1)) / 2)  # The int() is not needed because N*(N-1) is always even, but "it will produce an error in the future"...

    phIntMatr = phase_int_matrix(ts)  # Compute the Phase-Interaction Matrix
    triu_indices = np.triu_indices(phIntMatr.shape[1], k = 1)

    if not np.isnan(phIntMatr).any():  # No problems, go ahead!!!
        phIntMatr_upTri = np.zeros((npattmax, int(N * (N - 1) / 2)))  # The int() is not needed, but... (see above)
        for t in range(npattmax):
            phIntMatr_upTri[t,:] = phIntMatr[t][triu_indices]
        phfcd = np.zeros((size_kk3))
        kk3 = 0
        for t in range(npattmax - (windowsize_phase-1)):
            p1 = np.mean(phIntMatr_upTri[t:t + windowsize_phase, :], axis=0)
            for t2 in range(t + 1, npattmax - (windowsize_phase-1)):
                p2 = np.mean(phIntMatr_upTri[t2:t2 + windowsize_phase, :], axis=0)
                phfcd[kk3] = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))  # this (phFCD) what I want to get
                kk3 = kk3 + 1
    else:
        warnings.warn('############ Warning in phFCD: NAN found ############')
        phfcd = np.array([np.nan])
    # if saveMatrix:
    #     buildMatrixToSave(phfcd, npattmax - (window_length-1))
    return phfcd


def matrix_kolmogorov(m1, m2):
    """Computes the Kolmogorov distance between the distributions of lower-triangular entries of two matrices
    See: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test

    :param m1: matrix 1
    :type m1: np.ndarray
    :param m2: matrix 2
    :type m2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    # get the values of the lower triangle
    if m1.ndim > 1:
        triu_ind1 = np.triu_indices(m1.shape[0], k=1)
        m1_vals = m1[triu_ind1]

        triu_ind2 = np.triu_indices(m2.shape[0], k=1)
        m2_vals = m2[triu_ind2]
    else:
        m1_vals = m1
        m2_vals = m2
    # return the distance, omit p-value
    return scipy.stats.ks_2samp(m1_vals, m2_vals)[0]


def ts_kolmogorov(ts1, ts2, **fcd_kwargs):
    """Computes kolmogorov distance between two timeseries.
    This is done by first computing two FCD matrices (one for each timeseries)
    and then measuring the Kolmogorov distance of the upper triangle of these matrices.

    :param ts1: Timeseries 1
    :type ts1: np.ndarray
    :param ts2: Timeseries 2
    :type ts2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    fcd1 = fcd(ts1, **fcd_kwargs)
    fcd2 = fcd(ts2, **fcd_kwargs)

    return matrix_kolmogorov(fcd1, fcd2)


def ts_kolmogorov_phfcd(ts1, ts2, **fcd_kwargs):
    """Computes kolmogorov distance between the phase func. conn. dynamics of two timeseries.
    This is done by first computing two phFCD matrices (one for each timeseries)
    and then measuring the Kolmogorov distance of these arrays.

    :param ts1: Timeseries 1
    :type ts1: np.ndarray
    :param ts2: Timeseries 2
    :type ts2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    phFCD1 = phFCD(ts1, **fcd_kwargs)
    phFCD2 = phFCD(ts2, **fcd_kwargs)

    return scipy.stats.ks_2samp(phFCD1, phFCD2)[0]


def group_fc_dynamics(group, **kwargs):
    fcds1 = []
    phFCDs1 = []
    for ts1 in group.values():
        fcd1 = fcd(ts1, **kwargs)
        fcds1.append(fcd1)

        phFCD1 = phFCD(ts1)
        phFCDs1.append(phFCD1)
    
    return np.array(fcds1), np.array(phFCDs1)

def group_kolmogorov(group1, group2, **kwargs):
    """Computes kolmogorov distance between the fcd and phfcds of two groups of subjects.
    Useful for group comparisons (healthy vs. diseased)
    :param group11: group 1, a dictionary (e.g., {subjs:ts})
    :type group1: dict 
    :param group2: group2, a dictionary (e.g., {subjs:ts})
    :type group2: np.ndarray
    :return: 2-sample KS statistics
    :rtype: float
    """
    fcds1, phfcds1 = group_fc_dynamics(group1, **kwargs)
    fcds2, phfcds2 = group_fc_dynamics(group2, **kwargs)

    triu_ind1 = np.triu_indices(fcds1.shape[0], k=1)
    triu_ind2 = np.triu_indices(fcds2.shape[0], k=1)

    fcd1_vals = fcds1[triu_ind1]
    fcd2_vals = fcds2[triu_ind2]


    return scipy.stats.ks_2samp(fcd1_vals, fcd2_vals), scipy.stats.ks_2samp(phfcds1, phfcds2)


def calc_and_save_group_stats(group, save_dir, **kwargs):
    """Calculates and saves the cumulative fcd and phfcds of a group of subjects.
    Useful for comparing to model simulations
    :param group: group, a dictionary (e.g., {subjs:ts})
    :type group: dict 
    :return: fc, fcd, phfcd

    """
    import scipy.io as sio
    savename = save_dir / "group_stats.mat"
    try:
        m = sio.loadmat(savename)
        fc = m['fc']
        flat_fcd_arr = m['fcd']
        flat_phfcd_arr = m['phFCD']
    except:
        res_dict = {}
        print("Calculating group stats...")
        print("Calculating group FC...")
        fc = group_fc(group)
        print("Calculating group dynamics...")
        fcd_arr, phfcd_arr = group_fc_dynamics(group, **kwargs)
        flat_fcd_arr = fcd_arr.flatten()
        flat_phfcd_arr = phfcd_arr.flatten()
        res_dict['fc'] = fc
        res_dict['fcd'] = flat_fcd_arr
        res_dict['phFCD'] = flat_phfcd_arr
        sio.savemat(savename, res_dict)

    return fc, flat_fcd_arr.squeeze(), flat_phfcd_arr.squeeze()


def getPowerSpectrum(activity, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns a power spectrum using Welch's method.

    :param activity: One-dimensional timeseries
    :type activity: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """
    # convert to one-dimensional array if it is an (1xn)-D array
    if activity.shape[0] == 1 and activity.shape[1] > 1:
        activity = activity[0]
    assert len(activity.shape) == 1, "activity is not one-dimensional!"

    f, Pxx_spec = scipy.signal.welch(
        activity,
        1000 / dt,
        window="hanning",
        nperseg=int(spectrum_windowsize * 1000 / dt),
        scaling="spectrum",
    )
    f = f[f < maxfr]
    Pxx_spec = Pxx_spec[0 : len(f)]
    if normalize:
        Pxx_spec /= np.max(Pxx_spec)
    return f, Pxx_spec


def getMeanPowerSpectrum(activities, dt, maxfr=70, spectrum_windowsize=1.0, normalize=False):
    """Returns the mean power spectrum of multiple timeseries.

    :param activities: N-dimensional timeseries
    :type activities: np.ndarray
    :param dt: Simulation time step
    :type dt: float
    :param maxfr: Maximum frequency in Hz to cutoff from return, defaults to 70
    :type maxfr: int, optional
    :param spectrum_windowsize: Length of the window used in Welch's method (in seconds), defaults to 1.0
    :type spectrum_windowsize: float, optional
    :param normalize: Maximum power is normalized to 1 if True, defaults to False
    :type normalize: bool, optional

    :return: Frquencies and the power of each frequency
    :rtype: [np.ndarray, np.ndarray]
    """

    powers = np.zeros(getPowerSpectrum(activities[0], dt, maxfr, spectrum_windowsize)[0].shape)
    ps = []
    for rate in activities:
        f, Pxx_spec = getPowerSpectrum(rate, dt, maxfr, spectrum_windowsize)
        ps.append(Pxx_spec)
        powers += Pxx_spec
    powers /= len(ps)
    if normalize:
        powers /= np.max(powers)
    return f, powers

def calculate_sw_fc(ts, T, N, window_size=30, step_size=1):
    num_windows = (T - window_size) // step_size + 1
    corr_ts = np.zeros((num_windows, N, N))
    # 1.1 Calculate for each time window the pairwise correlation between the given nodes activities
    for win in range(num_windows):
        start = win * step_size
        end = start + window_size
        corr_ts[win,:,:] = np.corrcoef(ts[:, start:end])
    return corr_ts

def calculate_metaconnectivity(ts, window_size=10, step_size=1):
    """
    Calculates metaconnectivity as described in "White-matter degradation and dynamical compensation support age-related functional alterations in human brain" 
    by Petkoski et al @ Cerebral Cortex, 2023, 33, 6241–6256
    # From the Methods:
    # In addition we calculated higher order interactions between brain regions using metaconnectivity (MC) (Arbabyazd et al. 2020). 
    # Exactly as typical static FC analysis ignores time, the previously mentioned FCD analyses ignore space. However, FC reconfiguration may occur at different speeds for 
    # different sets of links (Lombardo et al. 2020). Furthermore, the fluctuations of certain FC links may coincide with the fluctuation of other FC links, but at the same
    # time be relatively independent from the fluctuation. Therefore, we compute a different dFC speed distribution for different sets of links, which constitute spatial dFC modules.
    # MC is defined as correlation between linkwise timeseries consisting of the pairwise correlations between the given nodes at each window. Hence it represents a fourth order 
    # statistics between nodes’ dynamics. 

    Args:
    ts (np.array): the timeseries of the subject in N x T format (regions x timepoints)
    window_size (int): number of TR to take for windowsize (TR x window_size gives you the length of the window in seconds). Defaults to 10.
    step_size (int): number of TR to skip between each window (TR x window_step gives how much to move the window in seconds each time). Defaults to 1 timestep.

    Returns:
    meta (np.array): the matrix of the metaconnectivity

    """
    N, T = ts.shape
    corr_ts = calculate_sw_fc(ts, T, N, window_size=window_size, step_size=step_size)
    # 2.1 Get the linkwise timeseries of the pairwise correlations previously found...
    corr_timeseries = []
    for i in range(N):
        for j in range(i+1, N):
            corr_timeseries.append(corr_ts[:, i, j])
    arr_corr_timeseries = np.array(corr_timeseries)
    # 2.2 ... and calculate the correlations
    meta = np.corrcoef(arr_corr_timeseries)
    return meta