"""
D_measure.py

Implementations of the D-measure (an alternative to the Fourier Phase Synchronization Index) 
to quantify regularity/irregularity/determinism in stereo-EEG time series. 

Developed for the paper [Title Redacted], currently under editorial review 
in the journal 'Brain'.

Includes:
- toroD             : Original CPU-based version (reference implementation).
- toroD_joblib_batch: Batch version using joblib (CPU parallelization).
- toroD_torch_batch : Batch version using PyTorch (CPU / CUDA / MPS acceleration).
"""

import numpy as np

# ---------------------------------------------------------------------
# toroD: Simple version of the J-measure based on absolute phases
# ---------------------------------------------------------------------
def toroD(x1, x2, fi, ff):
    """
    Alternative implementation of the J-measure.

    Parameters
    ----------
    x1, x2 : arrays
        1D real-valued arrays (time series).
    fi, ff : int
        Frequency indices (fi inclusive, ff exclusive).

    Returns
    -------
    j_0 : float
        Scalar value of J in the range [0, 1].
    caminata : array
        1D array containing the angles between consecutive steps (trajectory).
    """
    # FFT and phases
    F1 = np.fft.rfft(x1)
    F2 = np.fft.rfft(x2)
    ff1 = np.angle(F1)[fi:ff]
    ff2 = np.angle(F2)[fi:ff]

    if ff1.size < 2:
        return np.nan, np.array([])

    # Consecutive points (φ1, φ2)
    p1 = np.column_stack((ff1[:-1], ff2[:-1]))   # (N-1, 2)
    p2 = np.column_stack((ff1[1:],  ff2[1:]))    # (N-1, 2)

    # Normalized vectors
    eps = np.finfo(float).eps
    n1 = np.maximum(np.hypot(p1[:, 0], p1[:, 1]), eps)
    n2 = np.maximum(np.hypot(p2[:, 0], p2[:, 1]), eps)

    v1 = p1 / n1[:, None]
    v2 = p2 / n2[:, None]

    # 2D Dot and Cross products
    dotp = np.sum(v1 * v2, axis=1)
    cruz = v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]

    # Signed angle between v1 and v2
    angulos = np.arctan2(cruz, dotp)

    # J-measure (complex circular mean)
    j_0 = 1.0 - np.abs(np.mean(np.exp(1j * angulos)))
    caminata = angulos

    return j_0

# ---------------------------------------------------------------------
# toroD_joblib_batch: Batch version using joblib (CPU)
# ---------------------------------------------------------------------
def toroD_joblib_batch(a, b, fi, ff):
    """
    Evaluates toroD in parallel for all columns using joblib.

    Parameters
    ----------
    a, b : arrays
        Arrays of shape (n, m) with time series in columns.
    fi, ff : int
        Frequency indices (fi inclusive, ff exclusive).

    Returns
    -------
    J : array
        1D array of length m containing J values per column.
    """
    from joblib import Parallel, delayed

    m = a.shape[1]

    J_list = Parallel(n_jobs=-1)(
        delayed(toroD)(a[:, i], b[:, i], fi, ff)
        for i in range(m)
    )
    return np.asarray(J_list)

# ---------------------------------------------------------------------
# toroD_torch_batch: Batch version using PyTorch
# ---------------------------------------------------------------------
def toroD_torch_batch(A, B, fi, ff, device=None):
    """
    Batch implementation of toroD using PyTorch.

    Parameters
    ----------
    A, B : NumPy arrays
        Arrays of shape (n, m) with time series in columns.
    fi, ff : int
        Frequency indices (fi inclusive, ff exclusive).
    device : torch.device, optional
        Computing device (auto-selects: CUDA -> MPS -> CPU if None).

    Returns
    -------
    J : tensor
        1D tensor of shape (m,) containing the J value per column pair.
    """
    import torch

    # Automatic device selection
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load data as float32 with shape (m, n) on the device
    A_t = torch.from_numpy(A.T.astype(np.float32)).to(device)
    B_t = torch.from_numpy(B.T.astype(np.float32)).to(device)  # (m, n)

    # Real FFT (row-wise)
    F1 = torch.fft.rfft(A_t, dim=1)
    F2 = torch.fft.rfft(B_t, dim=1)

    ff1 = torch.angle(F1[:, fi:ff])    # (m, K)
    ff2 = torch.angle(F2[:, fi:ff])

    if ff1.size(1) < 2:
        return torch.full((A_t.size(0),), float("nan"), device=device)

    # Consecutive points
    p1_1 = ff1[:, :-1]   # φ1(k)
    p1_2 = ff2[:, :-1]   # φ2(k)
    p2_1 = ff1[:, 1:]    # φ1(k+1)
    p2_2 = ff2[:, 1:]    # φ2(k+1)

    # Norms and normalized vectors
    eps = torch.finfo(ff1.dtype).eps

    n1 = torch.sqrt(p1_1**2 + p1_2**2)
    n2 = torch.sqrt(p2_1**2 + p2_2**2)
    n1 = torch.clamp(n1, min=eps)
    n2 = torch.clamp(n2, min=eps)

    v1_1 = p1_1 / n1
    v1_2 = p1_2 / n1
    v2_1 = p2_1 / n2
    v2_2 = p2_2 / n2

    # Dot and Cross products
    dotp = v1_1*v2_1 + v1_2*v2_2
    cruz = v1_1*v2_2 - v1_2*v2_1

    # Signed angles
    angulos = torch.atan2(cruz, dotp)   # (m, K-1)

    # J-measure per column
    e = torch.exp(1j * angulos)
    J = 1.0 - torch.abs(torch.mean(e, dim=1))  # (m,)

    return J
