"""
D_measure.py

Implementaciones de la medida D (alternativa al índice de fases de Fourier) para medir
regularidad / irregularidad / determinismo en series de tiempo de estereoEEG. Para el
artícuilo xxxxxxxxx publiocado en la revisrta Brain (en proceso de revisión editorial)

Incluye:
- toroD             : versión original con ciclos en CPU (referencia).
- toroD_joblib_batch: versión por lotes usando joblib (CPU)
- toroD_torch_batch : versión por lotes en PyTorch (CPU / CUDA / MPS).
"""

import numpy as np

# ---------------------------------------------------------------------
# torvecdav: versión simple de la medida J basada en fases absolutas
# ---------------------------------------------------------------------
def toroD(x1, x2, fi, ff):
    """
    Implementación alternativa de la medida J.

    x1, x2 : arrays 1D reales.
    fi, ff : índices de frecuencia (fi incluido, ff excluido).

    Devuelve
    --------
    j_0      : valor escalar de J en [0, 1].
    caminata : array 1D con los ángulos entre pasos consecutivos.
    """
    # FFT y fases
    F1 = np.fft.rfft(x1)
    F2 = np.fft.rfft(x2)
    ff1 = np.angle(F1)[fi:ff]
    ff2 = np.angle(F2)[fi:ff]

    if ff1.size < 2:
        return np.nan, np.array([])

    # Puntos consecutivos (φ1, φ2)
    p1 = np.column_stack((ff1[:-1], ff2[:-1]))   # (N-1, 2)
    p2 = np.column_stack((ff1[1:],  ff2[1:]))    # (N-1, 2)

    # Vectores normalizados
    eps = np.finfo(float).eps
    n1 = np.maximum(np.hypot(p1[:, 0], p1[:, 1]), eps)
    n2 = np.maximum(np.hypot(p2[:, 0], p2[:, 1]), eps)

    v1 = p1 / n1[:, None]
    v2 = p2 / n2[:, None]

    # Producto punto y cruz en 2D
    dotp = np.sum(v1 * v2, axis=1)
    cruz = v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]

    # Ángulo firmado entre v1 y v2
    angulos = np.arctan2(cruz, dotp)

    # Medida J (media circular compleja)
    j_0 = 1.0 - np.abs(np.mean(np.exp(1j * angulos)))
    caminata = angulos

    return j_0

# ---------------------------------------------------------------------
# torvecdav_joblib_batch: versión por lotes usando joblib (CPU)
# ---------------------------------------------------------------------
def toroD_joblib_batch(a, b, fi, ff):
    """
    Evalúa torvecdav en paralelo para todas las columnas usando joblib.

    a, b : arrays (n, m) con series en columnas.
    fi, ff : índices de frecuencia (fi incluido, ff excluido).

    Devuelve
    --------
    J : array 1D de longitud m con los valores de J por columna.
    """
    from joblib import Parallel, delayed

    m = a.shape[1]

    J_list = Parallel(n_jobs=-1)(
        delayed(toroD)(a[:, i], b[:, i], fi, ff)
        for i in range(m)
    )
    return np.asarray(J_list)

    # ---------------------------------------------------------------------
# torvecdav_torch_batch: versión por lotes usando PyTorch
# ---------------------------------------------------------------------
def toroD_torch_batch(A, B, fi, ff, device=None):
    """
    Implementación por lotes de torvecdav usando PyTorch.

    A, B : arrays NumPy de forma (n, m) con series en columnas.
    fi, ff : índices de frecuencia (fi incluido, ff excluido).
    device : torch.device o None (auto: CUDA → MPS → CPU).

    Devuelve
    --------
    J : tensor 1D de forma (m,) con el valor de J por pareja de columnas.
    """
    import torch

    # Selección automática de dispositivo
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Datos como float32 en forma (m, n) en el device
    A_t = torch.from_numpy(A.T.astype(np.float32)).to(device)
    B_t = torch.from_numpy(B.T.astype(np.float32)).to(device)  # (m, n)

    # FFT real por filas
    F1 = torch.fft.rfft(A_t, dim=1)
    F2 = torch.fft.rfft(B_t, dim=1)

    ff1 = torch.angle(F1[:, fi:ff])   # (m, K)
    ff2 = torch.angle(F2[:, fi:ff])

    if ff1.size(1) < 2:
        return torch.full((A_t.size(0),), float("nan"), device=device)

    # Puntos consecutivos
    p1_1 = ff1[:, :-1]   # φ1(k)
    p1_2 = ff2[:, :-1]   # φ2(k)
    p2_1 = ff1[:, 1:]    # φ1(k+1)
    p2_2 = ff2[:, 1:]    # φ2(k+1)

    # Normas y vectores normalizados
    eps = torch.finfo(ff1.dtype).eps

    n1 = torch.sqrt(p1_1**2 + p1_2**2)
    n2 = torch.sqrt(p2_1**2 + p2_2**2)
    n1 = torch.clamp(n1, min=eps)
    n2 = torch.clamp(n2, min=eps)

    v1_1 = p1_1 / n1
    v1_2 = p1_2 / n1
    v2_1 = p2_1 / n2
    v2_2 = p2_2 / n2

    # Producto punto y cruz
    dotp = v1_1*v2_1 + v1_2*v2_2
    cruz = v1_1*v2_2 - v1_2*v2_1

    # Ángulos firmados
    angulos = torch.atan2(cruz, dotp)   # (m, K-1)

    # Medida J por columna
    e = torch.exp(1j * angulos)
    J = 1.0 - torch.abs(torch.mean(e, dim=1))  # (m,)

    return J

