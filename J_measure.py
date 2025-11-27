"""
J_measure.py

Implementaciones de la medida J (índice de fases de Fourier) para medir
regularidad / irregularidad / determinismo en series de tiempo.

Incluye:
- toro          : versión original con ciclos en CPU (referencia).
- toro2         : versión NumPy vectorizada con búsqueda explícita de 9 cuadrantes.
- toro2_1       : versión NumPy vectorizada con desplazamiento envuelto.
- toro2_2       : versión NumPy optimizada usando números complejos.
- toro2_2_torch_batch : versión por lotes en PyTorch (CPU / CUDA / MPS).
"""

import numpy as np


# ---------------------------------------------------------------------
# Función auxiliar
# ---------------------------------------------------------------------
def distancia(p1, p2):
    """Distancia euclidiana entre dos puntos 2D."""
    import math
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


# ---------------------------------------------------------------------
# Implementación original (ciclos en CPU)
# ---------------------------------------------------------------------
def toro(x1, x2, fi, ff):
    """
    Implementación base de la medida J usando ciclos explícitos.

    Parámetros
    ----------
    x1, x2 : array_like
        Series reales 1D.
    fi, ff : int
        Índices de frecuencia (fi incluido, ff excluido).

    Retorna
    -------
    j_0 : float
        Valor de la medida J en [0, 1].
    """
    # FFT real y fases
    f1 = np.fft.rfft(x1)
    f2 = np.fft.rfft(x2)
    ff1 = np.angle(f1)[fi:ff]
    ff2 = np.angle(f2)[fi:ff]

    vectores = []

    # Caminata de fases en el toro usando 9 candidatos por paso
    for i in range(len(ff1) - 1):
        p1 = [ff1[i], ff2[i]]
        p2 = [ff1[i + 1], ff2[i + 1]]

        cuadrante = [
            [p2[0] - p1[0],              p2[1] - p1[1]],
            [p2[0] - p1[0],              p2[1] + 2*np.pi - p1[1]],
            [p2[0] + 2*np.pi - p1[0],    p2[1] + 2*np.pi - p1[1]],
            [p2[0] + 2*np.pi - p1[0],    p2[1] - p1[1]],
            [p2[0] + 2*np.pi - p1[0],    p2[1] - 2*np.pi - p1[1]],
            [p2[0] - p1[0],              p2[1] - 2*np.pi - p1[1]],
            [p2[0] - 2*np.pi - p1[0],    p2[1] - 2*np.pi - p1[1]],
            [p2[0] - 2*np.pi - p1[0],    p2[1] - p1[1]],
            [p2[0] - 2*np.pi - p1[0],    p2[1] + 2*np.pi - p1[1]],
        ]

        # Seleccionar el desplazamiento más corto
        distancia1 = [distancia(p1, q) for q in cuadrante]
        dis_min1 = min(distancia1)
        for j in range(len(distancia1)):
            if distancia1[j] == dis_min1:
                p2 = cuadrante[j]
                break

        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        vectores.append(v1)

    # Ángulos entre vectores consecutivos
    angulos = []
    for i in range(len(vectores) - 1):
        v1 = np.array(vectores[i])
        v2 = np.array(vectores[i + 1])
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        angulo = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
        cruz = v1[0]*v2[1] - v1[1]*v2[0]

        if cruz > 0:
            angulo = np.pi - angulo
        if cruz == 0 and angulo < 0:
            angulo = np.pi
        if cruz < 0:
            angulo = angulo + np.pi

        angulos.append(angulo)

    # Media circular compleja
    e = [np.exp(1j * a) for a in angulos]
    e1 = np.sum(e) / len(angulos)
    j_0 = 1.0 - np.abs(e1)
    return j_0


# ---------------------------------------------------------------------
# toro2: versión NumPy vectorizada con 9 cuadrantes explícitos
# ---------------------------------------------------------------------
def toro2(x1, x2, fi, ff):
    """
    Versión vectorizada de toro usando búsqueda explícita de 9 cuadrantes.
    Reproduce la lógica original con operaciones NumPy.
    """
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(x2))

    ff1 = ff1[fi:ff]
    ff2 = ff2[fi:ff]

    vectores = np.zeros((2, len(ff1) - 1))

    p1 = np.array([ff1[:-1], ff2[:-1]])
    p2 = np.array([ff1[1:],  ff2[1:]])

    cuadrante = np.array([
        p2 - p1,
        np.array([p2[0],           2*np.pi + p2[1]]) - p1,
        np.array([p2[0] + 2*np.pi, p2[1] + 2*np.pi]) - p1,
        np.array([p2[0] + 2*np.pi, p2[1]])           - p1,
        np.array([p2[0] + 2*np.pi, p2[1] - 2*np.pi]) - p1,
        np.array([p2[0],           p2[1] - 2*np.pi]) - p1,
        np.array([p2[0] - 2*np.pi, p2[1] - 2*np.pi]) - p1,
        np.array([p2[0] - 2*np.pi, p2[1]])           - p1,
        np.array([p2[0] - 2*np.pi, p2[1] + 2*np.pi]) - p1,
    ])

    # Ajuste para medir distancia a 2*p1 (como en implementación original)
    p1_rep = np.array([p1]*9)
    cuadrante = cuadrante - p1_rep

    distancia1 = np.linalg.norm(cuadrante, axis=1)
    a = np.argmin(distancia1, axis=0)

    b = np.ones(len(ff1) - 1, dtype=np.int32)
    c = np.linspace(0, len(ff1) - 2, num=len(ff1) - 1, dtype=np.int32)

    vectores[0, :] = cuadrante[a, 0*b, c]
    vectores[1, :] = cuadrante[a, b,   c]

    v1 = vectores[:, :-1]
    v2 = vectores[:, 1:]

    v1_norm = v1 / np.linalg.norm(v1, axis=0)
    v2_norm = v2 / np.linalg.norm(v2, axis=0)

    angulo = np.arccos(np.clip(
        v1_norm[0, :]*v2_norm[0, :] + v1_norm[1, :]*v2_norm[1, :],
        -1.0, 1.0
    ))
    cruz = v1[0, :]*v2[1, :] - v1[1, :]*v2[0, :]
    cruz = np.sign(cruz)
    angulo = angulo * cruz

    e = np.exp(1j * angulo)
    e1 = np.sum(e) / len(angulo)
    j = 1 - np.abs(e1)
    return j


# ---------------------------------------------------------------------
# toro2_1: versión vectorizada con desplazamiento envuelto
# ---------------------------------------------------------------------
def toro2_1(x1, x2, fi, ff):
    """
    Versión vectorizada que envuelve el desplazamiento (p2 - 2*p1)
    a (-π, π] en cada componente, reemplazando la búsqueda de 9 cuadrantes.
    """
    ff1 = np.angle(np.fft.rfft(x1))
    ff2 = np.angle(np.fft.rfft(x2))

    ff1 = ff1[fi:ff]
    ff2 = ff2[fi:ff]

    if ff1.size < 3:
        return np.nan

    p1 = np.array([ff1[:-1], ff2[:-1]])
    p2 = np.array([ff1[1:],  ff2[1:]])

    # Desplazamiento p2 - 2*p1 envuelto a (-π, π]
    z_raw = p2 - 2.0 * p1
    z_wrapped = (z_raw + np.pi) % (2*np.pi) - np.pi

    vectores = z_wrapped

    v1 = vectores[:, :-1]
    v2 = vectores[:, 1:]

    norm_v1 = np.linalg.norm(v1, axis=0)
    norm_v2 = np.linalg.norm(v2, axis=0)

    eps = np.finfo(float).eps
    norm_v1[norm_v1 == 0] = eps
    norm_v2[norm_v2 == 0] = eps

    v1_norm = v1 / norm_v1
    v2_norm = v2 / norm_v2

    dotp = v1_norm[0, :]*v2_norm[0, :] + v1_norm[1, :]*v2_norm[1, :]
    dotp = np.clip(dotp, -1.0, 1.0)
    angulo = np.arccos(dotp)

    cruz = v1[0, :]*v2[1, :] - v1[1, :]*v2[0, :]
    signo = np.sign(cruz)
    angulo_firmado = angulo * signo

    e = np.exp(1j * angulo_firmado)
    e1 = np.mean(e)
    j = 1.0 - np.abs(e1)
    return j


# ---------------------------------------------------------------------
# toro2_2: versión NumPy optimizada con números complejos
# ---------------------------------------------------------------------
def toro2_2(x1, x2, fi, ff):
    """
    Versión optimizada de toro2_1 usando números complejos.
    Reduce memoria y número de operaciones manteniendo el mismo resultado.
    """
    ff1 = np.angle(np.fft.rfft(x1))[fi:ff]
    ff2 = np.angle(np.fft.rfft(x2))[fi:ff]

    if ff1.size < 3:
        return np.nan

    # Fases consecutivas
    p1_1 = ff1[:-1]
    p1_2 = ff2[:-1]
    p2_1 = ff1[1:]
    p2_2 = ff2[1:]

    # Desplazamiento p2 - 2*p1, envuelto a (-π, π]
    z1_raw = p2_1 - 2.0 * p1_1
    z2_raw = p2_2 - 2.0 * p1_2

    z1 = (z1_raw + np.pi) % (2*np.pi) - np.pi
    z2 = (z2_raw + np.pi) % (2*np.pi) - np.pi

    # Representación compleja de los pasos en el toro
    v = z1 + 1j*z2

    mag = np.abs(v)
    eps = np.finfo(float).eps
    mag[mag == 0] = eps
    u = v / mag

    u1 = u[:-1]
    u2 = u[1:]

    # Producto complejo: parte real = dot, imaginaria = cruz
    w = u1 * np.conj(u2)
    dotp = w.real
    cruz = w.imag

    angulos = np.arctan2(cruz, dotp)

    j = 1.0 - np.abs(np.mean(np.exp(1j * angulos)))
    return j

# ---------------------------------------------------------------------
# toro2_2_joblib_batch: versión por lotes usando joblib (CPU paralela)
# ---------------------------------------------------------------------
def toro2_2_joblib_batch(a, b, fi, ff):
    """
    Evalúa toro2_2 en paralelo para todas las columnas usando joblib.

    a, b : arrays NumPy de forma (n, m) con series en columnas.
    fi, ff : índices de frecuencia (fi incluido, ff excluido).

    Retorna
    -------
    lista con m valores de J, uno por cada columna.
    """
    from joblib import Parallel, delayed  # import local: sólo si se llama a la función

    m = a.shape[1]

    return Parallel(n_jobs=-1)(
        delayed(toro2_2)(a[:, i], b[:, i], fi, ff)
        for i in range(m)
    )
# ---------------------------------------------------------------------
# toro2_2_torch_batch: versión por lotes con PyTorch
# ---------------------------------------------------------------------
def toro2_2_torch_batch(A, B, fi, ff, device=None):
    """
    Implementación por lotes de toro2_2 usando PyTorch.

    A, B : arrays NumPy de forma (n, m), series en columnas.
    fi, ff : índices de frecuencia (fi incluido, ff excluido).
    device : torch.device o None (auto: CUDA → MPS → CPU).

    Devuelve
    --------
    J : tensor de forma (m,)
        Valor de J para cada pareja de columnas (A[:,j], B[:,j]).
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

    # Datos a float32, forma (m, n) en el dispositivo
    A_t = torch.from_numpy(A.T.astype(np.float32)).to(device)
    B_t = torch.from_numpy(B.T.astype(np.float32)).to(device)

    # FFT real por filas (dim=1)
    F1 = torch.fft.rfft(A_t, dim=1)
    F2 = torch.fft.rfft(B_t, dim=1)

    ff1 = torch.angle(F1[:, fi:ff])
    ff2 = torch.angle(F2[:, fi:ff])

    if ff1.size(1) < 3:
        return torch.full((A_t.size(0),), float("nan"), device=device)

    # Fases consecutivas
    p1_1 = ff1[:, :-1]
    p1_2 = ff2[:, :-1]
    p2_1 = ff1[:, 1:]
    p2_2 = ff2[:, 1:]

    # Desplazamiento envuelto
    pi = torch.pi
    two_pi = 2*pi

    z1_raw = p2_1 - 2.0 * p1_1
    z2_raw = p2_2 - 2.0 * p1_2

    z1 = (z1_raw + pi) % two_pi - pi
    z2 = (z2_raw + pi) % two_pi - pi

    v = z1 + 1j*z2

    mag = torch.abs(v)
    mag = torch.clamp(mag, min=torch.finfo(mag.dtype).eps)
    u = v / mag

    u1 = u[:, :-1]
    u2 = u[:, 1:]

    w = u1 * torch.conj(u2)
    dotp = w.real
    cruz = w.imag

    angulos = torch.atan2(cruz, dotp)

    e = torch.exp(1j * angulos)
    J = 1.0 - torch.abs(torch.mean(e, dim=1))
    return J
