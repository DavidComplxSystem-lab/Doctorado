# 🧩 J-optimizado y D-optimizado
**Implementaciones de alto rendimiento para detectar determinismo y no linealidad en series de tiempo basadas en fases de Fourier.**

**J-optimizado** es un proyecto de *benchmarking* y optimización de la **medida J**, un índice basado exclusivamente en las **fases de Fourier**. Esta herramienta permite detectar **determinismo**, **irregularidad**, **no linealidad** y estructuras dinámicas en series de tiempo sin necesidad de reconstruir el espacio de fases.

La medida J se calcula a partir de la **caminata de fases** en el **toro 2D** (fase del primer canal vs. fase del segundo canal) y de la **diferencia de ángulo** entre pasos consecutivos.

Este repositorio también incluye la implementación optimizada de la **medida D**, una alternativa más simple y eficiente basada en fases, actualmente **en revisión por la revista *Brain***.

---

## 📘 Artículo original de la medida J

La implementación sigue la metodología descrita en:

> **Aguilar-Hernández AI, Serrano-Solís DM, Ríos-Herrera WA, Zapata-Berruecos JF, Vilaclara G, Martínez-Mekler G, Müller MF.**
> *"Fourier phase index for extracting signatures of determinism and nonlinear features in time series."*
> *Chaos*. 2024;34(1):013103. DOI: [10.1063/5.0160555](https://doi.org/10.1063/5.0160555).

Este trabajo demuestra que la medida J:
- Detecta regularidad y determinismo incluso en presencia de ruido.
- Es sensible a estructuras no lineales.
- Es efectiva en señales reales complejas (ej. EEG intracraneal).
- **No requiere** reconstrucción del espacio de fases ni generación de surrogados.

---

## 📁 Contenido del repositorio

### 1. `J_measure.py`
Incluye todas las versiones optimizadas de la medida J:

| Versión | Tecnologías | Descripción |
| :--- | :--- | :--- |
| `toro` | NumPy + ciclos | Implementación original (referencia base). |
| `toro2` | NumPy vectorizado | Sustituye los ciclos por operaciones matriciales. |
| `toro2_1` | NumPy | Usa envoltura modular; más simple y rápida. |
| `toro2_2` | NumPy + complejos | Versión más eficiente en CPU puro. |
| `toro2_2_joblib_batch` | Joblib | Cómputo paralelo por columnas (CPU). |
| `toro2_2_torch_batch` | PyTorch | Procesamiento por lotes en CPU, CUDA o MPS. |

### 2. `D_measure.py`
Implementación optimizada de la medida D (artículo en revisión *Brain*):

| Versión | Tecnologías | Descripción |
| :--- | :--- | :--- |
| `toroD` | NumPy | Versión base, extremadamente ligera. |
| `toroD_joblib_batch` | Joblib | Paralelización en CPU. |
| `toroD_torch_batch` | PyTorch | Aceleración masiva por lotes en CPU/CUDA/MPS. |

### 3. `benchmark_J_D.ipynb`
Notebook interactivo que incluye:
- Comparación de rendimiento entre todas las versiones.
- Pruebas de paralelización con Joblib.
- Aceleración con PyTorch en CPU, GPU NVIDIA y Apple Silicon (MPS).
- Resultados reproducibles (100 repeticiones × 100 señales).

---

## 🚀 Benchmarks de Rendimiento

Los siguientes resultados se obtuvieron utilizando señales de tamaño `1000 × 100`, promediando 100 repeticiones.

| Versión | Factor de Aceleración (Speedup) |
| :--- | :--- |
| `toro2` | ~50× más rápido que `toro` |
| `toro2_1` | ~82× más rápido que `toro` |
| `toro2_2` | ~100× más rápido que `toro` |
| `toro2_2_joblib_batch`| ~38× más rápido que `toro` |
| **`toro2_2_torch_batch`** | **~440× más rápido que `toro`** |
| `toroD` | ~74× más rápido que `toro` |
| `toroD_joblib_batch` | ~40× más rápido que `toro` |
| **`toroD_torch_batch`** | **~675× más rápido que `toro`** |

### Hardware de referencia

* **GPU NVIDIA RTX 4060 Ti (CUDA):**
    * Máximo rendimiento observado (hasta 675×).
* **MacBook Pro M1 Pro (GPU 16-core, backend MPS):**
    * Rendimiento similar a la 4060 Ti (aprox. 3–10% menor).
* **CPU Multinúcleo:**
    * Aceleraciones moderadas (~20–40×).

> **Nota:** PyTorch selecciona automáticamente el mejor backend disponible (`cuda` para NVIDIA, `mps` para Apple Silicon, o `cpu`).

---

## 🛠 Instalación

### Dependencias generales
```bash
pip install numpy joblib
```

**PyTorch**
**GPU NVIDIA (CUDA)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Mac M1/M2 (Metal/MPS)**
```bash
pip install torch torchvision torchaudio
```

✉️ Contacto

David Michel Serrano Solís
davidser88@hotmail.com
