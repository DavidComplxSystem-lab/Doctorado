# 🧩 J-optimizado y D-optimizado
Implementaciones rápidas de medidas basadas en fases de Fourier para detectar determinismo y no linealidad en series de tiempo

**J-optimizado** es un proyecto de benchmarking y optimización de la **medida J**, un índice basado exclusivamente en las **fases de Fourier** que permite detectar **determinismo**, **irregularidad**, **no linealidad** y estructuras dinámicas en series de tiempo sin necesidad de reconstrucción del espacio de fases.

La medida J se calcula a partir de la **caminata de fases** en el **toro 2D** (fase del primer canal vs fase del segundo canal) y de la **diferencia de ángulo** entre pasos consecutivos.

Este repositorio también incluye la **medida D**, una alternativa más simple y eficiente basada en fases, actualmente **en revisión por la revista *Brain***.

---

## 📘 Artículo original de la medida J

La implementación sigue la metodología descrita en:

> Aguilar-Hernández AI, Serrano-Solís DM, Ríos-Herrera WA, Zapata-Berruecos JF, Vilaclara G, Martínez-Mekler G, Müller MF.  
> **Fourier phase index for extracting signatures of determinism and nonlinear features in time series**.  
> *Chaos*. 2024;34(1):013103. DOI: 10.1063/5.0160555.

Este trabajo demuestra que J:

- Detecta regularidad/determinismo incluso con ruido.  
- Es sensible a estructuras no lineales.  
- Funciona en señales reales como EEG intracraneal.  
- No requiere espacio de fases o surrogados.

---

## 📁 Contenido del repositorio

### `J_measure.py`
Incluye todas las versiones optimizadas de la medida J:

| Versión | Tecnologías | Descripción |
|--------|-------------|-------------|
| `toro` | NumPy + ciclos | Implementación original y más lenta. |
| `toro2` | NumPy vectorizado | Sustituye los ciclos por matrices. |
| `toro2_1` | NumPy | Usa envoltura modular; más simple y rápida. |
| `toro2_2` | NumPy + complejos | Versión más eficiente en CPU. |
| `toro2_2_joblib_batch` | joblib | Computo paralelo por columnas (CPU). |
| `toro2_2_torch_batch` | PyTorch | Procesamiento por lotes en CPU, CUDA o MPS. |

---

### `D_measure.py`
Implementación optimizada de la medida J (artículo en revisión *Brain*):

| Versión | Tecnologías | Descripción |
|--------|-------------|-------------|
| `toroD` | NumPy | Versión base, extremadamente ligera. |
| `toroD_joblib_batch` | joblib | Paralelización CPU. |
| `toroD_torch_batch` | PyTorch | Aceleración masiva por lotes en CPU/CUDA/MPS. |

---

### `benchmark_J_D.ipynb`

Notebook que incluye:

- Comparación de rendimiento entre todas las versiones  
- Paralelización con joblib  
- Aceleración con PyTorch en CPU, GPU NVIDIA o GPU Apple Silicon  
- Resultados reproducibles con 100 × 100 pruebas  

---

## 🚀 Rendimiento aproximado

Benchmarks típicos usando señales de tamaño `1000 × 100`, repetidos 100 veces:
-1toro2 ~50× más rápido que toro
-toro2_1 ~82× más rápido que toro
-toro2_2 ~100× más rápido que toro
-toro2_2_joblib_batch ~38× más rápido que toro
-toro2_2_torch_batch ~440× más rápido que toro
-toroD ~74× más rápido que toro
-toroD_joblib_batch ~40× más rápido
-toroD_torch_batch ~675× más rápido


### Hardware utilizado

- **GPU NVIDIA RTX 5070 Ti (CUDA)**  
  → Máximo rendimiento observado (hasta 675×).

- **MacBook Pro M1 Pro (GPU 16-core, backend MPS)**  
  → Rendimiento similar (3–10% menor que CUDA).

- **CPU multinúcleo**  
  → Aceleraciones moderadas (~20–40×).

PyTorch **selecciona automáticamente** el mejor backend disponible:
- `cuda` → GPU NVIDIA  
- `mps` → Apple Silicon (M1/M2)  
- `cpu` → cualquier PC sin GPU compatible  

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
