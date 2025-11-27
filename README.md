# J-optimizado: Implementaciones rápidas de la medida J

**J-optimizado** es un proyecto de benchmarking y optimización de la **medida J**, un índice basado exclusivamente en las **fases de Fourier** que permite detectar **determinismo**, **irregularidad**, **no linealidad** y estructuras dinámicas en series de tiempo sin necesidad de reconstrucción del espacio de fases.

La medida J se calcula a partir de la **caminata de fases** en el **toro 2D** (fase del primer canal vs fase del segundo canal) y de la **diferencia de ángulo** entre pasos consecutivos.

Este repositorio incluye diferentes versiones del mismo algoritmo, desde la implementación original basada en ciclos hasta versiones totalmente vectorizadas, aceleradas con GPU y optimizadas para batch processing.

---

## 📘 Artículo original de la medida J

La implementación sigue la metodología descrita en:

> Aguilar-Hernández AI, Serrano-Solís DM, Ríos-Herrera WA, Zapata-Berruecos JF, Vilaclara G, Martínez-Mekler G, Müller MF.  
> **Fourier phase index for extracting signatures of determinism and nonlinear features in time series**.  
> *Chaos*. 2024;34(1):013103. DOI: 10.1063/5.0160555.

Este trabajo demuestra que la medida J:

- Detecta regularidad/determinismo en datos con ruido.  
- Es sensible a estructuras no lineales.  
- Funciona en señales reales como EEG intracraneal.  
- No requiere técnicas como espacio de fases o surrogados.

---

## 📁 Contenido del repositorio

### `J_measure.py`
Contiene todas las versiones de la medida J:

| Versión | Tecnologías | Descripción |
|--------|--------------|-------------|
| `toro` | NumPy + ciclos | Implementación original, clara pero lenta. |
| `toro2` | NumPy vectorizado | Reemplaza ciclos por matrices; usa búsqueda explícita de 9 cuadrantes. |
| `toro2_1` | NumPy | Usa envoltura modular en lugar de 9 cuadrantes; más simple y rápida. |
| `toro2_2` | NumPy + complejos | Máxima velocidad en CPU; usa números complejos para reducir memoria y operaciones. |
| `toro2_2_torch_batch` | PyTorch (CPU/CUDA/MPS) | Versión por lotes; permite procesar cientos/miles de pares de señales en paralelo en CPU, GPU NVIDIA o GPU Apple Silicon (M1/M2). |

---

## 🚀 ¿Qué mejoras trae cada versión?

### ✔ `toro` — Versión base  
- Implementación directa.  
- Usa ciclos Python.  
- Sirve como referencia y validación, pero es lenta.

### ✔ `toro2` — Vectorización NumPy  
- Elimina ciclos.  
- Representa los 9 posibles desplazamientos del toro usando un arreglo 3D.  
- Mucho más rápida que `toro`.

### ✔ `toro2_1` — Envoltura modular  
- Observa que los 9 cuadrantes equivalen a envolver `p2 - 2*p1`.  
- Simplifica el código y reduce memoria.  
- Rendimiento mayor.

### ✔ `toro2_2` — Optimización con complejos  
- Representa cada vector como `x + i y`.  
- Usa la multiplicación compleja para obtener producto punto y cruz.  
- Mínima memoria y máxima velocidad en CPU.  
- Versión recomendada si usas sólo NumPy.

### ✔ `toro2_2_torch_batch` — PyTorch (CPU + GPU + MPS)  
- Procesa **m** pares de señales simultáneamente.  
- Compatible con:
  - CPU
  - GPU NVIDIA (`cuda`)
  - GPU Apple Silicon (`mps`)  
- Útil para grandes lotes o integración con frameworks de aprendizaje automático.

---

## 🧪 `J-paralelizada.ipynb`

Este notebook incluye:

- Comparación de tiempos entre todas las versiones.
- Paralelización CPU con `joblib`.
- Aceleración multicapa con PyTorch (CPU/CUDA/MPS).
- Gráficas y tablas de rendimiento.

---

## 🔧 Instalación

### Dependencias principales

```bash
pip install torch torchvision torchaudio
