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

🧩 J-optimizado y D-optimizado
Implementaciones rápidas de medidas basadas en fases de Fourier para detectar determinismo y no linealidad en series de tiempo

Repositorio por David Michel Serrano Solís

📌 Resumen

Este proyecto reúne implementaciones optimizadas de dos medidas basadas en fases de Fourier:

Medida J — publicada en Chaos (AIP), diseñada para detectar determinismo, no linealidad e irregularidad dinámica sin necesidad de reconstrucción de espacio de fases.

Medida D — una alternativa computacionalmente más ligera desarrollada recientemente y actualmente en revisión por la revista Brain.

Incluye versiones optimizadas para:

CPU (NumPy)

CPU multinúcleo (joblib)

GPU NVIDIA (CUDA)

GPU Apple Silicon (M1/M2, MPS backend)

Procesamiento por lotes (PyTorch batch mode)

Además, se incluye un notebook de benchmark para comparar el rendimiento de todas las versiones.

📘 Referencias científicas
Medida J (publicada)

Aguilar-Hernández AI, Serrano-Solís DM, Ríos-Herrera WA, Zapata-Berruecos JF, Vilaclara G, Martínez-Mekler G, Müller MF.
Fourier phase index for extracting signatures of determinism and nonlinear features in time series.
Chaos. 2024;34(1):013103. DOI: 10.1063/5.0160555.

La medida J captura determinismo, regularidad y estructuras no lineales incluso en presencia de ruido, y no requiere reconstrucción de espacio de fases ni surrogados.

Medida D (enviado a Brain)

Implementación alternativa basada en fases absolutas de Fourier.
Manuscrito actualmente sometido a revisión en la revista Brain; la versión incluida aquí es únicamente para demostración y benchmarking.

📁 Contenido del repositorio
J_measure.py

Implementaciones de la medida J:

Versión	Tecnologías	Descripción
toro	NumPy + ciclos	Implementación original (lenta; referencia).
toro2	NumPy vectorizado	Eliminación de ciclos; cálculo explícito de 9 cuadrantes.
toro2_1	NumPy	Envoltura modular para reemplazar cuadrantes.
toro2_2	NumPy + complejos	Versión más rápida en CPU (mínima memoria).
toro2_2_joblib_batch	joblib	Paralelización por columnas en CPU multinúcleo.
toro2_2_torch_batch	PyTorch	Procesamiento por lotes en CPU, GPU NVIDIA (CUDA) o GPU Apple Silicon (MPS).
D_measure.py

Implementaciones de la medida alternativa D:

Versión	Tecnologías	Descripción
toroD	NumPy	Versión escalar rápida y simple.
toroD_joblib_batch	joblib	Lote CPU para múltiples columnas.
toroD_torch_batch	PyTorch	Versión por lotes acelerada (CPU/CUDA/MPS).
benchmark_J_D.ipynb

Notebook que:

Ejecuta 100 repeticiones de 100 análisis por función.

Compara tiempos entre todas las versiones.

Evalúa aceleración usando CPU, joblib, CUDA y MPS.

Determina la mejor versión para cada arquitectura.

🚀 Rendimiento (resultados típicos)

Basado en el análisis de 100×100 ejecuciones con señales de longitud 1000:

toro2                    ~50× más rápido que toro
toro2_1                  ~82× más rápido
toro2_2                 ~100× más rápido
toro2_2_joblib_batch     ~38× más rápido
toro2_2_torch_batch     ~440× más rápido

toroD                    ~74× más rápido
toroD_joblib_batch       ~40× más rápido
toroD_torch_batch       ~675× más rápido


Estos resultados son aproximados y pueden variar dependiendo del hardware.

⚙️ Hardware usado para benchmarking

GPU NVIDIA RTX 5070 Ti (CUDA)
→ Máxima aceleración, ~440–675× según la versión.

MacBook Pro M1 Pro (16-core GPU, MPS backend)
→ Rendimiento muy similar a CUDA (diferencias ~3–10%).
→ Esta paridad se debe a la eficiencia del backend MPS de PyTorch.

CPU multinúcleo (Windows/Linux/macOS)
→ joblib produce aceleraciones de ~20–40×.

Para otras arquitecturas, el rendimiento puede variar.

🔧 Instalación
Dependencias principales
pip install numpy joblib

PyTorch
🔹 Para Nvidia CUDA (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

🔹 Para Mac M1/M2 (Metal / MPS)
pip install torch torchvision torchaudio

Detección automática del dispositivo

Las funciones *_torch_batch seleccionan automáticamente:

GPU NVIDIA → cuda

GPU Apple Silicon → mps

CPU → cpu

sin necesidad de configuración adicional.

🧠 ¿Por qué usar PyTorch?

Permite procesar miles de pares de señales en paralelo.

Maneja tensores complejos de forma nativa.

Selecciona automáticamente la mejor aceleración según el hardware.

Ideal para pipelines de investigación, machine learning o procesamiento masivo.

✉️ Contacto

David Michel Serrano Solís
Física — Ciencias Biomédicas — Análisis de series de tiempo fisiológicas
(Incluye aquí tu correo o tu LinkedIn)
