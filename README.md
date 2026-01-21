# 🧩 J-optimized & D-optimized
**High-performance implementations for detecting determinism and nonlinearity in time series based on Fourier phases.**

**J-optimized** is a benchmarking and optimization project for the **J-measure**, an index based exclusively on **Fourier phases**. This tool allows for the detection of **determinism**, **irregularity**, **nonlinearity**, and dynamic structures in time series without the need for phase space reconstruction.

The J-measure is calculated from the **phase walk** on the **2D torus** (phase of the first channel vs. phase of the second channel) and the **angle difference** between consecutive steps.

This repository also includes the optimized implementation of the **D-measure**, a simpler and more efficient alternative based on phases, currently **under review by *Brain* journal**.

---

## 📘 Original J-measure Paper

The implementation follows the methodology described in:

> **Aguilar-Hernández AI, Serrano-Solís DM, Ríos-Herrera WA, Zapata-Berruecos JF, Vilaclara G, Martínez-Mekler G, Müller MF.**
> *"Fourier phase index for extracting signatures of determinism and nonlinear features in time series."*
> *Chaos*. 2024;34(1):013103. DOI: [10.1063/5.0160555](https://doi.org/10.1063/5.0160555).

This work demonstrates that the J-measure:
- Detects regularity and determinism even in the presence of noise.
- Is sensitive to nonlinear structures.
- Is effective on complex real signals (e.g., intracranial EEG).
- **Does not require** phase space reconstruction or surrogate generation.

---

## 📁 Repository Contents

### 1. `J_measure.py`
Includes all optimized versions of the J-measure:

| Version | Technologies | Description |
| :--- | :--- | :--- |
| `toro` | NumPy + loops | Original implementation (baseline reference). |
| `toro2` | Vectorized NumPy | Replaces loops with matrix operations. |
| `toro2_1` | NumPy | Uses modular wrapping; simpler and faster. |
| `toro2_2` | NumPy + Complex | Most efficient pure CPU version. |
| `toro2_2_joblib_batch` | Joblib | Column-wise parallel computing (CPU). |
| `toro2_2_torch_batch` | PyTorch | Batch processing on CPU, CUDA, or MPS. |

### 2. `D_measure.py`
Optimized implementation of the D-measure (*Brain* article under review):

| Version | Technologies | Description |
| :--- | :--- | :--- |
| `toroD` | NumPy | Base version, extremely lightweight. |
| `toroD_joblib_batch` | Joblib | CPU parallelization. |
| `toroD_torch_batch` | PyTorch | Massive batch acceleration on CPU/CUDA/MPS. |

### 3. `benchmark_J_D.ipynb`
Interactive notebook including:
- Performance comparison across all versions.
- Parallelization tests with Joblib.
- Acceleration with PyTorch on CPU, NVIDIA GPUs, and Apple Silicon (MPS).
- Reproducible results (100 repetitions × 100 signals).

---

## 🚀 Performance Benchmarks

The following results were obtained using `1000 × 100` signals, averaging 100 repetitions.

| Version | Speedup Factor |
| :--- | :--- |
| `toro2` | ~50× faster than `toro` |
| `toro2_1` | ~82× faster than `toro` |
| `toro2_2` | ~100× faster than `toro` |
| `toro2_2_joblib_batch`| ~38× faster than `toro` |
| **`toro2_2_torch_batch`** | **~440× faster than `toro`** |
| `toroD` | ~74× faster than `toro` |
| `toroD_joblib_batch` | ~40× faster than `toro` |
| **`toroD_torch_batch`** | **~675× faster than `toro`** |

### Reference Hardware

* **NVIDIA RTX 4060 Ti GPU (CUDA):**
    * Maximum observed performance (up to 675×).
* **MacBook Pro M1 Pro (16-core GPU, MPS backend):**
    * Performance similar to 4060 Ti (approx. 3–10% lower).
* **Multi-core CPU:**
    * Moderate speedups (~20–40×).

> **Note:** PyTorch automatically selects the best available backend (`cuda` for NVIDIA, `mps` for Apple Silicon, or `cpu`).

---

## 🛠 Installation

### General Dependencies
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

✉️ Contact

David Michel Serrano Solís
davidser88@hotmail.com
