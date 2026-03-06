# Quantum Decision Web Crawler (BS4 + Parallel + Qiskit + Optional GPU)

(Still in Alpha Stage)

[![Watch the video](https://img.youtube.com/vi/UE0z3WZRcwc/hqdefault.jpg)](https://www.youtube.com/embed/UE0z3WZRcwc)

[<img src="https://img.youtube.com/vi/UE0z3WZRcwc/hqdefault.jpg" width="600" height="300"
/>](https://www.youtube.com/embed/UE0z3WZRcwc)


Visit: https://www.inspiresearch.io/en for search related products and more.

Update: 3/6/2026 19:07PM - quantum-decision-crawler3.py is the latest version. It has algorithm comparing benchmark mode.

This project is a **recursive web crawler** that prioritizes which URL to crawl next using a **quantum-inspired decision pipeline** backed by **Qiskit**. It starts from `seeds.txt` (one URL per line), follows links up to a configurable depth, and runs multiple fetch/parse workers in parallel.

The crawler is designed to be practical (Beautiful Soup 4 + requests) while exploring a “quantum decision” approach to frontier prioritization.

---

## What decision is it making?

This crawler implements **priority-based best-first crawling**:

1. Every time a page is fetched and parsed, it extracts outlinks.
2. Each candidate outlink receives a **priority score** in **[0, 1]**.
3. The crawler’s frontier is a **max-priority queue** (best-first).
4. The next URL crawled is always the **highest-priority item in the frontier** (ties broken by insertion order).

So the core decision is:

> **“Given all discovered links so far, crawl the link with the highest quantum decision score next.”**

---

## Decision pipeline (how a link is scored)

Each candidate outlink goes through the following pipeline to produce a final priority score.

### 1) Feature state construction (4×4 complex matrix)

For each outlink, the crawler builds a **normalized 4×4 complex-valued “feature state”** from heuristic signals:

**Signals used**
- **URL structure**
  - URL length (normalized)
  - path length (normalized)
- **Anchor / label context**
  - anchor text length (normalized)
  - keyword hit in anchor (`research`, `paper`, `docs`, `api`, `about`)
- **Relationship to the parent page**
  - same-host flag (candidate host == parent host)
  - depth penalty (deeper links slightly penalized)
- **Parent page signal**
  - parent title length (normalized)
- **Exploration noise**
  - `novelty = random()` injects mild exploration

These values populate the **real part** of a 4×4 matrix. A small hashed sinusoid adds an **imaginary component** to create a complex state. The state is normalized to behave like a “quantum-style” input.

### 2) Cohesive 4D transform (structured complex transform)

The feature state is passed through a **CohesiveDiagonalMatrix4D** transformation:

- Contracts a fixed 4D diagonal tensor with the 4×4 input state
- Normalizes the output
- Records fidelity/coherence metrics and optionally adapts the tensor over time

This produces a transformed 4×4 complex matrix, which becomes the input for Qiskit parameterization.

### 3) Qiskit “goodness probability” model

The transformed 4×4 matrix is reduced to circuit parameters:

- `theta[0..3]`: derived from row-wise mean magnitudes
- `phi[0..3]`: derived from row-wise mean phases

Those parameters are bound into a **5-qubit circuit**:
- 4 feature qubits
- 1 “goodness” qubit
- parameterized `RY` rotations on feature qubits
- entangling `CX` chain
- controlled interactions into the goodness qubit
- measurement of the goodness qubit

The crawler’s quantum score is:

> `p_good = P(goodness_qubit == 1)`

It is computed either by:
- **Qiskit Aer SamplerV2** (shot-based), or
- **Statevector fallback** (exact probability), if Aer is unavailable.

### 4) Final priority score

The final score used for frontier ordering is:

Where:
- `same_host_bonus = 1` if candidate host == parent host else `0`
- `depth_bonus` favors shallower pages (closer to seeds)

---

## Recursive crawling behavior

- Starts at depth 0 from seeds.
- Every parsed page’s outlinks are normalized and enqueued at `depth + 1`.
- Crawling continues until:
  - frontier is empty, or
  - hard page limit is reached, or
  - maximum depth is exceeded.

---

## HARD `--max-pages` enforcement

This crawler enforces `--max-pages` as a **hard upper bound on submitted tasks**.

Why this matters: in parallel crawling, if you only stop when “pages completed” reaches max, you can overshoot due to in-flight work. This implementation:

- Reserves a “submission slot” before scheduling a worker
- Stops submitting new work once slots are exhausted
- Signals workers to stop early and prevents writing output after the stop event

Result: **you don’t exceed `--max-pages` even with high `--workers`.**

---

## GPU notes (two separate GPU opportunities)

### A) GPU math for the feature state / cohesive transform (CuPy)
If CuPy is available, the feature-state construction and cohesive 4D transform can run on GPU (CuPy).

Enable with `--gpu-math`.

**Important:** CuPy requires:
- a working NVIDIA driver (`nvidia-smi`)
- a CuPy wheel matching your CUDA major version (`cupy-cuda11x`, `cupy-cuda12x`, etc.)
- CUDA runtime libraries (cuBLAS, etc.) available

### B) Qiskit Aer GPU
The script **requests** Aer GPU if available, but Aer will only actually use GPU if:
- you installed an Aer build with GPU support
- runtime libraries are present

Even with GPU Aer, note that small circuits (like 5 qubits) may not benefit much.

---

## Files

- `test-final.py` — main crawler script
- `seeds.txt` — seed URLs (one per line)
- `crawl.jsonl` — output (one JSON object per crawled page)

---

## Installation

### Using `uv` (recommended)

Create a venv and install dependencies:

```bash
uv venv
/opt/qiskit-quantum-crawler/.venv/bin/python3 -m ensurepip --upgrade
uv pip install -U pip

uv pip install requests beautifulsoup4 numpy qiskit qiskit-aer
