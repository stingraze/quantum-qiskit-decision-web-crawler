# Quantum Decision Web Crawler (BS4 + Parallel + Qiskit + Optional GPU)

(Still in Alpha Stage)

quantum-decision-crawler1.py - Video on YouTube

[![Watch the video](https://img.youtube.com/vi/UE0z3WZRcwc/hqdefault.jpg)](https://www.youtube.com/embed/UE0z3WZRcwc)

News 3/6/2026: An article was written on The Quantum Dragon (feat. IQT News)!: https://bsiegelwax.substack.com/p/quantum-web-crawler (3/1/2026)

https://bsiegelwax.substack.com/p/quantum-web-crawler

Visit: https://www.inspiresearch.io/en for search related products and more.

Update: 2026-03-06 - `quantum-decision-crawler4.py` is the latest version.

This project is a **recursive web crawler** that prioritizes which URL to crawl next using a **quantum-inspired decision pipeline** backed by **Qiskit**. It starts from `seeds.txt` (one URL per line), follows links up to a configurable depth, and runs multiple fetch/parse workers in parallel.

The crawler is designed to be practical (Beautiful Soup 4 + requests) while exploring a “quantum decision” approach to frontier prioritization.

---

## What is new in `quantum-decision-crawler4.py`?

`quantum-decision-crawler4.py` extends v3 with a new **hybrid quantum/probabilistic crawler** and **dead-end backtracking**.

Major additions in v4:

- **Hybrid scoring** that combines three signals:
  - **Quantum score** from the Qiskit-based decision circuit
  - **Heuristic score** from URL, anchor text, title, same-host, and depth features
  - **Exploration noise** for controlled diversification
- **Normalized weights** for:
  - `--quantum-weight`
  - `--heuristic-weight`
  - `--exploration-weight`
- **Exploration temperature** with `--exploration-temperature` to control how random the exploration component is
- **Dead-end detection** with `--dead-end-threshold`
- **Backtracking boost** with:
  - `--backtrack-boost`
  - `--dead-ends-before-boost`
- **Base crawler fallback** with `--use-base-crawler` so you can run v3-style behavior as a control baseline
- **Algorithm comparison mode** that can compare `quantum`, `dfs`, and `bfs` and export results to CSV and JSON

The v4 hybrid priority is conceptually:

```text
score(url) = quantum_weight * Q(url)
           + heuristic_weight * H(url)
           + exploration_weight * E(url)
```

This means the crawler is no longer purely quantum-ranked. It can balance exploitation, deterministic guidance, and exploration.

---

## What decision is it making?

This crawler implements **priority-based best-first crawling**:

1. Every time a page is fetched and parsed, it extracts outlinks.
2. Each candidate outlink receives a **priority score** in **[0, 1]**.
3. The crawler’s frontier is a **max-priority queue** (best-first).
4. The next URL crawled is always the **highest-priority item in the frontier** (ties broken by insertion order).

In `quantum-decision-crawler4.py`, that priority can be either:

- the original **quantum score** when `--use-base-crawler` is enabled, or
- the new **hybrid score** when using the default v4 crawler

So the core decision is:

> **“Given all discovered links so far, crawl the link with the highest decision score next.”**

---

## Decision pipeline (how a link is scored)

Each candidate outlink goes through the following pipeline to produce a final priority score.

### 1) Feature state construction (4×4 complex matrix)

For each outlink, the crawler builds a **normalized 4×4 complex-valued feature state** from heuristic signals.

**Signals used**
- **URL structure**
  - URL length (normalized)
  - path length (normalized)
- **Anchor / label context**
  - anchor text length (normalized)
  - keyword hits such as `research`, `paper`, `docs`, `api`, `about`, `guide`, `learn`, `tutorial`, `article`, `spec`, `reference`
- **Relationship to the parent page**
  - same-host flag (candidate host == parent host)
  - depth penalty (deeper links slightly penalized)
- **Parent page signal**
  - parent title length (normalized)
- **Exploration signal**
  - a random or temperature-controlled diversification component

These values populate the real part of a 4×4 matrix. A small hashed sinusoid adds an imaginary component to create a complex state. The state is normalized to behave like a quantum-style input.

### 2) Cohesive 4D transform (structured complex transform)

The feature state is passed through a **CohesiveDiagonalMatrix4D** transformation:

- contracts a fixed 4D diagonal tensor with the 4×4 input state
- normalizes the output
- records fidelity/coherence metrics
- can adapt the tensor over time

This produces a transformed 4×4 complex matrix, which becomes the input for Qiskit parameterization.

### 3) Qiskit goodness-probability model

The transformed 4×4 matrix is reduced to circuit parameters:

- `theta[0..3]`: derived from row-wise mean magnitudes
- `phi[0..3]`: derived from row-wise mean phases

Those parameters are bound into a **5-qubit circuit**:

- 4 feature qubits
- 1 goodness qubit
- parameterized `RY` rotations on feature qubits
- entangling `CX` chain
- controlled interactions into the goodness qubit
- measurement of the goodness qubit

The crawler’s quantum score is:

> `p_good = P(goodness_qubit == 1)`

It is computed either by:

- **Qiskit Aer SamplerV2** (shot-based), or
- **Statevector fallback** (exact probability), if Aer is unavailable.

### 4) Heuristic score in v4

`quantum-decision-crawler4.py` adds a deterministic heuristic scorer that prefers:

- same-host links
- shorter, more canonical-looking URLs
- useful anchor keywords
- informative anchor text length
- shallower depth
- parent pages with meaningful titles

### 5) Exploration score in v4

v4 also adds an exploration component that blends:

- a hash-derived pseudo-deterministic signal per URL, and
- true randomness

This is scaled by `--exploration-temperature` so low values stay mostly deterministic and high values increase exploration.

### 6) Final priority score in v4

By default in v4, the final score used for frontier ordering is the weighted combination of:

- quantum score
- heuristic score
- exploration score

If enough dead ends occur in the same branch, v4 can also add a small diversity bonus and re-boost sibling URLs already in the frontier.

---

## Dead-end backtracking in v4

A page is treated as a **dead end** when it enqueues fewer new links than `--dead-end-threshold`.

When enough dead ends accumulate within the same branch:

- sibling URLs discovered from the same parent can be reinserted into the frontier
- their priority is increased by `--backtrack-boost`
- previous lower-priority entries are effectively ignored later through lazy deletion

This helps the crawler avoid getting stuck in a narrow or unproductive subtree.

The crawler also writes `hybrid_branch_dead_ends` into JSONL output records so you can analyze backtracking behavior later.

---

## Comparison mode in v4

`quantum-decision-crawler4.py` can run a comparison workflow with:

- `quantum`
- `dfs`
- `bfs`

When `--compare-algorithms` is enabled, the script runs each strategy for `--compare-duration` seconds and writes:

- `comparison.csv`
- `comparison.json`

This makes it easier to benchmark crawl behavior across strategies from the same seed set.

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

Why this matters: in parallel crawling, if you only stop when pages completed reaches max, you can overshoot due to in-flight work. This implementation:

- reserves a submission slot before scheduling a worker
- stops submitting new work once slots are exhausted
- signals workers to stop early and prevents writing output after the stop event

Result: **you don’t exceed `--max-pages` even with high `--workers`.**

---

## GPU notes (two separate GPU opportunities)

### A) GPU math for the feature state / cohesive transform (CuPy)
If CuPy is available, the feature-state construction and cohesive 4D transform can run on GPU.

Enable with `--gpu-math`.

**Important:** CuPy requires:
- a working NVIDIA driver (`nvidia-smi`)
- a CuPy wheel matching your CUDA major version (`cupy-cuda11x`, `cupy-cuda12x`, etc.)
- CUDA runtime libraries available

### B) Qiskit Aer GPU
The script requests Aer GPU if available, but Aer will only actually use GPU if:
- you installed an Aer build with GPU support
- runtime libraries are present

Even with GPU Aer, note that small circuits like 5 qubits may not benefit much.

---

## Files

- `quantum-decision-crawler1.py` — early crawler version, referenced by the video
- `quantum-decision-crawler2.py` — later iteration
- `quantum-decision-crawler3.py` — v3 crawler with benchmark/comparison support
- `quantum-decision-crawler4.py` — latest crawler with hybrid scoring and dead-end backtracking
- `seeds.txt` — seed URLs (one per line)
- `crawl.jsonl` — output for normal crawling (one JSON object per crawled page)
- `comparison.csv` — optional comparison-mode CSV output
- `comparison.json` — optional comparison-mode JSON output

---

## Installation

### Using `uv` (recommended)

Create a venv and install dependencies:

```bash
uv venv
python3 -m ensurepip --upgrade
uv pip install -U pip
uv pip install requests beautifulsoup4 numpy qiskit qiskit-aer lxml
```

Optional GPU math support:

```bash
uv pip install cupy-cuda12x
```

Install the CuPy package that matches your CUDA version.

---

## Example usage

### Run the latest hybrid crawler

```bash
uv run quantum-decision-crawler4.py \
  --seeds seeds.txt \
  --out crawl.jsonl \
  --max-pages 200 \
  --max-depth 3 \
  --workers 16 \
  --force-curl \
  --dns-timeout 3.5 \
  --total-timeout 10 \
  --retries 1 \
  --debug
```

### Tune hybrid weights

```bash
uv run quantum-decision-crawler4.py \
  --seeds seeds.txt \
  --max-pages 200 \
  --max-depth 3 \
  --quantum-weight 0.50 \
  --heuristic-weight 0.30 \
  --exploration-weight 0.20 \
  --exploration-temperature 0.35 \
  --dead-end-threshold 3 \
  --backtrack-boost 0.30 \
  --dead-ends-before-boost 2 \
  --force-curl \
  --total-timeout 12
```

### Run v3-style base behavior from v4

```bash
uv run quantum-decision-crawler4.py \
  --seeds seeds.txt \
  --use-base-crawler
```

### Compare algorithms

```bash
uv run quantum-decision-crawler4.py \
  --seeds seeds.txt \
  --max-depth 3 \
  --compare-algorithms \
  --compare-duration 180 \
  --compare-csv comparison.csv \
  --compare-json comparison.json \
  --total-timeout 12 \
  --force-curl
```

---

## Output format

Normal crawl output is written to `crawl.jsonl`, one JSON object per line. Records can include fields such as:

- `url`
- `final_url`
- `depth`
- `status`
- `reason`
- `http_status`
- `content_type`
- `fetch_seconds`
- `fetch_engine`
- `title`
- `snippet`
- `out_links_found`
- `out_links_enqueued`
- `hybrid_branch_dead_ends` (v4 hybrid mode)
- `ts`

In comparison mode, the script writes flat benchmark-style records to CSV and detailed structured output to JSON.
