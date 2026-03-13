# Quantum Decision Web Crawler (BS4 + Parallel + Qiskit + Optional GPU)

(Still in Alpha Stage)

quantum-decision-crawler1.py - Video on YouTube

[![Watch the video](https://img.youtube.com/vi/UE0z3WZRcwc/hqdefault.jpg)](https://www.youtube.com/embed/UE0z3WZRcwc)

This project uses Qiskit. Qiskit is a registered trademark of IBM. 
This repository is independent and not affiliated with or endorsed by IBM.

News 3/6/2026: An article was written on The Quantum Dragon (feat. IQT News)!:

https://bsiegelwax.substack.com/p/quantum-web-crawler (3/1/2026)

Visit: https://www.inspiresearch.io/en for search related products and more.

Update: 2026-03-10 - `quantum-decision-crawler5.py` is the latest version (experimental). `quantum-decision-crawler4.py` is the latest stable version.

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

`quantum-decision-crawler4.py` adds a deterministic heuristic scorer that strongly
prefers high-value destinations by combining:

- **Same-host preference** – same-host links tend to be more topically relevant
- **URL brevity** – shorter, more canonical-looking URLs score higher
- **Anchor keyword matching** – an expanded keyword list covering:
  - documentation / specifications: `docs`, `api`, `reference`, `spec`, `guide`, `tutorial`, `handbook`, …
  - research / academia: `research`, `paper`, `whitepaper`, `report`, `publication`, …
  - product / company: `about`, `company`, `team`, `contact`, `careers`, `pricing`, `platform`, …
  - community / learning: `blog`, `news`, `learn`, `resources`, `developers`, `community`, `support`, …
  - navigation hubs: `overview`, `features`, `solutions`, `changelog`, `releases`, …
- **URL path keyword matching** – URL path segments are checked against the same
  high-value set (e.g. `/docs/`, `/about/`, `/research/`, `/api/` all receive a boost)
- **Low-value URL penalty** – URLs containing tokens like `signout`, `logout`, `cart`,
  `checkout`, `privacy-policy`, `terms-of-service`, `cookie-policy`, `share`, etc.
  are multiplied by `0.25` so they sink to the bottom of the priority queue without
  being completely eliminated
- **Depth and title** signals for shallower, more content-rich pages

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

The `quantum` strategy in comparison mode now uses the **full hybrid scoring pipeline**
(`_hybrid_score_candidate`: quantum circuit + heuristic + exploration noise), so it
makes meaningfully different priority decisions from BFS/DFS rather than collapsing
to near-BFS behaviour when quantum circuit outputs cluster around 0.5.

The summary table logged at the end of a comparison run now includes
**AvgPriority / MinPriority / MaxPriority** columns for each algorithm.  For `quantum`,
you should see a spread of scores (reflecting varied link quality); for `dfs` and `bfs`
these will be `N/A` since those strategies use a constant priority.

All `--quantum-weight`, `--heuristic-weight`, `--exploration-weight`, and
`--exploration-temperature` flags are respected in comparison mode so you can tune
the quantum prioritization behavior consistently across both normal and comparison runs.

This makes it easier to benchmark crawl behavior across strategies from the same seed set.

---

## What is new in `quantum-decision-crawler5.py`?

> ⚠️ **Experimental** — v5 needs further testing. Use at your own responsibility.

`quantum-decision-crawler5.py` extends v4 with two major additions: a **quantum-inspired annealing schedule** for the exploration temperature, and a **content relevance signal** as a fourth scoring dimension.

### Four-signal hybrid score

v5 promotes the scoring formula from v4's three signals to four:

```text
score(url) = quantum_weight    * Q(url)   [quantum gate-based signal]
           + heuristic_weight  * H(url)   [deterministic feature score]
           + relevance_weight  * R(url)   [keyword content-relevance]
           + exploration_weight * E(url)  [annealing-scaled noise]
```

All four weights are normalised internally to sum to 1.0.

### Quantum annealing schedule

The exploration temperature decays from `--annealing-initial-temp` (default 0.8) to
`--annealing-final-temp` (default 0.05) as the crawl progresses, following one of three
schedules selected by `--annealing-schedule`:

| Schedule      | Behaviour |
|---------------|-----------|
| `linear`      | T decreases at a constant rate |
| `cosine`      | T follows a cosine curve (fast drop at start and end, slow in middle) |
| `exponential` | T drops steeply early on |

This implements a quantum-inspired simulated annealing strategy: the crawler starts with
high exploration (broad search) and transitions toward exploitation (targeted, high-score
link following) as pages accumulate.

### Content relevance signal

`--relevance-keywords` accepts a comma-separated list of keywords. For each candidate URL,
the `ContentRelevanceScorer` checks whether any keyword appears in:

- the **URL path / query string** (always checked)
- the **anchor text** of the link (always checked)
- the **parent page title and snippet** (when `--content-boost` is enabled, which is the default)

The normalised match fraction (keywords matched / total keywords) forms R(url). When no
keywords are supplied, R(url) returns 0.5 (neutral) so the relevance signal has no effect.

### New CLI flags in v5 (versus v4)

```
--annealing-schedule {linear,cosine,exponential}
                        Decay schedule for exploration temperature (default: cosine)
--annealing-initial-temp T
                        Starting exploration temperature (default: 0.8)
--annealing-final-temp T
                        Minimum exploration temperature (default: 0.05)
--relevance-keywords KWS
                        Comma-separated relevance keywords, e.g. "AI,quantum,research"
--relevance-weight FLOAT
                        Weight for the content-relevance signal (default: 0.20)
--content-boost         Check parent title/snippet for keyword matches (default: on)
--no-content-boost      Disable parent-page content checking
--use-v4-crawler        Run HybridQuantumCrawler (v4 behaviour) as a control baseline
```

### New JSONL output fields in v5

- `adaptive_temperature` — exploration temperature in effect when the page was crawled
- `relevance_score` — mean relevance score of the children enqueued from this page

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
- `quantum-decision-crawler4.py` — v4 stable crawler with hybrid scoring and dead-end backtracking
- `quantum-decision-crawler5.py` — v5 experimental crawler with adaptive quantum annealing and content relevance scoring (needs further testing)
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

### Run the v5 adaptive crawler (experimental)

```bash
uv run quantum-decision-crawler5.py \
  --seeds seeds.txt \
  --out crawl.jsonl \
  --max-pages 200 \
  --max-depth 3 \
  --workers 16 \
  --force-curl \
  --dns-timeout 3.5 \
  --total-timeout 10 \
  --retries 1 \
  --relevance-keywords "AI,machine learning,quantum" \
  --annealing-schedule cosine \
  --annealing-initial-temp 0.9 \
  --annealing-final-temp 0.05 \
  --debug
```

### Tune v5 weights and annealing

```bash
uv run quantum-decision-crawler5.py \
  --seeds seeds.txt \
  --max-pages 200 \
  --max-depth 3 \
  --quantum-weight 0.35 \
  --heuristic-weight 0.20 \
  --exploration-weight 0.15 \
  --relevance-keywords "research,paper,dataset" \
  --relevance-weight 0.30 \
  --annealing-schedule exponential \
  --annealing-initial-temp 0.7 \
  --force-curl \
  --total-timeout 12
```

### Run v4-style base behavior from v5

```bash
uv run quantum-decision-crawler5.py \
  --seeds seeds.txt \
  --use-v4-crawler
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
- `adaptive_temperature` (v5 adaptive mode)
- `relevance_score` (v5 adaptive mode)
- `ts`

In comparison mode, the script writes flat benchmark-style records to CSV and detailed structured output to JSON.
