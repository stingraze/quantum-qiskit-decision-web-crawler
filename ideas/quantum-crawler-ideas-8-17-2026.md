(C)Tsubasa Kato - Inspire Search Corp. 2026/8/17 - Created with Perplexity Pro.

For millions to billions of websites, do **not** try to encode the corpus—or each page’s raw content—into a quantum state. Treat Qiskit as a per-decision coprocessor: a classical system retrieves a small candidate batch and compresses each candidate into a fixed, low-dimensional feature vector that is encoded into a circuit.

IBM describes data encoding/loading as a feature map from classical data into Hilbert space; the number of circuit features is tied to the number of qubits for common Qiskit feature maps. [quantum.cloud.ibm](https://quantum.cloud.ibm.com/learning/courses/quantum-machine-learning/data-encoding)

## Study note: large web data → Qiskit

### 1. Correct mental model

Your crawler’s large-scale state stays classical:

```text
Billions of URLs, pages, link edges, embeddings, metrics
        ↓
Classical storage + indexing + stream processing
        ↓
Candidate retrieval: top K URLs for this scheduling decision
        ↓
Feature extraction and compression
        ↓
d-dimensional vector per candidate, e.g. d = 8–32
        ↓
Parameterized Qiskit feature map / variational circuit
        ↓
Score, class label, probability distribution, or selected batch
        ↓
Classical frontier scheduler and fetch workers
```

The quantum circuit does **not** store the crawl corpus. It evaluates a compact representation of one candidate, a candidate pair, or a small constrained batch.

A useful practical frame:

\[
\text{billions of pages}
\rightarrow
\text{classical retrieval}
\rightarrow
K \text{ candidates}
\rightarrow
d \text{ engineered features}
\rightarrow
\text{quantum circuit}
\]

Typical values for an early system:

- Corpus: \(10^6\) to \(10^9+\) URLs/pages.
- Classical prefilter: \(K = 32\) to \(1{,}024\) candidates per scheduling cycle.
- Quantum input dimension: \(d = 4\), \(8\), \(12\), or \(16\) features.
- Circuit: one circuit evaluation per candidate or per candidate pair, usually batched.
- Final scheduling: classical, policy-constrained, auditable.

## 2. Separate population from representation

A website has many possible raw fields:

```text
URL, host, path tokens, anchor text, HTML, DOM structure,
outgoing links, incoming-link statistics, language, topic embedding,
freshness, content hash, MIME type, response time, change rate,
robots policy, crawl depth, spam score, historic yield, etc.
```

You must not map all of those fields directly to qubits. Instead, build a **feature contract**: a stable vector whose elements each have a semantic and bounded meaning.

Example: a 12-feature vector for crawl prioritization:

| Index | Feature | Example classical construction |
|---:|---|---|
| 0 | Seed/topic relevance | Cosine similarity of page/anchor embedding to seed embedding |
| 1 | Novelty | \(1 -\) maximum similarity to already indexed pages |
| 2 | Predicted information yield | Classical model estimate from prior fetch outcomes |
| 3 | Freshness need | Time since last successful crawl, adjusted by estimated change rate |
| 4 | Link authority | Log-scaled PageRank, host graph score, or link-quality estimate |
| 5 | Anchor-text confidence | Relevance score from incoming anchor text |
| 6 | Host reputation | Historical useful-content ratio, spam/trap indicators |
| 7 | Fetch cost | Estimated bytes, latency, errors, and render cost |
| 8 | Crawl depth | Normalized depth from target seeds |
| 9 | Duplicate risk | SimHash/MinHash or embedding-neighbor duplication estimate |
| 10 | Host budget pressure | Current requests relative to host-specific crawl budget |
| 11 | Exploration score | Uncertainty, rarity, or bandit exploration value |

The result for each URL is:

\[
x_u =
[x_0, x_1, \ldots, x_{11}] \in \mathbb{R}^{12}
\]

This vector—not the page—is the quantum input.

## 3. Build a classical compression pipeline

For web-scale data, most engineering value is in deriving reliable, cheap, bounded features.

### A. Keep raw and derived stores separate

```text
Raw pages / links / logs
   → object store, WARC, document DB, graph store

Derived web-scale features
   → columnar tables, feature store, vector DB, key-value cache

Quantum-ready feature vectors
   → compact float arrays, e.g. float32 [docs.quantum.ibm](https://docs.quantum.ibm.com/api/qiskit/0.37/primitives)
```

Do not query HTML or recalculate embeddings inside the quantum scheduling path. Precompute or incrementally update the features as pages are fetched.

### B. Use classical embeddings first

For text-heavy pages, create a classical embedding of title, main content, metadata, and anchor text. A page embedding may be hundreds or thousands of dimensions, so compress it before Qiskit:

\[
e_{\text{raw}} \in \mathbb{R}^{768}
\rightarrow
\operatorname{PCA}(e_{\text{raw}}) \in \mathbb{R}^{8}
\]

You can then combine those 8 reduced semantic dimensions with 4 operational crawl features to produce a 12-dimensional circuit input.

Appropriate classical reduction methods:

- PCA or IncrementalPCA for very large continuous embedding streams.
- Random projection for low-cost, online dimensionality reduction.
- Feature hashing for high-cardinality URL/path/domain token features.
- Autoencoders when you have sufficient data and a reason to learn compression.
- Aggregate statistics for graph-scale values: logarithms, percentiles, bins, and deciles.

## 4. Normalize before encoding

Angle-based and Pauli feature maps require bounded numerical inputs. Your distribution policy matters more than the exact gate choice.

For a raw feature \(v\), robustly clip it to historical quantiles before scaling:

\[
v' = \operatorname{clip}(v, q_{0.01}, q_{0.99})
\]

Then normalize to \([0, 1]\):

\[
z = \frac{v' - q_{0.01}}{q_{0.99} - q_{0.01}}
\]

For angle encoding, map to an interval such as \([0,\pi]\) or \([-\pi,\pi]\):

\[
\theta = \pi z
\]

Practical rules:

- Fit normalization statistics only on past/training data; save the exact version used.
- Clamp outliers. Web data has adversarial, malformed, and extreme values.
- Use a fixed feature ordering; changing it invalidates the trained model.
- Explicitly encode missingness: either impute and add a missing flag, or reserve a value with a corresponding mask feature.
- Use log transforms for heavily skewed quantities such as inlinks, bytes, crawl depth, host size, and latency.

## 5. Choose the encoding family

IBM identifies basis, amplitude, angle, and dense encoding as core data-encoding approaches. In practice, a web crawler prototype should start with **angle/Pauli feature maps**, not amplitude encoding. [quantum.cloud.ibm](https://quantum.cloud.ibm.com/learning/courses/quantum-machine-learning/data-encoding)

| Encoding | Input capacity | Strength | Major constraint | Web-crawler fit |
|---|---:|---|---|---|
| Basis encoding | 1 bit per qubit | Simple categorical flags | Does not naturally express continuous features | Good for a few booleans: robots allowed, HTTPS, prior error, duplicate flag |
| Angle encoding | About 1 continuous feature per qubit per layer | Simple, shallow, hardware-friendly | Requires dimension reduction | Best starting point |
| Pauli / ZZ feature map | Continuous features plus entangling interactions | Captures pairwise feature interactions | More two-qubit gates and noise sensitivity | Strong baseline for 4–16 curated features |
| Amplitude encoding | Up to \(2^n\) vector entries in \(n\) qubits | Theoretical compact state representation | General state preparation can be expensive; input must be normalized | Research experiment, not first production path |
| Dense / multi-feature encoding | Multiple variables via richer gate structures | Can use circuit structure efficiently | More design and training complexity | Later-stage experiment |

The Qiskit Pauli feature-map API supports control over feature dimension, repetitions, entanglement structure, Pauli terms, and the data-mapping function. Current documentation recommends the `pauli_feature_map` function rather than the older `PauliFeatureMap` class as that class heads toward removal in Qiskit 3.0. [quantum.cloud.ibm](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.0/qiskit.circuit.library.pauli_feature_map)

## 6. Start with angle encoding

With \(d\) normalized features, use \(d\) qubits and encode each feature as a rotation:

\[
|\psi(x)\rangle =
\bigotimes_{i=0}^{d-1}
R_y(x_i)R_z(x_i)|0\rangle
\]

A minimal 8-feature Qiskit circuit:

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

FEATURE_DIM = 8
x = ParameterVector("x", FEATURE_DIM)

feature_map = QuantumCircuit(FEATURE_DIM, name="web_features")

for i in range(FEATURE_DIM):
    feature_map.ry(x[i], i)
    feature_map.rz(x[i], i)

for i in range(FEATURE_DIM - 1):
    feature_map.cx(i, i + 1)
```

Bind one normalized feature vector only at execution time:

```python
raw = np.array([
    0.81,  # topical relevance
    0.62,  # novelty
    0.55,  # predicted yield
    0.44,  # freshness need
    0.70,  # authority
    0.48,  # anchor confidence
    0.12,  # fetch cost
    0.33,  # duplicate risk
], dtype=np.float64)

angles = np.pi * np.clip(raw, 0.0, 1.0)
bound_circuit = feature_map.assign_parameters(dict(zip(x, angles)))
```

The mapping itself is a feature map: Qiskit’s QML guidance frames encoding as transforming a classical vector into a quantum state and then measuring properties of that state. [quantum.cloud.ibm](https://quantum.cloud.ibm.com/learning/courses/quantum-machine-learning/introduction)

## 7. Add controlled feature interactions

A pure per-qubit rotation circuit does not explicitly model relationships such as:

- High topic relevance **and** low duplicate risk.
- High freshness need **and** low fetch cost.
- Strong authority **but** high host-budget pressure.
- High semantic relevance **with** a likely crawler trap.

For that, use a modest entangling feature map. The standard Qiskit Z feature map has only first-order \(Z\) terms and no entangling gates, while ZZ/Pauli maps add interactions. [quantum.cloud.ibm](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.z_feature_map)

A conservative 8-feature Pauli/ZZ-style starting configuration:

```python
from qiskit.circuit.library import pauli_feature_map

feature_map = pauli_feature_map(
    feature_dimension=8,
    reps=1,
    entanglement="linear",
    paulis=["Z", "ZZ"],
)
```

Use **linear entanglement** first:

```text
q0 — q1 — q2 — q3 — q4 — q5 — q6 — q7
```

Avoid full all-to-all entanglement initially. It grows the number of two-qubit operations, complicates transpilation, and can worsen noise on current hardware. Only increase circuit depth or interaction density when a controlled offline experiment demonstrates a gain over your classical baseline.

## 8. Candidate batching, not corpus loading

Suppose your crawler has 1 billion URL records. At one scheduling event:

1. Query the classical frontier for 10,000 eligible URLs.
2. Apply deterministic safety rules: `robots.txt`, host rate limit, retry policy, MIME policy, URL normalization, deny/allow lists.
3. Score and prefilter classically to 128 candidates.
4. Materialize \(128 \times 8\) normalized values.
5. Bind the same 8-qubit circuit to 128 parameter sets.
6. Submit them in batches through Qiskit primitives or local Aer simulation.
7. Use quantum-derived outputs as a feature or reranking signal.
8. Schedule the next 10–50 URLs with a classical host-aware scheduler.

Do not build one giant circuit with every candidate represented in superposition unless you have a specifically justified oracle and a very small, fixed problem formulation. A live frontier changes continuously, and loading/querying its state is itself a dominant systems problem.

Qiskit primitives are intended as reusable computational building blocks; the `Sampler` produces circuit outcome probabilities or quasi-probabilities, while the `Estimator` estimates expectation values for observables. [quantum.cloud.ibm](https://quantum.cloud.ibm.com/docs/en/api/qiskit/primitives)

## 9. Two recommended experiment paths

### Path A: quantum-kernel reranker

Use this when you have labeled URLs/pages: relevant vs. irrelevant, useful vs. low-value, or likely-good-fetch vs. likely-wasted-fetch.

```text
Classical features (8–16)
  → scaling / PCA / feature selection
  → Qiskit ZZ or Pauli feature map
  → quantum kernel
  → classical SVM or kernel classifier
  → relevance probability
  → combine with cost, host budget, and freshness
```

Qiskit’s quantum-kernel tutorials use a feature map to embed classical data in a higher-dimensional feature space, and projected-kernel workflows can extract measured reduced-density-matrix features for subsequent classical kernel processing. [quantum.cloud.ibm](https://quantum.cloud.ibm.com/docs/en/tutorials/projected-quantum-kernels)

### Path B: variational quantum scorer

Use this when you want a scalar priority score rather than a pairwise kernel matrix.

```text
8–12 normalized URL features
  → angle / Pauli feature map
  → shallow trainable ansatz
  → expectation value, e.g. ⟨Z₀⟩
  → priority score in [-1, 1]
  → classical scheduler constraints
```

The Qiskit Machine Learning project supports building variational quantum models from a feature map and ansatz, including VQC and regression-style workflows. [qiskit-community.github](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html)

## 10. Evaluation protocol

Do not evaluate by circuit accuracy alone. Measure whether the hybrid policy produces better crawl outcomes than an equivalent classical system.

Track:

- **Useful-page yield:** useful pages / pages fetched.
- **Cost-adjusted yield:** useful pages per MB, CPU second, browser-render second, or host request.
- **Time-to-discovery:** time until key pages/entities/documents appear.
- **Unique-content rate:** low-duplicate useful pages per 1,000 fetches.
- **Domain concentration:** whether the system over-focuses on a few hosts.
- **Freshness:** rate at which changed pages are rediscovered.
- **Constraint violations:** robots, host-rate, retry, and crawl-budget failures—target zero.
- **Quantum overhead:** queueing, transpilation, execution, and postprocessing cost.
- **Baseline lift:** compare against logistic regression, gradient boosting, a standard neural ranker, random sampling, and heuristic priority queues.

Run an A/B or interleaved experiment:

```text
50% crawl budget → your best classical prioritizer
50% crawl budget → same prioritizer + quantum score

Keep:
- identical seed sets
- identical host budgets
- identical crawler resources
- identical time windows
```

If the quantum component does not improve cost-adjusted yield or another explicit operational goal, keep it in the research lane rather than the production critical path.

## Recommended first implementation

Build this first:

```text
Classical page/URL feature store
    → 8 normalized features
    → 8-qubit, one-repetition, linear-entanglement ZZ/Pauli feature map
    → quantum-kernel classifier or shallow variational scorer
    → output one “quantum relevance” value
    → classical final priority:
       0.45 relevance
     + 0.20 novelty
     + 0.15 freshness
     + 0.10 exploration
     - 0.10 fetch cost
    → host-aware queue
```

Keep the quantum output as **one bounded signal** in a transparent classical policy. This lets you test it rigorously, roll it back safely, and scale the crawler to billions of records without making quantum state preparation or hardware calls the bottleneck.
