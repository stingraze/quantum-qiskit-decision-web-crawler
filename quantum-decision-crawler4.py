#!/usr/bin/env python3
"""
(C)Tsubasa Kato - Inspire Search Corp.
Created with help of Gemini 3.1 Pro, Perplexity and ChatGPT 5.2 Thinking
Contact: tsubasa@inspiresearch.io for advanced web crawling consultation service and collaboration discussion.

Quantum Decision Crawler v4 -- hybrid quantum/probabilistic crawl strategy with dead-end backtracking.

Extends quantum-decision-crawler3.py with a new HybridQuantumCrawler that combines:
  - Quantum-gate-based scoring (inherited from QuantumCrawler in v3)
  - Heuristic scoring (URL/anchor/depth features -- deterministic guidance)
  - Controlled stochastic exploration noise (temperature-scaled diversification)
  - Dead-end detection and backtrack-aware priority adjustment

Hybrid selection strategy
--------------------------
    Score(url) = quantum_weight  * Q(url)   [quantum gate-based signal]
               + heuristic_weight * H(url)   [deterministic feature score]
               + exploration_weight * E(url) [controlled stochastic noise]

Weights are normalised internally so they always sum to 1.0.  The result is a
hybrid priority that is neither fully greedy (pure quantum) nor fully random.

Exploration temperature
-----------------------
--exploration-temperature (0.0-1.0) controls how much the stochastic component
diverges from deterministic signals.  0.0 = no noise; 1.0 = maximum randomness.

Dead-end backtracking
---------------------
A page is a *dead end* when it enqueues fewer new links than --dead-end-threshold.

After --dead-ends-before-boost dead ends accumulate for the same parent branch,
all unvisited sibling URLs still in the frontier are re-inserted with priority
boosted by --backtrack-boost.  Old (lower-priority) entries are silently skipped
via lazy-deletion (they find the URL already visited when eventually popped).

New CLI flags versus v3
-----------------------
  --quantum-weight FLOAT        Weight for the quantum scoring signal (default 0.45)
  --heuristic-weight FLOAT      Weight for the heuristic scoring signal (default 0.35)
  --exploration-weight FLOAT    Weight for the stochastic noise signal (default 0.20)
  --exploration-temperature T   Noise randomness 0.0-1.0 (default 0.4)
  --dead-end-threshold N        Min new links to not be a dead end (default 2)
  --backtrack-boost FLOAT       Priority boost on backtrack event (default 0.25)
  --dead-ends-before-boost N    Dead ends per branch before siblings are boosted (default 1)
  --use-base-crawler            Run plain QuantumCrawler (v3 behaviour) as control

Example usage
-------------
uv run quantum-decision-crawler4.py \
  --seeds seeds.txt --max-pages 200 --max-depth 3 --workers 16 \
  --force-curl --dns-timeout 3.5 --total-timeout 10 --retries 1 --debug

# Custom hybrid weights + aggressive backtracking:
uv run quantum-decision-crawler4.py \
  --seeds seeds.txt --max-pages 200 --max-depth 3 \
  --quantum-weight 0.5 --heuristic-weight 0.3 --exploration-weight 0.2 \
  --exploration-temperature 0.35 \
  --dead-end-threshold 3 --backtrack-boost 0.3 --dead-ends-before-boost 2 \
  --force-curl --total-timeout 12

# Comparison mode (3 algorithms x 3 minutes each):
uv run quantum-decision-crawler4.py \
  --seeds seeds.txt --max-depth 3 \
  --compare-algorithms --compare-duration 180 \
  --compare-csv comparison.csv --compare-json comparison.json \
  --total-timeout 12 --force-curl
"""

import csv
import os
import re
import json
import time
import math
import heapq
import random
import logging
import threading
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import urllib.robotparser as robotparser
import xml.etree.ElementTree as ET

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Statevector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("quantum_crawler")

# ----------------------------
# Optional GPU backend (CuPy)
# ----------------------------
_CUPY_OK = False
cp = None  # type: ignore
try:
    import cupy as _cp  # type: ignore
    cp = _cp  # type: ignore
    try:
        n = int(cp.cuda.runtime.getDeviceCount())
        if n > 0:
            _CUPY_OK = True
            dev = int(cp.cuda.runtime.getDevice())
            props = cp.cuda.runtime.getDeviceProperties(dev)
            name = props.get("name", b"")
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", errors="ignore")
            logger.info("[GPU] CuPy available. devices=%d active=%d name=%s", n, dev, name)
        else:
            logger.warning("[GPU] CuPy imported but no CUDA devices detected (deviceCount=0).")
            _CUPY_OK = False
    except Exception as e:
        logger.warning("[GPU] CuPy imported but CUDA runtime check failed: %s: %s", type(e).__name__, e)
        _CUPY_OK = False
except Exception as e:
    logger.info("[GPU] CuPy not available. Falling back to CPU. (%s: %s)", type(e).__name__, e)
    _CUPY_OK = False


def to_cpu(a):
    if cp is not None and _CUPY_OK and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


# ----------------------------
# URL extraction (SPA-friendly)
# ----------------------------
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
PATH_RE = re.compile(r"(?<![a-zA-Z0-9])/(?:[a-zA-Z0-9\-\._~%!$&'()*+,;=:@/]+)", re.IGNORECASE)

SKIP_PATH_PREFIXES = ("/_next/", "/static/", "/assets/", "/build/", "/dist/")
SKIP_EXTENSIONS = (
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".map", ".mp4", ".mp3", ".wav",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z",
)


def extract_urls_from_text(text: str, limit: int = 500) -> List[str]:
    if not text:
        return []
    return URL_RE.findall(text)[:limit]


def extract_paths_from_text(text: str, limit: int = 500) -> List[str]:
    if not text:
        return []
    raw = PATH_RE.findall(text)
    out: List[str] = []
    for p in raw[: 5 * limit]:
        if p.startswith("//"):
            continue
        if any(p.startswith(pref) for pref in SKIP_PATH_PREFIXES):
            continue
        if p.lower().endswith(SKIP_EXTENSIONS):
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


# ----------------------------
# Qiskit decision policy (unchanged from v3)
# ----------------------------
class QuantumDecisionPolicy:
    def __init__(self, shots: int = 256, seed: int = 1234, request_gpu: bool = True):
        self.shots = int(shots)
        self.seed = int(seed)
        self.request_gpu = bool(request_gpu)

        self._theta = ParameterVector("theta", length=4)
        self._phi = ParameterVector("phi", length=4)
        self._circuit = self._build_param_circuit()

        self._aer_sampler = None
        try:
            from qiskit_aer.primitives import SamplerV2 as AerSampler  # type: ignore

            backend_options = {"seed_simulator": self.seed}
            if self.request_gpu:
                backend_options["device"] = "GPU"
            backend_options["method"] = "statevector"

            self._aer_sampler = AerSampler(options=dict(backend_options=backend_options))
            logger.info("[Qiskit] Using Aer SamplerV2 (GPU requested=%s).", self.request_gpu)
        except Exception as e:
            self._aer_sampler = None
            logger.info("[Qiskit] Aer SamplerV2 not available; using Statevector fallback. (%s)", type(e).__name__)

    def _build_param_circuit(self) -> QuantumCircuit:
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        feature = [0, 1, 2, 3]
        qg = 4

        for i in feature:
            qc.h(q[i])
        for i in feature:
            qc.ry(self._theta[i], q[i])
            qc.ry(self._phi[i], q[i])

        for i in range(3):
            qc.append(CXGate(), [q[i], q[i + 1]])
        qc.append(CXGate(), [q[3], q[0]])

        for i in feature:
            qc.cx(q[i], q[qg])
            qc.ry(0.7, q[qg])
            qc.cx(q[i], q[qg])

        qc.h(q[qg])
        qc.measure(q[qg], c[0])
        return qc

    def _state_to_features(self, state4x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if state4x4.shape != (4, 4):
            state4x4 = state4x4.reshape(4, 4)

        mag = np.abs(state4x4)
        ang = np.angle(state4x4)

        mag_pool = mag.mean(axis=1)
        ang_pool = ang.mean(axis=1)

        mag_norm = mag_pool / (mag_pool.max() + 1e-9)
        theta = np.clip(mag_norm * np.pi, 0.0, np.pi)
        phi = np.clip((ang_pool + np.pi) / 2.0, 0.0, np.pi)
        return theta, phi

    def _score_statevector(self, bound_circuit: QuantumCircuit) -> float:
        qc_nom = bound_circuit.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_nom)
        probs = sv.probabilities()
        p1 = float(probs[1::2].sum())
        return max(0.0, min(1.0, p1))

    def score(self, state4x4_cpu: np.ndarray) -> float:
        theta, phi = self._state_to_features(state4x4_cpu)

        bind = {self._theta[i]: float(theta[i]) for i in range(4)}
        bind.update({self._phi[i]: float(phi[i]) for i in range(4)})

        qc = self._circuit.assign_parameters(bind, inplace=False)

        if self._aer_sampler is not None:
            try:
                pub = (qc, None, self.shots)
                result = self._aer_sampler.run([pub]).result()
                pub_result = result[0]
                counts = pub_result.data.meas.get_counts()
                c1 = counts.get("1", 0)
                return max(0.0, min(1.0, float(c1) / float(self.shots)))
            except Exception:
                return self._score_statevector(qc)

        return self._score_statevector(qc)


# ----------------------------
# Cohesive 4D transform (GPU-capable, unchanged from v3)
# ----------------------------
class CohesiveDiagonalMatrix4D:
    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.matrix_cpu = self._initialize_cohesive_matrix_np()
        self.matrix_gpu = None
        self.performance_history = []
        self.adaptation_count = 0

    def _initialize_cohesive_matrix_np(self) -> np.ndarray:
        m = np.zeros((self.dimension, self.dimension, self.dimension, self.dimension), dtype=np.complex128)
        for i in range(self.dimension):
            for j in range(self.dimension):
                phase = np.exp(1j * np.pi * (i + j) / self.dimension)
                amplitude = 1.0 / np.sqrt(self.dimension)
                m[i, j, i, j] = amplitude * phase
        return m

    def _ensure_gpu_matrix(self):
        if not (_CUPY_OK and cp is not None):
            return
        if self.matrix_gpu is None:
            self.matrix_gpu = cp.asarray(self.matrix_cpu)

    def decision_function(self, input_state):
        start = time.time()
        if input_state.shape != (self.dimension, self.dimension):
            input_state = input_state.reshape(self.dimension, self.dimension)

        if not (_CUPY_OK and cp is not None and isinstance(input_state, cp.ndarray)):
            result = np.tensordot(self.matrix_cpu, input_state, axes=([0, 1], [0, 1]))
            norm = np.linalg.norm(result)
            normalized = result if norm == 0 else (result / norm)
            fidelity = float(np.abs(np.vdot(input_state.flatten(), normalized.flatten())))
            self.performance_history.append({
                "processing_time": time.time() - start,
                "fidelity": fidelity,
                "coherence_measure": float(np.abs(np.trace(result))),
            })
            return normalized

        self._ensure_gpu_matrix()
        result = cp.tensordot(self.matrix_gpu, input_state, axes=([0, 1], [0, 1]))
        norm = cp.linalg.norm(result)
        normalized = result if float(norm) == 0.0 else (result / norm)
        fidelity = float(cp.abs(cp.vdot(input_state.ravel(), normalized.ravel())).get())
        coherence = float(cp.abs(cp.trace(result)).get())
        self.performance_history.append({
            "processing_time": time.time() - start,
            "fidelity": fidelity,
            "coherence_measure": coherence,
        })
        return normalized

    def adapt_matrix(self, performance_threshold: float = 0.8, window: int = 200) -> bool:
        if len(self.performance_history) < window:
            return False
        recent = self.performance_history[-window:]
        avg_fid = float(np.mean([p["fidelity"] for p in recent]))
        if avg_fid < performance_threshold:
            self.matrix_cpu *= 1.05
            self.matrix_gpu = None
            self.adaptation_count += 1
            logger.info("[Cohesive4D] Adapted matrix (count=%d) avg_fidelity=%.4f", self.adaptation_count, avg_fid)
            return True
        return False


@dataclass(order=True)
class FrontierItem:
    priority: float
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)
    discovered_from: str = field(compare=False, default="")


def load_seeds(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"seeds file not found: {path}")
    seeds: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            seeds.append(s)
    if not seeds:
        raise ValueError("seeds.txt is empty")
    return seeds


def host_of(url: str) -> str:
    return urlparse(url).netloc


def looks_like_html(text: str) -> bool:
    head = (text[:4096] if text else "").lower()
    if not head:
        return False
    return ("<html" in head) or ("<!doctype html" in head) or ("<head" in head) or ("<body" in head)


def normalize_url(base: str, href: str) -> Optional[str]:
    if href is None:
        return None
    href = href.strip()
    if not href:
        return None
    if href.startswith("#"):
        return None
    low = href.lower()
    if low.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return None

    try:
        abs_url = urljoin(base, href)
        abs_url, _ = urldefrag(abs_url)
        p = urlparse(abs_url)
        if p.scheme not in ("http", "https"):
            return None

        netloc = (p.netloc or "").lower()
        if netloc.endswith(":80") and p.scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and p.scheme == "https":
            netloc = netloc[:-4]

        path = p.path or "/"
        if len(abs_url) > 4096:
            return None

        return p._replace(netloc=netloc, path=path).geturl()
    except Exception:
        return None


class RobotsCacheEntry:
    def __init__(self, rp: robotparser.RobotFileParser, crawl_delay: Optional[float], sitemaps: List[str], fetched_at: float):
        self.rp = rp
        self.crawl_delay = crawl_delay
        self.sitemaps = sitemaps
        self.fetched_at = fetched_at


def _parse_robots_sitemaps(robots_text: str) -> List[str]:
    sitemaps: List[str] = []
    if not robots_text:
        return sitemaps
    for line in robots_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("sitemap:"):
            u = line.split(":", 1)[1].strip()
            if u:
                sitemaps.append(u)
    out: List[str] = []
    seen: Set[str] = set()
    for u in sitemaps:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _safe_xml_parse(content: str) -> Optional[ET.Element]:
    try:
        return ET.fromstring(content)
    except Exception:
        return None


def _xml_tag_localname(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


# ----------------------------
# Base crawler (v3-compatible)
# ----------------------------
class QuantumCrawler:
    """Base quantum-priority crawler from v3 (unchanged logic).

    This class is preserved in full so that quantum-decision-crawler4.py can
    be run standalone without importing v3, and to allow --use-base-crawler to
    exercise v3 behaviour as a control when benchmarking the hybrid strategy.
    """

    def __init__(
        self,
        seeds_path: str = "seeds.txt",
        out_jsonl: str = "crawl.jsonl",
        max_pages: int = 500,
        max_depth: int = 3,
        max_workers: int = 16,
        per_host_delay: float = 0.6,
        user_agent: str = "QuantumCrawler/1.0 (+https://example.invalid)",
        allow_regex: Optional[str] = None,
        deny_regex: Optional[str] = None,
        qiskit_shots: int = 128,
        debug: bool = False,
        gpu_math: bool = True,
        aer_request_gpu: bool = True,
        respect_robots: bool = True,
        use_sitemaps: bool = True,
        robots_cache_ttl_s: int = 6 * 3600,
        sitemap_max_urls_per_host: int = 5000,
        sitemap_max_index_depth: int = 2,
        connect_timeout: float = 5.0,
        read_timeout: float = 25.0,
        total_timeout: float = 20.0,
        retries: int = 1,
        backoff: float = 0.6,
        force_curl: bool = False,
        dns_timeout: float = 0.0,
    ):
        self.seeds_path = seeds_path
        self.out_jsonl = out_jsonl
        self.max_pages = int(max_pages)

        self.max_depth = max(1, int(max_depth))
        self.max_workers = int(max_workers)
        self.per_host_delay = float(per_host_delay)
        self.user_agent = user_agent
        self.user_agent_token = (user_agent.split(" ", 1)[0] or "QuantumCrawler").strip()
        self.debug = bool(debug)

        self.allow_re = re.compile(allow_regex) if allow_regex else None
        self.deny_re = re.compile(deny_regex) if deny_regex else None

        self.gpu_math = bool(gpu_math) and _CUPY_OK
        if bool(gpu_math) and not _CUPY_OK:
            logger.info("[GPU] gpu-math requested but unavailable; using CPU math.")

        self.cohesive = CohesiveDiagonalMatrix4D(4)
        self.qpolicy = QuantumDecisionPolicy(shots=int(qiskit_shots), request_gpu=bool(aer_request_gpu))

        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        self.total_timeout = float(total_timeout)
        self.retries = max(0, int(retries))
        self.backoff = max(0.0, float(backoff))
        self.force_curl = bool(force_curl)
        self.dns_timeout = float(dns_timeout)

        self._stop_event = threading.Event()
        self._submitted = 0
        self._submitted_lock = threading.Lock()

        self._frontier: List[Tuple[float, int, FrontierItem]] = []
        self._seq = 0
        self._frontier_lock = threading.Lock()

        self._seen: Set[str] = set()
        self._visited: Set[str] = set()
        self._seen_lock = threading.Lock()
        self._visited_lock = threading.Lock()

        self._host_next_ok: Dict[str, float] = {}
        self._host_lock = threading.Lock()

        self._pages_crawled = 0
        self._pages_lock = threading.Lock()

        self._out_lock = threading.Lock()

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        self.respect_robots = bool(respect_robots)
        self.use_sitemaps = bool(use_sitemaps)
        self.robots_cache_ttl_s = int(robots_cache_ttl_s)
        self.sitemap_max_urls_per_host = int(sitemap_max_urls_per_host)
        self.sitemap_max_index_depth = int(sitemap_max_index_depth)

        self._robots_cache: Dict[str, RobotsCacheEntry] = {}
        self._robots_lock = threading.Lock()
        self._sitemaps_seeded_hosts: Set[str] = set()

    def _try_reserve_page_slot(self) -> bool:
        with self._submitted_lock:
            if self._submitted >= self.max_pages:
                self._stop_event.set()
                return False
            self._submitted += 1
            if self._submitted >= self.max_pages:
                self._stop_event.set()
            return True

    def _allowed_by_regex(self, url: str) -> bool:
        if self.deny_re and self.deny_re.search(url):
            return False
        if self.allow_re and not self.allow_re.search(url):
            return False
        return True

    def _respect_host_delay(self, url: str):
        h = host_of(url)
        sleep_time = 0.0

        with self._host_lock:
            now = time.time()
            next_ok = self._host_next_ok.get(h, 0.0)
            if now < next_ok:
                sleep_time = next_ok - now
                self._host_next_ok[h] = next_ok + self.per_host_delay
            else:
                self._host_next_ok[h] = now + self.per_host_delay

        if sleep_time > 0:
            time.sleep(sleep_time)

    def _push_frontier(self, item: FrontierItem):
        with self._frontier_lock:
            self._seq += 1
            heapq.heappush(self._frontier, (-item.priority, self._seq, item))

    def _pop_frontier(self) -> Optional[FrontierItem]:
        with self._frontier_lock:
            if not self._frontier:
                return None
            _neg, _seq, item = heapq.heappop(self._frontier)
            return item

    def _frontier_size(self) -> int:
        with self._frontier_lock:
            return len(self._frontier)

    def _mark_seen(self, url: str) -> bool:
        with self._seen_lock:
            if url in self._seen:
                return False
            self._seen.add(url)
            return True

    def _mark_visited(self, url: str) -> bool:
        with self._visited_lock:
            if url in self._visited:
                return False
            self._visited.add(url)
            return True

    def _is_visited(self, url: str) -> bool:
        with self._visited_lock:
            return url in self._visited

    # ----------------------------
    # HARD timeout fetch (curl)
    # ----------------------------
    def _curl_fetch(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str], int, float, str]:
        start = time.time()

        max_time = self.total_timeout if self.total_timeout > 0 else (self.connect_timeout + self.read_timeout + 5.0)

        # IMPORTANT: curl has NO --dns-timeout. DNS is inside "connect phase".
        # We enforce dns_timeout by tightening --connect-timeout.
        connect_to = max(1.0, float(self.connect_timeout))
        if self.dns_timeout and self.dns_timeout > 0:
            connect_to = max(0.2, min(connect_to, float(self.dns_timeout)))

        marker = "CURLMETA:"
        cmd = [
            "curl",
            "-sS",
            "-L",
            "--compressed",
            "-A", self.user_agent,
            "--connect-timeout", f"{connect_to:.3f}",
            "--max-time", f"{float(max_time):.3f}",
            "-o", "-",
            "-w", f"\n{marker}%{{http_code}} %{{url_effective}} %{{content_type}}\n",
            url,
        ]

        try:
            hard = float(max_time) + 2.0
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=hard)
            sec = time.time() - start

            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
                if proc.returncode == 28:
                    return None, None, None, 0, sec, "curl_timeout"
                return None, None, None, 0, sec, f"curl_error:{proc.returncode}:{err[:200]}"

            out = (proc.stdout or b"").decode("utf-8", errors="replace")
            if marker not in out:
                return out, None, None, 0, sec, "curl_no_meta"

            body, meta = out.rsplit(marker, 1)
            meta = meta.strip()
            parts = meta.split(" ", 2)
            if len(parts) < 2:
                return body, None, None, 0, sec, "curl_bad_meta"

            try:
                status = int(parts[0])
            except Exception:
                status = 0
            final_url = parts[1].strip()
            content_type = parts[2].strip() if len(parts) >= 3 else ""

            return body, content_type.lower(), final_url, status, sec, "ok"
        except subprocess.TimeoutExpired:
            sec = time.time() - start
            return None, None, None, 0, sec, "subprocess_timeout"
        except FileNotFoundError:
            sec = time.time() - start
            return None, None, None, 0, sec, "curl_not_found"
        except Exception as e:
            sec = time.time() - start
            return None, None, None, 0, sec, f"curl_exception:{type(e).__name__}"

    def _requests_get(self, url: str) -> Tuple[Optional[requests.Response], float, str]:
        start_total = time.time()
        last_exc = None
        for attempt in range(self.retries + 1):
            if self._stop_event.is_set():
                return None, time.time() - start_total, "stopped"
            try:
                resp = self.session.get(url, timeout=(self.connect_timeout, self.read_timeout), allow_redirects=True)
                status = int(resp.status_code)
                if status == 429 or (500 <= status <= 599):
                    if attempt < self.retries:
                        time.sleep(self.backoff * (2 ** attempt) + random.random() * 0.1)
                        continue
                return resp, time.time() - start_total, "ok"
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (2 ** attempt) + random.random() * 0.1)
                    continue
                return None, time.time() - start_total, f"timeout:{type(e).__name__}"
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.backoff * (2 ** attempt) + random.random() * 0.1)
                    continue
                return None, time.time() - start_total, f"exception:{type(e).__name__}"
        return None, time.time() - start_total, f"exception:{type(last_exc).__name__ if last_exc else 'unknown'}"

    def _fetch(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str], int, float, str, str]:
        self._respect_host_delay(url)

        use_curl = self.force_curl or (self.total_timeout > 0) or (self.dns_timeout > 0)
        if use_curl:
            html, ct, final_url, status, sec, reason = self._curl_fetch(url)
            engine = "curl"
        else:
            resp, sec, reason = self._requests_get(url)
            engine = "requests"
            if resp is None:
                return None, None, None, 0, sec, reason, engine
            status = int(resp.status_code)
            ct = (resp.headers.get("Content-Type") or "").lower()
            final_url = resp.url
            html = resp.text or ""

        if self._stop_event.is_set():
            return None, ct, final_url, status, sec, "stopped", engine

        if status >= 400:
            if html and looks_like_html(html):
                return html, ct, final_url, status, sec, "ok_html_error_page", engine
            return None, ct, final_url, status, sec, "http_error", engine

        is_html = False
        if html:
            if ct and ("text/html" in ct or "application/xhtml" in ct):
                is_html = True
            else:
                is_html = looks_like_html(html)

        if not is_html:
            return None, ct, final_url, status, sec, "non_html", engine

        return html, ct, final_url, status, sec, "ok", engine

    # ---- robots/sitemap ----
    def _robots_url(self, base_url: str) -> str:
        p = urlparse(base_url)
        return f"{p.scheme}://{p.netloc}/robots.txt"

    def _fetch_text(self, url: str) -> Tuple[Optional[str], int, float, str]:
        if self.force_curl or (self.total_timeout > 0) or (self.dns_timeout > 0):
            body, _ct, _final, st, sec2, rsn = self._curl_fetch(url)
            return body, st, sec2, ("ok" if rsn == "ok" else rsn)
        self._respect_host_delay(url)
        resp, sec2, rsn = self._requests_get(url)
        if resp is None:
            return None, 0, sec2, rsn
        return (resp.text or ""), int(resp.status_code), sec2, "ok"

    def _get_robots_entry(self, any_url_on_host: str) -> Optional[RobotsCacheEntry]:
        host = host_of(any_url_on_host)
        now = time.time()
        with self._robots_lock:
            entry = self._robots_cache.get(host)
            if entry and (now - entry.fetched_at) <= self.robots_cache_ttl_s:
                return entry

        robots_url = self._robots_url(any_url_on_host)
        txt, status, _sec, _reason = self._fetch_text(robots_url)

        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        sitemaps: List[str] = []
        crawl_delay: Optional[float] = None

        if txt is None or status >= 400 or status == 0:
            rp.parse([])
            entry = RobotsCacheEntry(rp=rp, crawl_delay=None, sitemaps=[], fetched_at=now)
            with self._robots_lock:
                self._robots_cache[host] = entry
            return entry

        rp.parse(txt.splitlines())
        sitemaps = _parse_robots_sitemaps(txt)
        try:
            cd = rp.crawl_delay(self.user_agent_token)  # type: ignore[attr-defined]
            if cd is not None:
                crawl_delay = float(cd)
        except Exception:
            crawl_delay = None

        entry = RobotsCacheEntry(rp=rp, crawl_delay=crawl_delay, sitemaps=sitemaps, fetched_at=now)
        with self._robots_lock:
            self._robots_cache[host] = entry

        if crawl_delay is not None:
            with self._host_lock:
                self.per_host_delay = max(self.per_host_delay, float(crawl_delay))
        return entry

    def _robots_allows(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        entry = self._get_robots_entry(url)
        if entry is None:
            return True
        try:
            return bool(entry.rp.can_fetch(self.user_agent_token, url))
        except Exception:
            return True

    def _discover_and_seed_sitemaps_for_host(self, any_url_on_host: str):
        if not self.use_sitemaps:
            return
        host = host_of(any_url_on_host)
        with self._robots_lock:
            if host in self._sitemaps_seeded_hosts:
                return
            self._sitemaps_seeded_hosts.add(host)

        entry = self._get_robots_entry(any_url_on_host)
        sitemaps: List[str] = []
        if entry:
            sitemaps.extend(entry.sitemaps)

        if not sitemaps:
            p = urlparse(any_url_on_host)
            sitemaps.append(f"{p.scheme}://{p.netloc}/sitemap.xml")

        seeded = 0
        visited_sitemap_urls: Set[str] = set()

        def enqueue_url(u: str):
            nonlocal seeded
            if seeded >= self.sitemap_max_urls_per_host:
                return
            if not self._allowed_by_regex(u):
                return
            if not self._robots_allows(u):
                return
            if self._is_visited(u):
                return
            if not self._mark_seen(u):
                return
            self._push_frontier(FrontierItem(priority=0.85, url=u, depth=0, discovered_from="sitemap"))
            seeded += 1

        def parse_sitemap(sitemap_url: str, index_depth: int):
            if seeded >= self.sitemap_max_urls_per_host:
                return
            if sitemap_url in visited_sitemap_urls:
                return
            visited_sitemap_urls.add(sitemap_url)

            txt, status, _sec, _reason = self._fetch_text(sitemap_url)
            if txt is None or status >= 400 or status == 0:
                return

            root = _safe_xml_parse(txt)
            if root is None:
                return

            top = _xml_tag_localname(root.tag).lower()
            if top == "urlset":
                for el in root.findall(".//"):
                    if seeded >= self.sitemap_max_urls_per_host:
                        break
                    if _xml_tag_localname(el.tag).lower() == "loc":
                        loc = (el.text or "").strip()
                        if loc.startswith(("http://", "https://")):
                            enqueue_url(loc)
                return

            if top == "sitemapindex" and index_depth < self.sitemap_max_index_depth:
                for el in root.findall(".//"):
                    if _xml_tag_localname(el.tag).lower() == "loc":
                        loc = (el.text or "").strip()
                        if loc.startswith(("http://", "https://")):
                            parse_sitemap(loc, index_depth + 1)

        for sm in sitemaps:
            if seeded >= self.sitemap_max_urls_per_host:
                break
            parse_sitemap(sm, 0)

    # ----------------------------
    # Quantum candidate scoring
    # ----------------------------
    def _feature_state_for_candidate(self, candidate_url: str, anchor_text: str, parent_url: str, parent_depth: int, parent_title: str):
        url_len = len(candidate_url)
        path_len = len(urlparse(candidate_url).path or "/")
        is_same_host = 1.0 if host_of(candidate_url) == host_of(parent_url) else 0.0

        at = (anchor_text or "").strip()
        anchor_len = len(at)
        anchor_has_keyword = 1.0 if any(k in at.lower() for k in ("research", "paper", "docs", "api", "about")) else 0.0

        title = (parent_title or "").strip()
        title_len = len(title)

        depth_penalty = float(parent_depth + 1) / max(1.0, float(self.max_depth))
        novelty = random.random()

        f_url_len = min(1.0, url_len / 200.0)
        f_path_len = min(1.0, path_len / 120.0)
        f_anchor_len = min(1.0, anchor_len / 80.0)
        f_title_len = min(1.0, title_len / 120.0)

        xp = cp if (self.gpu_math and cp is not None) else np

        real = xp.array([
            [1.0 - depth_penalty, is_same_host, f_title_len, novelty],
            [f_url_len, f_path_len, f_anchor_len, anchor_has_keyword],
            [is_same_host, 1.0 - f_url_len, 0.5 * f_anchor_len, 0.5 * novelty],
            [depth_penalty, 0.2 + 0.8 * anchor_has_keyword, 0.3 + 0.7 * f_title_len, 1.0 - novelty],
        ], dtype=xp.float64)

        h = (hash(candidate_url) ^ hash(parent_url)) & 0xFFFFFFFF
        phase = (h % 360) * (math.pi / 180.0)
        imag = (0.05 * math.sin(phase)) * xp.ones((4, 4), dtype=xp.float64)

        state = real.astype(xp.complex128) + 1j * imag.astype(xp.complex128)
        n = xp.linalg.norm(state)
        if float(n) == 0.0:
            return state
        return state / n

    def _score_candidate(self, candidate_url: str, anchor_text: str, parent_url: str, parent_depth: int, parent_title: str) -> float:
        state = self._feature_state_for_candidate(candidate_url, anchor_text, parent_url, parent_depth, parent_title)
        transformed = self.cohesive.decision_function(state)
        p_good = self.qpolicy.score(to_cpu(transformed))
        same_host_bias = 0.05 if host_of(candidate_url) == host_of(parent_url) else 0.0
        depth_bias = 0.03 * (1.0 - (parent_depth / max(1.0, float(self.max_depth))))
        score = float(p_good + same_host_bias + depth_bias)
        return max(0.0, min(1.0, score))

    def _parse(self, html: str, base_url: str) -> Tuple[str, str, List[Tuple[str, str]]]:
        # Truncate to prevent BeautifulSoup from hanging on massive/malformed pages.
        MAX_HTML_SIZE = 1_000_000
        if len(html) > MAX_HTML_SIZE:
            html = html[:MAX_HTML_SIZE]

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except Exception as e:
                if self.debug:
                    logger.debug("[PARSE_ERROR] Skipping unparseable markup at %s: %s", base_url, e)
                return "", "", []

        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        text = soup.get_text(" ", strip=True)
        snippet = text[:4000]

        candidates: List[Tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            u = normalize_url(base_url, a.get("href") or "")
            if u:
                anchor = a.get_text(" ", strip=True)[:200] or "a"
                candidates.append((u, anchor))

        for link in soup.find_all("link", href=True):
            u = normalize_url(base_url, link.get("href") or "")
            if u:
                rel = " ".join(link.get("rel") or []) if link.get("rel") else "link"
                candidates.append((u, (rel or "link")[:200]))

        for meta in soup.find_all("meta"):
            content = meta.get("content")
            if content and ("http://" in content or "https://" in content):
                for u0 in extract_urls_from_text(content):
                    u = normalize_url(base_url, u0)
                    if u:
                        candidates.append((u, "meta"))

        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                data = json.loads(s.get_text() or "")
                blob = json.dumps(data)
                for u0 in extract_urls_from_text(blob):
                    u = normalize_url(base_url, u0)
                    if u:
                        candidates.append((u, "jsonld"))
            except Exception:
                pass

        for s in soup.find_all("script"):
            script_text = s.get_text() or ""
            for u0 in extract_urls_from_text(script_text):
                u = normalize_url(base_url, u0)
                if u:
                    candidates.append((u, "script-url"))
            for pth in extract_paths_from_text(script_text):
                u = normalize_url(base_url, pth)
                if u:
                    candidates.append((u, "script-path"))

        for u0 in extract_urls_from_text(html):
            u = normalize_url(base_url, u0)
            if u:
                candidates.append((u, "raw-html"))

        seen: Set[str] = set()
        links: List[Tuple[str, str]] = []
        for u, label in candidates:
            if u in seen:
                continue
            seen.add(u)
            links.append((u, label))
        return title, snippet, links

    def _write_jsonl(self, obj: Dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self._out_lock:
            with open(self.out_jsonl, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _crawl_one(self, item: FrontierItem) -> Tuple[FrontierItem, int, int]:
        if self._stop_event.is_set():
            return item, 0, 0

        url = item.url
        if item.depth > self.max_depth:
            return item, 0, 0
        if not self._allowed_by_regex(url):
            return item, 0, 0

        if self.respect_robots and (not self._robots_allows(url)):
            self._write_jsonl({
                "url": url, "final_url": None, "depth": item.depth,
                "status": "skip", "reason": "robots_disallow",
                "http_status": 0, "content_type": None, "fetch_seconds": 0.0,
                "fetch_engine": None, "ts": time.time(),
            })
            return item, 0, 0

        self._discover_and_seed_sitemaps_for_host(url)

        html, ct, final_url, status, fetch_s, reason, engine = self._fetch(url)
        if not html or not final_url:
            self._write_jsonl({
                "url": url, "final_url": final_url, "depth": item.depth,
                "status": "skip", "reason": reason,
                "http_status": status, "content_type": ct,
                "fetch_seconds": fetch_s, "fetch_engine": engine,
                "ts": time.time(),
            })
            return item, 0, 0

        title, snippet, links = self._parse(html, final_url)
        found_links = len(links)
        enqueued = 0

        next_depth = item.depth + 1
        if next_depth <= self.max_depth:
            for out_url, anchor in links:
                if not self._allowed_by_regex(out_url):
                    continue
                if self.respect_robots and (not self._robots_allows(out_url)):
                    continue
                if self._is_visited(out_url):
                    continue
                if not self._mark_seen(out_url):
                    continue
                score = self._score_candidate(out_url, anchor, final_url, item.depth, title)
                self._push_frontier(FrontierItem(priority=score, url=out_url, depth=next_depth, discovered_from=final_url))
                enqueued += 1

        self._write_jsonl({
            "url": url, "final_url": final_url, "depth": item.depth,
            "status": "ok", "reason": reason,
            "http_status": status, "content_type": ct,
            "fetch_seconds": fetch_s, "fetch_engine": engine,
            "title": title, "snippet": snippet,
            "out_links_found": found_links, "out_links_enqueued": enqueued,
            "ts": time.time(),
        })

        if self.debug:
            logger.info("[PAGE] depth=%d found_links=%d enqueued=%d engine=%s url=%s", item.depth, found_links, enqueued, engine, final_url)

        self.cohesive.adapt_matrix(performance_threshold=0.8, window=200)
        return item, found_links, enqueued

    def run(self):
        with self._out_lock:
            with open(self.out_jsonl, "w", encoding="utf-8") as f:
                f.write("")

        seeds = load_seeds(self.seeds_path)
        logger.info("Loaded %d seeds from %s", len(seeds), self.seeds_path)

        engine = "curl" if (self.force_curl or self.total_timeout > 0 or self.dns_timeout > 0) else "requests"
        logger.info(
            "FetchEngine=%s | connect=%.2fs read=%.2fs total=%.2fs dns_timeout=%.2fs | retries=%d backoff=%.2fs",
            engine, self.connect_timeout, self.read_timeout, self.total_timeout, self.dns_timeout, self.retries, self.backoff
        )
        if self.dns_timeout > 0:
            logger.info("NOTE: dns_timeout tightens curl --connect-timeout (DNS is inside connect-phase).")

        seeded = 0
        for s in seeds:
            if not self._allowed_by_regex(s):
                continue
            if self.respect_robots and (not self._robots_allows(s)):
                continue
            if self._mark_seen(s):
                self._push_frontier(FrontierItem(priority=1.0, url=s, depth=0, discovered_from="seed"))
                seeded += 1

        if seeded == 0:
            logger.warning("No usable seeds after filtering/robots. Exiting.")
            return

        inflight: Dict = {}
        total_enqueued_links = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            while True:
                while len(inflight) < self.max_workers and not self._stop_event.is_set():
                    nxt = self._pop_frontier()
                    if not nxt:
                        break
                    if not self._mark_visited(nxt.url):
                        continue
                    if not self._try_reserve_page_slot():
                        break
                    fut = ex.submit(self._crawl_one, nxt)
                    inflight[fut] = nxt

                if (self._stop_event.is_set() and not inflight) or (not inflight and self._frontier_size() == 0):
                    break

                if inflight:
                    done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        item = inflight.pop(fut, None)
                        try:
                            _item, _found, enq = fut.result()
                            total_enqueued_links += enq
                            with self._pages_lock:
                                self._pages_crawled += 1
                                pages = self._pages_crawled
                            if pages % 10 == 0 or pages == 1:
                                logger.info(
                                    "Crawled=%d/%d | submitted=%d/%d | inflight=%d | frontier=%d | seen=%d | visited=%d | total_enqueued_links=%d",
                                    pages, self.max_pages, self._submitted, self.max_pages,
                                    len(inflight), self._frontier_size(),
                                    len(self._seen), len(self._visited),
                                    total_enqueued_links
                                )
                        except Exception as e:
                            logger.warning("Worker failed for %s: %s", item.url if item else "unknown", e)

        logger.info(
            "Done. pages_crawled=%d submitted=%d max_pages=%d frontier_remaining=%d seen=%d visited=%d total_enqueued_links=%d cohesive_adaptations=%d",
            self._pages_crawled, self._submitted, self.max_pages,
            self._frontier_size(), len(self._seen), len(self._visited),
            total_enqueued_links, self.cohesive.adaptation_count
        )
        logger.info("Output written to: %s", self.out_jsonl)


# ============================================================
# NEW in v4: Hybrid quantum/probabilistic crawler with
# dead-end backtracking
# ============================================================

class HybridQuantumCrawler(QuantumCrawler):
    """Hybrid quantum/probabilistic crawler with dead-end backtracking (v4).

    Extends QuantumCrawler with a three-signal scoring formula:

        score(url) = quantum_weight  * Q(url)    [quantum gate-based signal]
                   + heuristic_weight * H(url)    [deterministic feature score]
                   + exploration_weight * E(url)  [controlled stochastic noise]

    Weights are normalised so they always sum to 1.0.  The result is a hybrid
    priority that is neither fully greedy (pure quantum) nor fully random.

    Exploration temperature
    -----------------------
    ``exploration_temperature`` (0.0-1.0) controls how much the stochastic
    component diversifies away from deterministic signals.  At 0.0 the noise
    term is zero (fully deterministic); at 1.0 the noise is maximally random.
    Intermediate values blend hash-derived pseudo-randomness (stable per URL)
    with true randomness to produce reproducible-yet-diverse exploration.

    Dead-end backtracking
    ---------------------
    A page is a *dead end* when it yields fewer new enqueueable links than
    ``dead_end_link_threshold``.

    After ``dead_ends_before_boost`` dead ends are detected for pages sharing
    the same parent URL (same branch), all remaining unvisited sibling URLs
    still waiting in the frontier are:
      1. Removed from the "seen" set so they can be re-inserted.
      2. Re-inserted with priority = original_priority + backtrack_boost.

    The original (lower-priority) frontier entries are superseded and silently
    discarded via lazy-deletion: when eventually popped, they find the URL
    already marked visited and are skipped.

    This mechanism biases future selection toward alternative unexplored paths
    in the same branch rather than letting the crawler get stuck in a narrow,
    unproductive subtree.
    """

    def __init__(
        self,
        # ---- hybrid scoring weights (raw; normalised internally) ----
        quantum_weight: float = 0.45,
        heuristic_weight: float = 0.35,
        exploration_weight: float = 0.20,
        # ---- exploration control ----
        exploration_temperature: float = 0.4,
        # ---- dead-end / backtrack parameters ----
        dead_end_link_threshold: int = 2,
        backtrack_boost: float = 0.25,
        dead_ends_before_boost: int = 1,
        **kwargs,
    ):
        """Initialise the hybrid crawler.

        Args:
            quantum_weight:        Weight for the quantum scoring signal.
                                   Higher = crawler relies more on quantum circuit output.
            heuristic_weight:      Weight for the deterministic heuristic signal.
                                   Higher = more stable, feature-driven crawl order.
            exploration_weight:    Weight for the stochastic exploration noise.
                                   Higher = more diverse, less deterministic selection.
            exploration_temperature: Noise randomness scale [0.0, 1.0].
                                   0.0 = no noise; 1.0 = fully random noise component.
            dead_end_link_threshold: Pages enqueuing fewer new links than this are
                                   classified as dead ends and trigger backtracking.
            backtrack_boost:       Priority increase applied to sibling URLs when
                                   a backtrack event fires.
            dead_ends_before_boost: Number of dead-end siblings in a branch before
                                   the backtrack boost is triggered.
            **kwargs:              Forwarded to QuantumCrawler.__init__.
        """
        super().__init__(**kwargs)

        # Normalise weights so they always sum to 1.0.
        _total = quantum_weight + heuristic_weight + exploration_weight
        if _total <= 0:
            _total = 1.0
        self.quantum_weight: float = quantum_weight / _total
        self.heuristic_weight: float = heuristic_weight / _total
        self.exploration_weight: float = exploration_weight / _total

        self.exploration_temperature: float = max(0.0, min(1.0, float(exploration_temperature)))
        self.dead_end_link_threshold: int = max(0, int(dead_end_link_threshold))
        self.backtrack_boost: float = max(0.0, float(backtrack_boost))
        self.dead_ends_before_boost: int = max(1, int(dead_ends_before_boost))

        # Branch registry: parent_url -> list of FrontierItem children pushed to
        # the frontier from that parent.  Enables efficient sibling lookup during
        # backtracking without scanning the entire heap.
        self._branch_frontier: Dict[str, List[FrontierItem]] = {}
        self._branch_lock = threading.Lock()

        # Dead-end counters per branch (keyed by parent URL).
        self._branch_dead_ends: Dict[str, int] = {}
        self._dead_end_lock = threading.Lock()

        # Cumulative statistics surfaced in the final log line.
        self._dead_ends_detected: int = 0
        self._backtrack_boosts_applied: int = 0

        logger.info(
            "[HYBRID] weights: quantum=%.3f heuristic=%.3f exploration=%.3f "
            "temperature=%.2f dead_end_threshold=%d backtrack_boost=%.3f "
            "dead_ends_before_boost=%d",
            self.quantum_weight, self.heuristic_weight, self.exploration_weight,
            self.exploration_temperature, self.dead_end_link_threshold,
            self.backtrack_boost, self.dead_ends_before_boost,
        )

    # ------------------------------------------------------------------
    # Frontier override: track children per branch for backtracking
    # ------------------------------------------------------------------

    def _push_frontier(self, item: FrontierItem):
        """Push item to the priority heap and register it in the branch registry.

        The branch registry maps each parent URL to the list of child
        FrontierItems it produced, enabling ``_apply_backtrack_boost`` to locate
        sibling URLs in O(branch_size) time without scanning the full heap.
        """
        super()._push_frontier(item)
        if item.discovered_from:
            with self._branch_lock:
                if item.discovered_from not in self._branch_frontier:
                    self._branch_frontier[item.discovered_from] = []
                self._branch_frontier[item.discovered_from].append(item)

    # ------------------------------------------------------------------
    # Hybrid scoring helpers
    # ------------------------------------------------------------------

    # Anchor/title keywords that signal a high-value crawl target.
    _ANCHOR_KEYWORD_HINTS: frozenset = frozenset((
        # Documentation / specifications
        "docs", "documentation", "api", "reference", "spec", "specifications",
        "guide", "guides", "tutorial", "tutorials", "manual", "handbook",
        # Research / academia
        "research", "paper", "papers", "publication", "publications",
        "whitepaper", "whitepapers", "report", "reports",
        # Products / company info
        "about", "company", "team", "contact", "careers", "jobs",
        "product", "products", "platform", "pricing", "enterprise",
        # Community / learning
        "blog", "news", "article", "articles", "press", "learn", "learning",
        "resources", "examples", "quickstart", "getting-started",
        "developers", "developer", "community", "forum", "support",
        # Navigation hubs
        "overview", "intro", "introduction", "features", "solutions",
        "changelog", "release", "releases",
    ))

    # URL path segments that typically lead to high-value destinations.
    _IMPORTANT_PATH_SEGS: frozenset = frozenset((
        "docs", "documentation", "doc", "api", "apis", "reference", "ref",
        "spec", "specifications", "specs",
        "research", "papers", "publications", "whitepaper", "whitepapers",
        "about", "company", "team", "contact", "careers", "jobs",
        "blog", "news", "articles", "press", "media",
        "learn", "learning", "tutorial", "tutorials", "guide", "guides",
        "product", "products", "platform", "pricing", "enterprise",
        "developers", "developer", "dev", "community", "forum", "support",
        "resources", "examples", "handbook", "manual",
        "overview", "features", "solutions", "use-cases",
    ))

    # URL path tokens that flag low-value crawl targets.  Matches are applied
    # against the lowercase URL path (not the full URL so query strings don't
    # accidentally trigger this).
    _LOW_VALUE_PATH_TOKENS: frozenset = frozenset((
        "signout", "sign-out", "sign_out", "logout", "log-out", "log_out",
        "cart", "checkout", "basket", "add-to-cart",
        "privacy-policy", "terms-of-service", "terms-and-conditions",
        "cookie-policy", "legal", "dmca",
        "share", "tweet", "facebook", "pinterest",
        "print", "unsubscribe",
    ))

    def _heuristic_score(
        self,
        candidate_url: str,
        anchor_text: str,
        parent_url: str,
        parent_depth: int,
        parent_title: str,
    ) -> float:
        """Deterministic heuristic score using URL and anchor features.

        Combines same-host preference, URL brevity, anchor keyword hints,
        URL path keyword hints, depth penalty, and title length into a
        single [0, 1] value.  Low-value URLs (signout, cart, legal-only
        pages, social-share links, etc.) receive a heavy penalty so the
        crawler naturally avoids them.

        Component weights (before low-value penalty):
            same-host score      0.20
            URL brevity          0.10
            anchor keyword       0.20   (expanded keyword list)
            URL path keyword     0.20   (NEW: path segment matching)
            anchor length        0.05
            depth score          0.15
            title length         0.10
        Low-value multiplier:  0.25 × base score when a low-value token
            is found in the URL path.
        """
        at = (anchor_text or "").strip()
        title = (parent_title or "").strip()
        at_lower = at.lower()

        # Same-host links tend to be more topically relevant.
        same_host_score = 1.0 if host_of(candidate_url) == host_of(parent_url) else 0.4

        # Shorter URLs often point to more canonical/important pages.
        url_score = max(0.0, 1.0 - min(1.0, len(candidate_url) / 200.0))

        # Anchor text keywords strongly suggest a valuable target.
        anchor_keyword = 1.0 if any(k in at_lower for k in self._ANCHOR_KEYWORD_HINTS) else 0.0
        anchor_len_score = min(1.0, len(at) / 60.0) if at else 0.0

        # URL path segment scoring: split path and look for important segments.
        # urlparse is robust against malformed input and rarely raises, but we
        # guard with AttributeError in case of unexpected None returns.
        try:
            url_path = urlparse(candidate_url).path.lower().strip("/")
        except AttributeError:
            url_path = ""
        path_segments = set(url_path.split("/")) if url_path else set()
        path_keyword = 1.0 if path_segments & self._IMPORTANT_PATH_SEGS else 0.0

        # Prefer shallower pages (closer to seed distance).
        depth_score = 1.0 - float(parent_depth) / max(1.0, float(self.max_depth))

        # Non-empty titles typically indicate richer content pages.
        title_score = min(1.0, len(title) / 80.0) if title else 0.0

        # Low-value URL detection: check path for undesirable tokens.
        is_low_value = any(t in url_path for t in self._LOW_VALUE_PATH_TOKENS)

        base_score = (
            0.20 * same_host_score
            + 0.10 * url_score
            + 0.20 * anchor_keyword
            + 0.20 * path_keyword
            + 0.05 * anchor_len_score
            + 0.15 * depth_score
            + 0.10 * title_score
        )
        # Apply a heavy penalty for low-value targets so they sink to the
        # bottom of the priority queue without being completely eliminated.
        if is_low_value:
            base_score *= 0.25
        return max(0.0, min(1.0, base_score))

    def _exploration_noise(self, candidate_url: str) -> float:
        """Controlled stochastic noise scaled by exploration_temperature.

        Blends a hash-derived pseudo-deterministic component (same value every
        time for a given URL, enabling reproducibility) with a truly random
        component (for genuine diversification).  The mixing ratio is determined
        by ``exploration_temperature``:

        - temperature = 0.0  ->  always returns 0.0 (pure determinism)
        - temperature = 0.5  ->  half deterministic, half random, scaled to [0, 0.5]
        - temperature = 1.0  ->  fully random, result in [0, 1]

        Multiplying the blend by ``temperature`` keeps the noise magnitude small
        when temperature is low, so it cannot dominate the quantum/heuristic
        signals at low temperature settings.
        """
        t = self.exploration_temperature
        if t <= 0.0:
            return 0.0
        # Deterministic component derived from the URL hash (stable per URL).
        h = hash(candidate_url) & 0xFFFFFF
        base_noise = (h % 1000) / 1000.0          # in [0, 1)
        # Truly random component for genuine diversification.
        rand_noise = random.random()               # in [0, 1)
        # Blend the two: low t -> more deterministic, high t -> more random.
        mixed = (1.0 - t) * base_noise + t * rand_noise
        # Scale by temperature so noise magnitude is proportional to t.
        return mixed * t

    def _hybrid_score_candidate(
        self,
        candidate_url: str,
        anchor_text: str,
        parent_url: str,
        parent_depth: int,
        parent_title: str,
        branch_dead_ends: int = 0,
    ) -> float:
        """Hybrid priority score combining quantum, heuristic, and exploration signals.

        Score = quantum_weight * Q(url) + heuristic_weight * H(url)
              + exploration_weight * E(url)

        All three components are in [0, 1] and the weights are normalised to
        sum to 1.0, so the result is bounded in [0, 1].

        Dead-end diversity bonus
        ------------------------
        When ``branch_dead_ends >= dead_ends_before_boost``, a small diversity
        bonus (capped at 0.15) is added to encourage exploration of the remaining
        URLs in a branch that has already produced dead ends.  This complements
        the explicit backtrack boost (which re-prioritises siblings already in the
        frontier) by also raising the scores of *newly discovered* candidates from
        the same grandparent context.

        Args:
            candidate_url:    URL being scored.
            anchor_text:      Link anchor text.
            parent_url:       URL of the page that contained this link.
            parent_depth:     Crawl depth of the parent page.
            parent_title:     HTML title of the parent page.
            branch_dead_ends: Dead-end count for the branch parent_url belongs to.

        Returns:
            Combined score in [0.0, 1.0].
        """
        # --- Quantum signal (parameterised circuit measurement probability) ---
        q_score = 0.5  # safe fallback; replaced below unless circuit evaluation fails
        try:
            state = self._feature_state_for_candidate(
                candidate_url, anchor_text, parent_url, parent_depth, parent_title
            )
            transformed = self.cohesive.decision_function(state)
            q_score = self.qpolicy.score(to_cpu(transformed))
        except Exception:
            pass  # keep 0.5 fallback so scoring degrades gracefully

        # --- Deterministic heuristic signal ---
        h_score = self._heuristic_score(
            candidate_url, anchor_text, parent_url, parent_depth, parent_title
        )

        # --- Controlled stochastic exploration noise ---
        e_noise = self._exploration_noise(candidate_url)

        # --- Weighted combination ---
        combined = (
            self.quantum_weight * q_score
            + self.heuristic_weight * h_score
            + self.exploration_weight * e_noise
        )

        # --- Dead-end diversity bonus ---
        # After enough sibling dead ends in the same branch, nudge remaining
        # candidates from that context upward to favour alternative paths.
        if branch_dead_ends >= self.dead_ends_before_boost:
            diversity_bonus = min(0.15, self.backtrack_boost * 0.5 * branch_dead_ends)
            combined = min(1.0, combined + diversity_bonus)

        return max(0.0, min(1.0, combined))

    # ------------------------------------------------------------------
    # Dead-end detection and backtracking
    # ------------------------------------------------------------------

    def _apply_backtrack_boost(self, dead_end_url: str, parent_url: str):
        """Boost sibling URLs in the frontier after a dead end is detected.

        Locates all unvisited siblings of ``dead_end_url`` (URLs discovered from
        the same ``parent_url``) that are currently waiting in the priority heap,
        removes their "seen" mark, and re-inserts them with priority increased by
        ``backtrack_boost``.

        The old (lower-priority) heap entries are superseded and will be silently
        discarded when eventually popped: the run loop calls ``_mark_visited``
        before processing, so a URL that has already been handled is skipped.

        This is the core of the backtracking mechanism: rather than continuing
        to follow a branch that has proven unproductive, the crawler is guided
        toward the best-scoring alternative URLs from the same parent.

        Args:
            dead_end_url: URL that was classified as a dead end.
            parent_url:   Parent URL from which ``dead_end_url`` was discovered.
        """
        if not parent_url:
            return

        with self._branch_lock:
            siblings = list(self._branch_frontier.get(parent_url, []))

        boosted = 0
        for sib_item in siblings:
            if sib_item.url == dead_end_url:
                continue  # skip the dead-end page itself
            if self._is_visited(sib_item.url):
                continue  # already crawled; nothing to boost

            # Clear the "seen" mark so the URL can be re-queued.
            with self._seen_lock:
                self._seen.discard(sib_item.url)

            # Re-insert with boosted priority.  The old heap entry will be
            # lazily discarded (visited check) when it is eventually popped.
            boosted_priority = min(1.0, sib_item.priority + self.backtrack_boost)
            boosted_item = FrontierItem(
                priority=boosted_priority,
                url=sib_item.url,
                depth=sib_item.depth,
                discovered_from=sib_item.discovered_from,
            )
            self._push_frontier(boosted_item)
            boosted += 1

        if boosted > 0:
            self._backtrack_boosts_applied += boosted
            if self.debug:
                logger.debug(
                    "[BACKTRACK] dead_end=%s parent=%s boosted_siblings=%d",
                    dead_end_url, parent_url, boosted,
                )

    def _handle_dead_end(self, url: str, parent_url: str, enqueued: int):
        """Register a dead-end page and trigger backtracking when the threshold is met.

        A page qualifies as a dead end when it enqueues fewer new links than
        ``dead_end_link_threshold``.  After ``dead_ends_before_boost`` such
        events accumulate for the same ``parent_url`` branch,
        ``_apply_backtrack_boost`` is called to re-prioritise its siblings.

        Args:
            url:        URL of the page classified as a dead end.
            parent_url: URL from which ``url`` was discovered (its branch key).
            enqueued:   Number of new links the dead-end page enqueued.
        """
        if not parent_url:
            return

        with self._dead_end_lock:
            self._branch_dead_ends[parent_url] = (
                self._branch_dead_ends.get(parent_url, 0) + 1
            )
            count = self._branch_dead_ends[parent_url]
            self._dead_ends_detected += 1

        if self.debug:
            logger.debug(
                "[DEAD-END] url=%s parent=%s enqueued=%d branch_dead_ends=%d",
                url, parent_url, enqueued, count,
            )

        if count >= self.dead_ends_before_boost:
            self._apply_backtrack_boost(url, parent_url)

    # ------------------------------------------------------------------
    # Core crawl loop override
    # ------------------------------------------------------------------

    def _crawl_one(self, item: FrontierItem) -> Tuple[FrontierItem, int, int]:
        """Crawl one URL using hybrid scoring and dead-end backtracking.

        Overrides QuantumCrawler._crawl_one with two key changes:

        1. **Hybrid scoring**: Link candidates are scored with
           ``_hybrid_score_candidate`` (quantum + heuristic + noise) instead of
           the pure quantum ``_score_candidate`` from v3.

        2. **Dead-end detection**: After crawling, if the page enqueued fewer
           than ``dead_end_link_threshold`` new links it is registered as a dead
           end via ``_handle_dead_end``, which may trigger a backtrack boost for
           its sibling URLs in the frontier.

        The JSONL output record includes ``hybrid_branch_dead_ends`` so analysts
        can correlate individual decisions with the backtracking state.
        """
        if self._stop_event.is_set():
            return item, 0, 0

        url = item.url
        if item.depth > self.max_depth:
            return item, 0, 0
        if not self._allowed_by_regex(url):
            return item, 0, 0

        if self.respect_robots and (not self._robots_allows(url)):
            self._write_jsonl({
                "url": url, "final_url": None, "depth": item.depth,
                "status": "skip", "reason": "robots_disallow",
                "http_status": 0, "content_type": None, "fetch_seconds": 0.0,
                "fetch_engine": None, "ts": time.time(),
            })
            return item, 0, 0

        self._discover_and_seed_sitemaps_for_host(url)

        html, ct, final_url, status, fetch_s, reason, engine = self._fetch(url)
        if not html or not final_url:
            self._write_jsonl({
                "url": url, "final_url": final_url, "depth": item.depth,
                "status": "skip", "reason": reason,
                "http_status": status, "content_type": ct,
                "fetch_seconds": fetch_s, "fetch_engine": engine,
                "ts": time.time(),
            })
            # Fetch failures count as dead ends (0 new links) for backtracking.
            self._handle_dead_end(url, item.discovered_from, 0)
            return item, 0, 0

        title, snippet, links = self._parse(html, final_url)
        found_links = len(links)
        enqueued = 0

        # Look up current dead-end count for this URL's branch.  Passed into
        # the hybrid scorer to apply the diversity bonus where appropriate.
        with self._dead_end_lock:
            branch_dead_ends = self._branch_dead_ends.get(item.discovered_from or "", 0)

        next_depth = item.depth + 1
        if next_depth <= self.max_depth:
            for out_url, anchor in links:
                if not self._allowed_by_regex(out_url):
                    continue
                if self.respect_robots and (not self._robots_allows(out_url)):
                    continue
                if self._is_visited(out_url):
                    continue
                if not self._mark_seen(out_url):
                    continue
                # Hybrid scorer: replaces the pure quantum scorer from v3.
                score = self._hybrid_score_candidate(
                    out_url, anchor, final_url, item.depth, title,
                    branch_dead_ends=branch_dead_ends,
                )
                self._push_frontier(
                    FrontierItem(
                        priority=score,
                        url=out_url,
                        depth=next_depth,
                        discovered_from=final_url,
                    )
                )
                enqueued += 1

        self._write_jsonl({
            "url": url, "final_url": final_url, "depth": item.depth,
            "status": "ok", "reason": reason,
            "http_status": status, "content_type": ct,
            "fetch_seconds": fetch_s, "fetch_engine": engine,
            "title": title, "snippet": snippet,
            "out_links_found": found_links, "out_links_enqueued": enqueued,
            "ts": time.time(),
            # Extra field: dead-end count for this page's branch at crawl time.
            "hybrid_branch_dead_ends": branch_dead_ends,
        })

        if self.debug:
            logger.info(
                "[HYBRID PAGE] depth=%d found=%d enqueued=%d engine=%s "
                "branch_dead_ends=%d url=%s",
                item.depth, found_links, enqueued, engine, branch_dead_ends, final_url,
            )

        # Dead-end classification: trigger backtracking when links are scarce.
        if enqueued < self.dead_end_link_threshold:
            self._handle_dead_end(final_url, item.discovered_from, enqueued)

        self.cohesive.adapt_matrix(performance_threshold=0.8, window=200)
        return item, found_links, enqueued

    def run(self):
        """Run the hybrid crawler and report backtracking statistics on completion."""
        super().run()
        logger.info(
            "[HYBRID] Dead-ends detected: %d | Backtrack boosts applied: %d",
            self._dead_ends_detected, self._backtrack_boosts_applied,
        )


# ----------------------------
# Single-threaded comparable crawler (for --compare-algorithms mode)
# ----------------------------
_COMPARE_ALGORITHMS = ("quantum", "dfs", "bfs")
_DEFAULT_NON_QUANTUM_PRIORITY = 0.5


class _SingleThreadComparableCrawler(HybridQuantumCrawler):
    """Single-threaded variant used exclusively for --compare-algorithms mode.

    Overrides frontier management to support quantum (priority queue), dfs (LIFO
    stack), and bfs (FIFO queue) traversal strategies.  Instead of writing JSONL
    records to disk, it accumulates them in _cmp_records so the comparison runner
    can write unified CSV/JSON output.

    Inherits from HybridQuantumCrawler so that the quantum mode uses the full
    hybrid scoring pipeline (_hybrid_score_candidate: quantum + heuristic +
    exploration noise) rather than the bare quantum signal from the base class.
    This ensures quantum ordering is meaningfully different from BFS/DFS.
    """

    def __init__(self, algorithm: str, **kwargs):
        if algorithm not in _COMPARE_ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algorithm!r}; expected one of {_COMPARE_ALGORITHMS}")
        kwargs["max_workers"] = 1
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self._dfs_stack: List[FrontierItem] = []
        self._bfs_deque: deque = deque()
        self._cmp_records: List[dict] = []
        self._cmp_step: int = 0

    def _push_frontier(self, item: FrontierItem):
        if self.algorithm == "dfs":
            with self._frontier_lock:
                self._dfs_stack.append(item)
        elif self.algorithm == "bfs":
            with self._frontier_lock:
                self._bfs_deque.append(item)
        else:
            super()._push_frontier(item)

    def _pop_frontier(self) -> Optional[FrontierItem]:
        if self.algorithm == "dfs":
            with self._frontier_lock:
                return self._dfs_stack.pop() if self._dfs_stack else None
        elif self.algorithm == "bfs":
            with self._frontier_lock:
                return self._bfs_deque.popleft() if self._bfs_deque else None
        else:
            return super()._pop_frontier()

    def _frontier_size(self) -> int:
        if self.algorithm == "dfs":
            with self._frontier_lock:
                return len(self._dfs_stack)
        elif self.algorithm == "bfs":
            with self._frontier_lock:
                return len(self._bfs_deque)
        else:
            return super()._frontier_size()

    def _write_jsonl(self, obj: dict):
        """Intercept record writes: tag with algorithm/step and store in memory."""
        obj["algorithm"] = self.algorithm
        obj["step"] = self._cmp_step
        self._cmp_step += 1
        self._cmp_records.append(obj)

    def _crawl_one(self, item: FrontierItem) -> Tuple[FrontierItem, int, int]:
        if self._stop_event.is_set():
            return item, 0, 0

        url = item.url
        if item.depth > self.max_depth:
            return item, 0, 0
        if not self._allowed_by_regex(url):
            return item, 0, 0

        priority_score: Optional[float] = item.priority if self.algorithm == "quantum" else None

        if self.respect_robots and not self._robots_allows(url):
            self._write_jsonl({
                "url": url, "final_url": None, "depth": item.depth,
                "status": "skip", "reason": "robots_disallow",
                "http_status": 0, "content_type": None, "fetch_seconds": 0.0,
                "fetch_engine": None, "ts": time.time(),
                "discovered_from": item.discovered_from,
                "priority_score": priority_score,
            })
            return item, 0, 0

        self._discover_and_seed_sitemaps_for_host(url)

        html, ct, final_url, status, fetch_s, reason, engine = self._fetch(url)
        if not html or not final_url:
            self._write_jsonl({
                "url": url, "final_url": final_url, "depth": item.depth,
                "status": "skip", "reason": reason,
                "http_status": status, "content_type": ct,
                "fetch_seconds": fetch_s, "fetch_engine": engine,
                "ts": time.time(),
                "discovered_from": item.discovered_from,
                "priority_score": priority_score,
            })
            return item, 0, 0

        title, snippet, links = self._parse(html, final_url)
        found_links = len(links)
        enqueued = 0

        next_depth = item.depth + 1
        if next_depth <= self.max_depth:
            for out_url, anchor in links:
                if not self._allowed_by_regex(out_url):
                    continue
                if self.respect_robots and not self._robots_allows(out_url):
                    continue
                if self._is_visited(out_url):
                    continue
                if not self._mark_seen(out_url):
                    continue
                if self.algorithm == "quantum":
                    # Use the full hybrid scorer so quantum mode makes
                    # meaningfully differentiated priority decisions rather
                    # than collapsing to near-BFS behaviour from weak signals.
                    child_priority = self._hybrid_score_candidate(
                        out_url, anchor, final_url, item.depth, title
                    )
                else:
                    child_priority = _DEFAULT_NON_QUANTUM_PRIORITY
                self._push_frontier(
                    FrontierItem(
                        priority=child_priority,
                        url=out_url,
                        depth=next_depth,
                        discovered_from=final_url,
                    )
                )
                enqueued += 1

        self._write_jsonl({
            "url": url, "final_url": final_url, "depth": item.depth,
            "status": "ok", "reason": reason,
            "http_status": status, "content_type": ct,
            "fetch_seconds": fetch_s, "fetch_engine": engine,
            "title": title, "snippet": snippet,
            "out_links_found": found_links, "out_links_enqueued": enqueued,
            "ts": time.time(),
            "discovered_from": item.discovered_from,
            "priority_score": priority_score,
        })

        if self.debug:
            logger.info(
                "[%s] depth=%d found_links=%d enqueued=%d engine=%s url=%s",
                self.algorithm.upper(), item.depth, found_links, enqueued, engine, final_url,
            )

        if self.algorithm == "quantum":
            self.cohesive.adapt_matrix(performance_threshold=0.8, window=200)

        return item, found_links, enqueued

    def run_timed(self, seeds: List[str], duration_s: float) -> List[dict]:
        """Single-threaded time-bounded crawl returning per-step records."""
        deadline = time.time() + duration_s
        logger.info("[COMPARE] Starting algorithm=%s duration=%.0fs", self.algorithm, duration_s)

        self._cmp_records = []
        self._cmp_step = 0

        seeded = 0
        for s in seeds:
            if not self._allowed_by_regex(s):
                continue
            if self.respect_robots and not self._robots_allows(s):
                continue
            if self._mark_seen(s):
                self._push_frontier(FrontierItem(priority=1.0, url=s, depth=0, discovered_from="seed"))
                seeded += 1

        if seeded == 0:
            logger.warning("[COMPARE] No usable seeds for algorithm=%s", self.algorithm)
            return self._cmp_records

        logger.info("[COMPARE] Seeded %d URL(s) for algorithm=%s", seeded, self.algorithm)

        pages = 0
        while time.time() < deadline:
            item = self._pop_frontier()
            if item is None:
                logger.info(
                    "[COMPARE] Frontier empty for algorithm=%s after %d page(s)",
                    self.algorithm, pages,
                )
                break
            if not self._mark_visited(item.url):
                continue
            logger.debug(
                "[COMPARE] %s step=%d depth=%d url=%s (from=%s)",
                self.algorithm, self._cmp_step, item.depth, item.url, item.discovered_from,
            )
            self._crawl_one(item)
            pages += 1
            if pages % 10 == 0:
                remaining = max(0.0, deadline - time.time())
                logger.info(
                    "[COMPARE] %s: pages=%d frontier=%d remaining=%.1fs",
                    self.algorithm, pages, self._frontier_size(), remaining,
                )

        elapsed = duration_s - max(0.0, deadline - time.time())
        logger.info(
            "[COMPARE] Done. algorithm=%s pages=%d records=%d elapsed=%.1fs",
            self.algorithm, pages, len(self._cmp_records), elapsed,
        )
        return self._cmp_records


# ----------------------------
# Comparison-mode orchestrator
# ----------------------------

_CSV_FIELDS = [
    "algorithm",
    "step",
    "timestamp",
    "source_url",
    "visited_url",
    "final_url",
    "depth",
    "fetch_outcome",
    "fetch_reason",
    "http_status",
    "content_type",
    "fetch_duration_s",
    "links_found",
    "links_enqueued",
    "priority_score",
    "title",
]


def _record_to_csv_row(r: dict) -> dict:
    """Map an internal record dict to the flat CSV row format."""
    return {
        "algorithm": r.get("algorithm", ""),
        "step": r.get("step", ""),
        "timestamp": r.get("ts", ""),
        "source_url": r.get("discovered_from", ""),
        "visited_url": r.get("url", ""),
        "final_url": r.get("final_url", "") or "",
        "depth": r.get("depth", ""),
        "fetch_outcome": r.get("status", ""),
        "fetch_reason": r.get("reason", ""),
        "http_status": r.get("http_status", ""),
        "content_type": r.get("content_type", "") or "",
        "fetch_duration_s": r.get("fetch_seconds", ""),
        "links_found": r.get("out_links_found", ""),
        "links_enqueued": r.get("out_links_enqueued", ""),
        "priority_score": (lambda ps: "" if ps is None else ps)(r.get("priority_score")),
        "title": r.get("title", "") or "",
    }


def run_comparison_mode(
    seeds: List[str],
    duration_per_algo_s: float,
    out_csv: str,
    out_json: str,
    crawler_kwargs: dict,
) -> None:
    """Orchestrate a three-way quantum / DFS / BFS comparison.

    Runs each algorithm sequentially for duration_per_algo_s seconds from the
    same seeds list.  After all three runs writes:
    - out_csv: one row per URL visit attempt across all three algorithms.
    - out_json: full records plus a per-algorithm aggregate summary.
    """
    run_start_ts = time.time()
    all_records: List[dict] = []
    summaries: Dict[str, dict] = {}

    for algo in _COMPARE_ALGORITHMS:
        logger.info("[COMPARE] === Starting %s crawl (%.0f seconds) ===", algo, duration_per_algo_s)
        crawler = _SingleThreadComparableCrawler(algorithm=algo, **crawler_kwargs)
        records = crawler.run_timed(seeds, duration_per_algo_s)
        all_records.extend(records)

        visited_recs = [r for r in records if r.get("status") == "ok"]
        skipped_recs = [r for r in records if r.get("status") == "skip"]
        fetch_times = [r["fetch_seconds"] for r in visited_recs if "fetch_seconds" in r]
        priority_scores = [
            r["priority_score"] for r in visited_recs if r.get("priority_score") is not None
        ]
        depths = [r.get("depth", 0) for r in visited_recs]

        summaries[algo] = {
            "algorithm": algo,
            "total_records": len(records),
            "pages_visited": len(visited_recs),
            "pages_skipped": len(skipped_recs),
            "avg_fetch_seconds": float(np.mean(fetch_times)) if fetch_times else 0.0,
            "max_fetch_seconds": float(max(fetch_times)) if fetch_times else 0.0,
            "total_links_found": sum(r.get("out_links_found", 0) for r in visited_recs),
            "total_links_enqueued": sum(r.get("out_links_enqueued", 0) for r in visited_recs),
            "unique_depths_reached": sorted(set(depths)),
            "max_depth_reached": max(depths) if depths else 0,
            "avg_priority_score": float(np.mean(priority_scores)) if priority_scores else None,
            "min_priority_score": float(min(priority_scores)) if priority_scores else None,
            "max_priority_score": float(max(priority_scores)) if priority_scores else None,
        }

        logger.info("[COMPARE] %s summary: %s", algo, json.dumps(summaries[algo]))

    # ---- Write CSV ----
    if all_records:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in all_records:
                writer.writerow(_record_to_csv_row(r))
        logger.info("[COMPARE] CSV written to: %s (%d rows)", out_csv, len(all_records))

    # ---- Write JSON ----
    output = {
        "comparison_run": {
            "run_start_timestamp": run_start_ts,
            "run_end_timestamp": time.time(),
            "duration_per_algo_s": duration_per_algo_s,
            "algorithms_run": list(_COMPARE_ALGORITHMS),
            "seeds": seeds,
        },
        "summaries": summaries,
        "records": all_records,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("[COMPARE] JSON written to: %s", out_json)

    # ---- Human-readable summary table ----
    logger.info("[COMPARE] === Final Comparison Summary ===")
    header = (
        f"{'Algorithm':<10} {'Visited':>8} {'Skipped':>8} "
        f"{'AvgFetch':>10} {'LinksFound':>12} {'MaxDepth':>10} "
        f"{'AvgPriority':>13} {'MinPriority':>13} {'MaxPriority':>13}"
    )
    logger.info("[COMPARE] %s", header)
    logger.info("[COMPARE] %s", "-" * len(header))
    for algo in _COMPARE_ALGORITHMS:
        s = summaries[algo]
        avg_p = s.get("avg_priority_score")
        min_p = s.get("min_priority_score")
        max_p = s.get("max_priority_score")
        avg_p_str = f"{avg_p:>13.4f}" if avg_p is not None else f"{'N/A':>13}"
        min_p_str = f"{min_p:>13.4f}" if min_p is not None else f"{'N/A':>13}"
        max_p_str = f"{max_p:>13.4f}" if max_p is not None else f"{'N/A':>13}"
        logger.info(
            "[COMPARE] %-10s %8d %8d %10.2fs %12d %10d %13s %13s %13s",
            algo,
            s["pages_visited"],
            s["pages_skipped"],
            s["avg_fetch_seconds"],
            s["total_links_found"],
            s["max_depth_reached"],
            avg_p_str,
            min_p_str,
            max_p_str,
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Quantum Decision Crawler v4 -- hybrid quantum/probabilistic crawl strategy "
            "with dead-end backtracking (extends v3)."
        )
    )

    # ---- Standard crawl parameters (same as v3) ----
    p.add_argument("--seeds", default="seeds.txt")
    p.add_argument("--out", default="crawl.jsonl")
    p.add_argument("--max-pages", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--delay", type=float, default=0.6)
    p.add_argument("--allow", default=None)
    p.add_argument("--deny", default=None)
    p.add_argument("--shots", type=int, default=128)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--gpu-math", action="store_true")
    p.add_argument("--no-aer-gpu", action="store_true")

    p.add_argument("--respect-robots", action="store_true")
    p.add_argument("--no-respect-robots", action="store_true")
    p.add_argument("--use-sitemaps", action="store_true")
    p.add_argument("--no-use-sitemaps", action="store_true")
    p.add_argument("--robots-ttl", type=int, default=21600)
    p.add_argument("--sitemap-max-urls", type=int, default=5000)
    p.add_argument("--sitemap-index-depth", type=int, default=2)

    p.add_argument("--connect-timeout", type=float, default=5.0)
    p.add_argument("--read-timeout", type=float, default=25.0)
    p.add_argument(
        "--total-timeout", type=float, default=12.0,
        help="Hard wall-clock max time (curl --max-time).",
    )
    p.add_argument(
        "--dns-timeout", type=float, default=4.0,
        help="Caps curl connect-phase time (DNS is inside it).",
    )
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--backoff", type=float, default=0.6)
    p.add_argument("--force-curl", action="store_true", help="Always use curl engine.")

    # ---- NEW in v4: hybrid mode parameters ----
    hybrid_group = p.add_argument_group(
        "hybrid mode",
        description=(
            "Controls the three-signal hybrid scoring strategy introduced in v4. "
            "Weights are automatically normalised to sum to 1.0 before use."
        ),
    )
    hybrid_group.add_argument(
        "--quantum-weight", type=float, default=0.45,
        help=(
            "Weight for the quantum gate-based scoring signal. "
            "Higher values make the crawler rely more on quantum circuit output. "
            "(default: 0.45)"
        ),
    )
    hybrid_group.add_argument(
        "--heuristic-weight", type=float, default=0.35,
        help=(
            "Weight for the deterministic heuristic signal (URL length, anchor "
            "keywords, depth, same-host preference). Higher values make the crawl "
            "more deterministic and feature-driven. (default: 0.35)"
        ),
    )
    hybrid_group.add_argument(
        "--exploration-weight", type=float, default=0.20,
        help=(
            "Weight for the stochastic exploration noise component. Higher values "
            "introduce more randomness to diversify link selection. (default: 0.20)"
        ),
    )
    hybrid_group.add_argument(
        "--exploration-temperature", type=float, default=0.4,
        help=(
            "Controls how random the exploration noise is. 0.0 = no noise (fully "
            "deterministic); 1.0 = maximum noise (fully random within the exploration "
            "component). Does not affect quantum or heuristic components. (default: 0.4)"
        ),
    )
    hybrid_group.add_argument(
        "--dead-end-threshold", type=int, default=2,
        help=(
            "Minimum number of new links a page must enqueue to avoid being classified "
            "as a dead end. Pages yielding fewer links may trigger backtracking. "
            "(default: 2)"
        ),
    )
    hybrid_group.add_argument(
        "--backtrack-boost", type=float, default=0.25,
        help=(
            "Priority bonus added to sibling URLs when a dead-end backtrack event "
            "fires. Higher values make the crawler jump to alternative branches more "
            "aggressively. (default: 0.25)"
        ),
    )
    hybrid_group.add_argument(
        "--dead-ends-before-boost", type=int, default=1,
        help=(
            "Number of dead-end siblings within a branch before the backtrack boost "
            "is applied to remaining siblings. Higher values delay the boost until "
            "more evidence of a poor branch has accumulated. (default: 1)"
        ),
    )
    hybrid_group.add_argument(
        "--use-base-crawler", action="store_true",
        help=(
            "Use the plain QuantumCrawler (v3 behaviour) instead of HybridQuantumCrawler. "
            "Useful as a control baseline when benchmarking the hybrid strategy."
        ),
    )

    # ---- Comparison-mode flags (same as v3) ----
    p.add_argument(
        "--compare-algorithms", action="store_true",
        help=(
            "Run quantum, DFS, and BFS crawling strategies sequentially from the "
            "same seeds.txt input and produce unified CSV/JSON comparison artifacts. "
            "Each strategy runs for --compare-duration seconds (default 180 s = 3 min)."
        ),
    )
    p.add_argument(
        "--compare-duration", type=float, default=180.0,
        help="Wall-clock seconds allocated to each algorithm in comparison mode (default: 180).",
    )
    p.add_argument(
        "--compare-csv", default="comparison.csv",
        help="Output CSV file for comparison mode (default: comparison.csv).",
    )
    p.add_argument(
        "--compare-json", default="comparison.json",
        help="Output JSON file for comparison mode (default: comparison.json).",
    )

    args = p.parse_args()

    respect_robots = True
    if args.no_respect_robots:
        respect_robots = False
    elif args.respect_robots:
        respect_robots = True

    use_sitemaps = True
    if args.no_use_sitemaps:
        use_sitemaps = False
    elif args.use_sitemaps:
        use_sitemaps = True

    # Shared kwargs forwarded to both normal and comparison modes.
    _crawler_kwargs = dict(
        seeds_path=args.seeds,
        out_jsonl=args.out,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        max_workers=args.workers,
        per_host_delay=args.delay,
        allow_regex=args.allow,
        deny_regex=args.deny,
        qiskit_shots=args.shots,
        debug=args.debug,
        gpu_math=args.gpu_math,
        aer_request_gpu=(not args.no_aer_gpu),
        respect_robots=respect_robots,
        use_sitemaps=use_sitemaps,
        robots_cache_ttl_s=args.robots_ttl,
        sitemap_max_urls_per_host=args.sitemap_max_urls,
        sitemap_max_index_depth=args.sitemap_index_depth,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        total_timeout=args.total_timeout,
        retries=args.retries,
        backoff=args.backoff,
        force_curl=args.force_curl,
        dns_timeout=args.dns_timeout,
    )

    if args.compare_algorithms:
        seeds = load_seeds(args.seeds)
        # Merge hybrid scoring parameters so the quantum algorithm in
        # comparison mode benefits from the same tuned weights used in
        # normal HybridQuantumCrawler mode.
        _compare_kwargs = dict(
            **_crawler_kwargs,
            quantum_weight=args.quantum_weight,
            heuristic_weight=args.heuristic_weight,
            exploration_weight=args.exploration_weight,
            exploration_temperature=args.exploration_temperature,
            dead_end_link_threshold=args.dead_end_threshold,
            backtrack_boost=args.backtrack_boost,
            dead_ends_before_boost=args.dead_ends_before_boost,
        )
        run_comparison_mode(
            seeds=seeds,
            duration_per_algo_s=args.compare_duration,
            out_csv=args.compare_csv,
            out_json=args.compare_json,
            crawler_kwargs=_compare_kwargs,
        )
    elif args.use_base_crawler:
        # Run without hybrid features -- useful as a v3 baseline comparison.
        logger.info("[v4] Using base QuantumCrawler (v3 behaviour) via --use-base-crawler.")
        crawler = QuantumCrawler(**_crawler_kwargs)
        crawler.run()
    else:
        # Default: run the new HybridQuantumCrawler.
        hybrid_kwargs = dict(
            quantum_weight=args.quantum_weight,
            heuristic_weight=args.heuristic_weight,
            exploration_weight=args.exploration_weight,
            exploration_temperature=args.exploration_temperature,
            dead_end_link_threshold=args.dead_end_threshold,
            backtrack_boost=args.backtrack_boost,
            dead_ends_before_boost=args.dead_ends_before_boost,
        )
        crawler = HybridQuantumCrawler(**_crawler_kwargs, **hybrid_kwargs)
        crawler.run()
