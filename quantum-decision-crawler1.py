#!/usr/bin/env python3
"""
Quantum Decision Crawler (recursive) — BS4 + parallel + seeds.txt + Qiskit + GPU-compatible scoring.

(C)Tsubasa Kato - Inspire Search Corp.
- Tested on uv environment on Ubuntu 22.04 and CUDA 13.1
- Created with help of Perplexity and ChatGPT GPT5.2 Thinking

Key features
- Starts from seeds.txt (one URL per line)
- Recursively follows links up to --max-depth
- Parallel crawling with --workers
- Quantum decision scoring (Qiskit circuit) to prioritize frontier
- GPU-compatible math path (CuPy) for feature-state + cohesive transform (optional)
- HARD respects --max-pages (caps *submitted* tasks; stops workers from writing beyond limit)

Run:
  uv run test-final.py --seeds seeds.txt --max-pages 200 --max-depth 3 --workers 16 --shots 64 --gpu-math --debug
"""

import os
import re
import json
import time
import math
import heapq
import random
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# ---- Qiskit imports ----
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Statevector

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("quantum_crawler")

# ----------------------------
# Optional GPU backend (CuPy) — strict diagnostics
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
    """Convert cupy array to numpy; pass-through numpy."""
    if cp is not None and _CUPY_OK and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


# ----------------------------
# URL extraction helpers for SPA/JS-heavy pages
# ----------------------------
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
PATH_RE = re.compile(
    r"(?<![a-zA-Z0-9])/(?:[a-zA-Z0-9\-\._~%!$&'()*+,;=:@/]+)",
    re.IGNORECASE,
)

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
# Qiskit-backed quantum decision policy (SamplerV2 + fallback)
# ----------------------------
class QuantumDecisionPolicy:
    """
    4x4 complex -> pooled row mag/phase -> theta[4], phi[4] -> param circuit -> score ~ P(goodness=1).

    Uses qiskit-aer SamplerV2 if available. Tries to request GPU device best-effort.
    Otherwise falls back to Statevector exact probability.
    """

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

            # Best-effort: if Aer GPU build exists, this may enable it. If not, Aer runs CPU.
            if self.request_gpu:
                backend_options["device"] = "GPU"

            backend_options["method"] = "statevector"

            self._aer_sampler = AerSampler(options=dict(backend_options=backend_options))
            logger.info("[Qiskit] Using Aer SamplerV2 (GPU requested=%s).", self.request_gpu)
        except Exception as e:
            self._aer_sampler = None
            logger.info("[Qiskit] Aer SamplerV2 not available; using Statevector fallback. (%s)", type(e).__name__)

    def _build_param_circuit(self) -> QuantumCircuit:
        q = QuantumRegister(5, "q")  # 4 feature + 1 goodness
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
        p1 = float(probs[1::2].sum())  # last qubit == 1
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
# Cohesive 4D transform (GPU-capable)
# ----------------------------
class CohesiveDiagonalMatrix4D:
    """
    Stored on CPU as numpy; a GPU copy is created on demand.
    """

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.matrix_cpu = self._initialize_cohesive_matrix_np()
        self.matrix_gpu = None  # cupy array if GPU enabled
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
        """
        input_state: (4,4) complex array; numpy or cupy.
        returns: same type module as input_state (cupy if GPU, else numpy)
        """
        start = time.time()

        if input_state.shape != (self.dimension, self.dimension):
            input_state = input_state.reshape(self.dimension, self.dimension)

        # CPU path
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

        # GPU path
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


# ----------------------------
# Frontier item
# ----------------------------
@dataclass(order=True)
class FrontierItem:
    priority: float
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)
    discovered_from: str = field(compare=False, default="")


# ----------------------------
# Helpers
# ----------------------------
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


# ----------------------------
# Quantum crawler (GPU-capable scoring) + HARD max-pages enforcement
# ----------------------------
class QuantumCrawler:
    def __init__(
        self,
        seeds_path: str = "seeds.txt",
        out_jsonl: str = "crawl.jsonl",
        max_pages: int = 500,
        max_depth: int = 3,
        max_workers: int = 16,
        request_timeout: int = 10,
        per_host_delay: float = 0.6,
        user_agent: str = "QuantumCrawler/1.0 (+https://example.invalid)",
        allow_regex: Optional[str] = None,
        deny_regex: Optional[str] = None,
        qiskit_shots: int = 128,
        debug: bool = False,
        gpu_math: bool = True,
        aer_request_gpu: bool = True,
    ):
        self.seeds_path = seeds_path
        self.out_jsonl = out_jsonl
        self.max_pages = int(max_pages)

        self.max_depth = int(max_depth)
        if self.max_depth < 1:
            self.max_depth = 1

        self.max_workers = int(max_workers)
        self.request_timeout = int(request_timeout)
        self.per_host_delay = float(per_host_delay)
        self.user_agent = user_agent
        self.debug = bool(debug)

        self.allow_re = re.compile(allow_regex) if allow_regex else None
        self.deny_re = re.compile(deny_regex) if deny_regex else None

        self.gpu_math = bool(gpu_math) and _CUPY_OK
        if bool(gpu_math) and not _CUPY_OK:
            logger.info("[GPU] gpu-math requested but unavailable; using CPU math.")

        self.cohesive = CohesiveDiagonalMatrix4D(4)
        self.qpolicy = QuantumDecisionPolicy(shots=int(qiskit_shots), request_gpu=bool(aer_request_gpu))

        # HARD limit enforcement: cap SUBMITTED tasks, not completed tasks
        self._stop_event = threading.Event()
        self._submitted = 0
        self._submitted_lock = threading.Lock()

        # frontier: max-heap via negative priority
        self._frontier: List[Tuple[float, int, FrontierItem]] = []
        self._seq = 0
        self._frontier_lock = threading.Lock()

        # seen vs visited
        self._seen: Set[str] = set()
        self._visited: Set[str] = set()
        self._seen_lock = threading.Lock()
        self._visited_lock = threading.Lock()

        # per-host delay
        self._host_next_ok: Dict[str, float] = {}
        self._host_lock = threading.Lock()

        # progress
        self._pages_crawled = 0
        self._pages_lock = threading.Lock()

        # output
        self._out_lock = threading.Lock()

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def _try_reserve_page_slot(self) -> bool:
        """Reserve a slot for submitting a crawl task. Enforces hard --max-pages."""
        with self._submitted_lock:
            if self._submitted >= self.max_pages:
                self._stop_event.set()
                return False
            self._submitted += 1
            if self._submitted >= self.max_pages:
                self._stop_event.set()
            return True

    def _allowed(self, url: str) -> bool:
        if self.deny_re and self.deny_re.search(url):
            return False
        if self.allow_re and not self.allow_re.search(url):
            return False
        return True

    def _respect_host_delay(self, url: str):
        h = host_of(url)
        now = time.time()
        with self._host_lock:
            next_ok = self._host_next_ok.get(h, 0.0)
            if now < next_ok:
                time.sleep(max(0.0, next_ok - now))
            self._host_next_ok[h] = time.time() + self.per_host_delay

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

    # --- GPU-capable feature state ---
    def _feature_state_for_candidate(
        self,
        candidate_url: str,
        anchor_text: str,
        parent_url: str,
        parent_depth: int,
        parent_title: str,
    ):
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

    def _score_candidate(
        self,
        candidate_url: str,
        anchor_text: str,
        parent_url: str,
        parent_depth: int,
        parent_title: str,
    ) -> float:
        state = self._feature_state_for_candidate(candidate_url, anchor_text, parent_url, parent_depth, parent_title)
        transformed = self.cohesive.decision_function(state)

        transformed_cpu = to_cpu(transformed)
        p_good = self.qpolicy.score(transformed_cpu)

        same_host_bias = 0.05 if host_of(candidate_url) == host_of(parent_url) else 0.0
        depth_bias = 0.03 * (1.0 - (parent_depth / max(1.0, float(self.max_depth))))

        score = float(p_good + same_host_bias + depth_bias)
        return max(0.0, min(1.0, score))

    # --- Fetch/parse ---
    def _fetch(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str], int, float, str]:
        self._respect_host_delay(url)
        start = time.time()
        try:
            r = self.session.get(url, timeout=self.request_timeout, allow_redirects=True)
            status = int(r.status_code)
            ct = (r.headers.get("Content-Type") or "").lower()
            final_url = r.url
            body = r.text or ""

            if status >= 400:
                if looks_like_html(body):
                    return body, ct, final_url, status, time.time() - start, "ok_html_error_page"
                return None, ct, final_url, status, time.time() - start, "http_error"

            is_html = ("text/html" in ct) or ("application/xhtml" in ct) or looks_like_html(body)
            if not is_html:
                return None, ct, final_url, status, time.time() - start, "non_html"

            return body, ct, final_url, status, time.time() - start, "ok"

        except Exception as e:
            return None, None, None, 0, time.time() - start, f"exception:{type(e).__name__}"

    def _parse(self, html: str, base_url: str) -> Tuple[str, str, List[Tuple[str, str]]]:
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        text = soup.get_text(" ", strip=True)
        snippet = text[:4000]

        candidates: List[Tuple[str, str]] = []

        # 1) anchors
        for a in soup.find_all("a", href=True):
            u = normalize_url(base_url, a.get("href") or "")
            if u:
                anchor = a.get_text(" ", strip=True)[:200] or "a"
                candidates.append((u, anchor))

        # 2) link[href]
        for link in soup.find_all("link", href=True):
            u = normalize_url(base_url, link.get("href") or "")
            if u:
                rel = " ".join(link.get("rel") or []) if link.get("rel") else "link"
                candidates.append((u, (rel or "link")[:200]))

        # 3) meta URLs
        for meta in soup.find_all("meta"):
            content = meta.get("content")
            if not content:
                continue
            if "http://" in content or "https://" in content:
                for u0 in extract_urls_from_text(content):
                    u = normalize_url(base_url, u0)
                    if u:
                        candidates.append((u, "meta"))

        # 4) JSON-LD
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

        # 5) scripts
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

        # 6) raw HTML scan for absolute URLs
        for u0 in extract_urls_from_text(html):
            u = normalize_url(base_url, u0)
            if u:
                candidates.append((u, "raw-html"))

        # Deduplicate preserving order
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

    # --- Worker ---
    def _crawl_one(self, item: FrontierItem) -> Tuple[FrontierItem, int, int]:
        # HARD stop: if we've hit max-pages, do not work or write output
        if self._stop_event.is_set():
            return item, 0, 0

        url = item.url
        if item.depth > self.max_depth:
            return item, 0, 0

        html, ct, final_url, status, fetch_s, reason = self._fetch(url)
        if self._stop_event.is_set():
            return item, 0, 0

        if not html or not final_url:
            if not self._stop_event.is_set():
                self._write_jsonl({
                    "url": url,
                    "final_url": final_url,
                    "depth": item.depth,
                    "status": "skip",
                    "reason": reason,
                    "http_status": status,
                    "content_type": ct,
                    "fetch_seconds": fetch_s,
                    "ts": time.time(),
                })
            return item, 0, 0

        title, snippet, links = self._parse(html, final_url)
        found_links = len(links)
        enqueued = 0

        next_depth = item.depth + 1
        if next_depth <= self.max_depth:
            for out_url, anchor in links:
                if self._stop_event.is_set():
                    break
                if not self._allowed(out_url):
                    continue
                if self._is_visited(out_url):
                    continue
                if not self._mark_seen(out_url):
                    continue

                score = self._score_candidate(out_url, anchor, final_url, item.depth, title)
                self._push_frontier(FrontierItem(priority=score, url=out_url, depth=next_depth, discovered_from=final_url))
                enqueued += 1

        if not self._stop_event.is_set():
            self._write_jsonl({
                "url": url,
                "final_url": final_url,
                "depth": item.depth,
                "status": "ok",
                "reason": reason,
                "http_status": status,
                "content_type": ct,
                "fetch_seconds": fetch_s,
                "title": title,
                "snippet": snippet,
                "out_links_found": found_links,
                "out_links_enqueued": enqueued,
                "ts": time.time(),
            })

        if self.debug:
            logger.info("[PAGE] depth=%d found_links=%d enqueued=%d url=%s", item.depth, found_links, enqueued, final_url)

        self.cohesive.adapt_matrix(performance_threshold=0.8, window=200)
        return item, found_links, enqueued

    # --- Main loop ---
    def run(self):
        # reset output
        with self._out_lock:
            with open(self.out_jsonl, "w", encoding="utf-8") as f:
                f.write("")

        seeds = load_seeds(self.seeds_path)
        logger.info("Loaded %d seeds from %s", len(seeds), self.seeds_path)

        seeded = 0
        for s in seeds:
            if not self._allowed(s):
                continue
            if self._mark_seen(s):
                self._push_frontier(FrontierItem(priority=1.0, url=s, depth=0, discovered_from="seed"))
                seeded += 1

        if seeded == 0:
            logger.warning("No usable seeds after allow/deny filtering. Exiting.")
            return

        inflight: Dict = {}
        total_enqueued_links = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            while True:
                # Fill inflight up to workers, but HARD-stop when slots are exhausted
                while len(inflight) < self.max_workers and not self._stop_event.is_set():
                    nxt = self._pop_frontier()
                    if not nxt:
                        break

                    if not self._mark_visited(nxt.url):
                        continue

                    # HARD limit: reserve a slot before submitting
                    if not self._try_reserve_page_slot():
                        break

                    fut = ex.submit(self._crawl_one, nxt)
                    inflight[fut] = nxt

                # Exit conditions
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


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Quantum Decision Crawler (BS4 + Parallel + Qiskit + GPU-compatible math)")
    p.add_argument("--seeds", default="seeds.txt")
    p.add_argument("--out", default="crawl.jsonl")
    p.add_argument("--max-pages", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--delay", type=float, default=0.6)
    p.add_argument("--allow", default=None, help="Regex allowlist (optional)")
    p.add_argument("--deny", default=None, help="Regex denylist (optional)")
    p.add_argument("--shots", type=int, default=128, help="Qiskit shots (Aer only)")
    p.add_argument("--debug", action="store_true", help="Verbose per-page link stats")
    p.add_argument("--gpu-math", action="store_true", help="Enable GPU math via CuPy (if available)")
    p.add_argument("--no-aer-gpu", action="store_true", help="Do not request Aer GPU (if Aer exists)")
    args = p.parse_args()

    crawler = QuantumCrawler(
        seeds_path=args.seeds,
        out_jsonl=args.out,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        max_workers=args.workers,
        request_timeout=args.timeout,
        per_host_delay=args.delay,
        allow_regex=args.allow,
        deny_regex=args.deny,
        qiskit_shots=args.shots,
        debug=args.debug,
        gpu_math=args.gpu_math,
        aer_request_gpu=(not args.no_aer_gpu),
    )
    crawler.run()
