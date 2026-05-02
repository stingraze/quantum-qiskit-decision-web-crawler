#!/usr/bin/env python3
"""
(C)Tsubasa Kato - Inspire Search Corp. 2026/5/2
Created with help from Chat GPT, Gemini and Codex.
Quantum Decision Crawler v6 (lightweight MLX + quantum hop simulator)

- Trains a tiny model (MLX on Apple GPU when available; NumPy fallback elsewhere)
  to rank candidate URLs in the next N crawl sequence.
- Builds a realistic site graph from crawled links.
- Uses a quantum-circuit-inspired hop simulator (Qiskit Statevector) that biases
  movement using graph connectivity and learned URL scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("quantum_crawler_v6")

try:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
    MLX_AVAILABLE = True
except Exception:
    mx = None  # type: ignore
    nn = None  # type: ignore
    MLX_AVAILABLE = False


@dataclass
class Page:
    url: str
    title: str
    depth: int
    links: List[str]


class WebGraphCrawler:
    def __init__(self, timeout: float = 8.0, max_links_per_page: int = 64):
        self.timeout = float(timeout)
        self.max_links_per_page = int(max_links_per_page)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "QuantumDecisionCrawler6/1.0"})

    def crawl(self, seeds: List[str], max_pages: int = 120, max_depth: int = 2) -> Dict[str, Page]:
        q = deque((self._norm(s), 0) for s in seeds if s.strip())
        seen: Set[str] = set()
        pages: Dict[str, Page] = {}
        logger.debug("[crawl] start seeds=%d max_pages=%d max_depth=%d", len(seeds), max_pages, max_depth)

        while q and len(pages) < max_pages:
            url, depth = q.popleft()
            logger.debug("[crawl] pop depth=%d queue=%d seen=%d url=%s", depth, len(q), len(seen), url)
            if url in seen or depth > max_depth:
                continue
            seen.add(url)

            html = self._fetch(url)
            if not html:
                continue

            title, links = self._extract(url, html)
            pages[url] = Page(url=url, title=title, depth=depth, links=links)
            logger.debug("[crawl] accepted url=%s title_len=%d links=%d pages=%d", url, len(title), len(links), len(pages))
            for nxt in links:
                if nxt not in seen and len(pages) + len(q) < max_pages * 3:
                    q.append((nxt, depth + 1))

        logger.debug("[crawl] completed pages=%d seen=%d", len(pages), len(seen))
        return pages

    def _fetch(self, url: str) -> Optional[str]:
        try:
            r = self.session.get(url, timeout=self.timeout)
            if "text/html" not in (r.headers.get("Content-Type") or ""):
                return None
            if r.status_code >= 400:
                return None
            return r.text
        except Exception:
            return None

    def _extract(self, base: str, html: str) -> Tuple[str, List[str]]:
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else ""
        out: List[str] = []
        for a in soup.find_all("a", href=True):
            href = self._norm(urljoin(base, a["href"]))
            if self._allowed(href):
                out.append(href)
            if len(out) >= self.max_links_per_page:
                break
        return title, list(dict.fromkeys(out))

    @staticmethod
    def _allowed(url: str) -> bool:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        path = (p.path or "").lower()
        if re.search(r"\.(png|jpg|jpeg|gif|svg|webp|pdf|zip|mp4|mp3)$", path):
            return False
        return True

    @staticmethod
    def _norm(url: str) -> str:
        clean = urldefrag(url.strip())[0]
        p = urlparse(clean)
        scheme = p.scheme.lower() or "https"
        netloc = p.netloc.lower()
        path = p.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return f"{scheme}://{netloc}{path}" + (f"?{p.query}" if p.query else "")


class LightweightURLRanker:
    """Tiny ranking model for 'next n' pages.

    Features: [same_domain, path_depth, in_degree, out_degree, title_len_norm, seen_depth_norm]
    """

    def __init__(self, lr: float = 0.05, epochs: int = 120):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.w = np.zeros(6, dtype=np.float32)
        self.b = 0.0
        self.mlx_used = False

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            logger.debug("[ranker] empty training matrix")
            return
        if MLX_AVAILABLE:
            try:
                self._fit_mlx(X, y)
                self.mlx_used = True
                return
            except Exception as e:
                logger.warning("MLX training failed, fallback to NumPy: %s", e)

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            grad_w = (X.T @ (p - y)) / len(X)
            grad_b = float(np.mean(p - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        logger.debug("[ranker] numpy fit complete epochs=%d samples=%d", self.epochs, len(X))

    def _fit_mlx(self, X: np.ndarray, y: np.ndarray) -> None:
        X_m = mx.array(X.astype(np.float32))
        y_m = mx.array(y.astype(np.float32)).reshape((-1, 1))
        model = nn.Linear(6, 1)

        def loss_fn(model, xb, yb):
            logits = model(xb)
            probs = 1 / (1 + mx.exp(-logits))
            eps = 1e-6
            loss = -(yb * mx.log(probs + eps) + (1 - yb) * mx.log(1 - probs + eps))
            return mx.mean(loss)

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        for _ in range(self.epochs):
            loss, grads = loss_and_grad(model, X_m, y_m)
            model.update({k: v - self.lr * grads[k] for k, v in model.parameters().items()})
            mx.eval(loss)
        logger.debug("[ranker] mlx fit complete epochs=%d samples=%d", self.epochs, len(X))

        self.w = np.asarray(model.weight).reshape(-1).astype(np.float32)
        self.b = float(np.asarray(model.bias).reshape(-1)[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return self._sigmoid(X @ self.w + self.b).astype(np.float32)


class QuantumHopSimulator:
    def __init__(self, alpha: float = 0.65):
        self.alpha = float(alpha)

    def simulate(self, start_url: str, adjacency: Dict[str, List[str]], score: Dict[str, float], steps: int) -> List[str]:
        if start_url not in adjacency:
            return []
        path = [start_url]
        cur = start_url
        logger.debug("[sim] start=%s steps=%d", start_url, steps)
        for _ in range(steps):
            nbrs = adjacency.get(cur, [])
            if not nbrs:
                logger.debug("[sim] dead-end at=%s", cur)
                break
            cur = self._quantum_pick(cur, nbrs, score)
            path.append(cur)
            logger.debug("[sim] hop -> %s", cur)
        return path

    def _quantum_pick(self, cur: str, nbrs: List[str], score: Dict[str, float]) -> str:
        local = nbrs
        probs = np.asarray(self._connectivity_probs(cur, local, score), dtype=np.float64)

        if len(local) <= 4:
            weighted = self._quantum_bucket_probs(probs, len(local))
            if weighted.sum() <= 0:
                return random.choice(local)
            weighted /= weighted.sum()
            return np.random.choice(local, p=weighted)

        # For larger neighborhoods, use the quantum circuit on top candidates
        # and sample tail candidates classically to preserve realistic diversity.
        idx = np.argsort(-probs)
        head_idx, tail_idx = idx[:4], idx[4:]
        head_probs = probs[head_idx]
        head_probs = head_probs / max(head_probs.sum(), 1e-9)
        head_weighted = self._quantum_bucket_probs(head_probs, 4)

        quantum_mass = min(0.85, max(0.55, float(probs[head_idx].sum())))
        mixed = np.zeros_like(probs)
        mixed[head_idx] = quantum_mass * (head_weighted / max(head_weighted.sum(), 1e-9))

        if len(tail_idx) > 0:
            tail_raw = probs[tail_idx]
            tail_raw = tail_raw / max(tail_raw.sum(), 1e-9)
            mixed[tail_idx] = (1.0 - quantum_mass) * tail_raw

        mixed /= max(mixed.sum(), 1e-9)
        return np.random.choice(local, p=mixed)


    def _quantum_bucket_probs(self, probs: np.ndarray, n: int) -> np.ndarray:
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        for i, p in enumerate(probs[:n]):
            theta = 2.0 * math.asin(min(0.999, max(0.001, math.sqrt(float(p)))))
            if i in (1, 3):
                qc.x(0)
            if i in (2, 3):
                qc.x(1)
            qc.ry(theta * self.alpha, 0)
            qc.ry(theta * (1 - self.alpha), 1)
            if i in (1, 3):
                qc.x(0)
            if i in (2, 3):
                qc.x(1)

        sv = Statevector.from_instruction(qc)
        d = sv.probabilities_dict()
        return np.array([d.get(b, 0.0) for b in ("00", "01", "10", "11")], dtype=np.float64)[:n]

    @staticmethod
    def _connectivity_probs(cur: str, nbrs: List[str], score: Dict[str, float]) -> List[float]:
        raw = []
        cdom = urlparse(cur).netloc
        for u in nbrs:
            dom = urlparse(u).netloc
            same_dom = 1.0 if dom == cdom else 0.0
            raw.append(0.45 * same_dom + 0.55 * score.get(u, 0.5))
        arr = np.asarray(raw, dtype=np.float64)
        arr = np.clip(arr, 1e-4, None)
        arr /= arr.sum()
        return arr.tolist()


def build_graph(pages: Dict[str, Page]) -> Dict[str, List[str]]:
    keys = set(pages.keys())
    g: Dict[str, List[str]] = {}
    for url, pg in pages.items():
        g[url] = [u for u in pg.links if u in keys]
    return g


def build_training_matrix(pages: Dict[str, Page], graph: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    indeg = defaultdict(int)
    for src, outs in graph.items():
        for d in outs:
            indeg[d] += 1

    urls = list(pages.keys())
    if not urls:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

    top_domain = max((urlparse(u).netloc for u in urls), key=lambda d: sum(1 for x in urls if urlparse(x).netloc == d))
    max_in = max(indeg.values()) if indeg else 1
    max_out = max((len(v) for v in graph.values()), default=1)
    max_depth = max((p.depth for p in pages.values()), default=1)

    X, y, ordered = [], [], []
    for u in urls:
        pg = pages[u]
        p = urlparse(u)
        path_depth = len([x for x in p.path.split("/") if x])
        feat = [
            1.0 if p.netloc == top_domain else 0.0,
            min(path_depth / 8.0, 1.0),
            min(indeg[u] / max_in, 1.0),
            min(len(graph.get(u, [])) / max_out, 1.0),
            min(len(pg.title) / 120.0, 1.0),
            min(pg.depth / max_depth, 1.0),
        ]
        # pseudo-label: good hubs are those with higher in-degree and moderate depth.
        label = 1.0 if (indeg[u] >= np.percentile(list(indeg.values()) or [0], 60) and pg.depth <= max(1, max_depth // 2)) else 0.0
        X.append(feat)
        y.append(label)
        ordered.append(u)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), ordered


def decide_next_n(ordered_urls: List[str], X: np.ndarray, model: LightweightURLRanker, n: int) -> List[Tuple[str, float]]:
    probs = model.predict_proba(X)
    ranked = sorted(zip(ordered_urls, probs.tolist()), key=lambda t: t[1], reverse=True)
    return ranked[:n]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quantum Decision Crawler v6")
    ap.add_argument("--seeds", type=str, default="seeds.txt", help="Seed file (one URL per line)")
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--next-n", type=int, default=15)
    ap.add_argument("--sim-steps", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--out-json", type=str, default="crawler6_output.json")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("[main] debug logging enabled")
    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]

    crawler = WebGraphCrawler(timeout=args.timeout)
    pages = crawler.crawl(seeds=seeds, max_pages=args.max_pages, max_depth=args.max_depth)
    graph = build_graph(pages)

    X, y, ordered = build_training_matrix(pages, graph)
    logger.debug("[main] matrix rows=%d cols=%d", X.shape[0], X.shape[1] if X.ndim == 2 else 0)
    ranker = LightweightURLRanker()
    ranker.fit(X, y)

    next_n = decide_next_n(ordered, X, ranker, args.next_n)
    score_map = {u: s for u, s in zip(ordered, ranker.predict_proba(X).tolist())}

    start = next_n[0][0] if next_n else (ordered[0] if ordered else "")
    sim = QuantumHopSimulator()
    simulated_path = sim.simulate(start, graph, score_map, args.sim_steps) if start else []

    result = {
        "pages_crawled": len(pages),
        "mlx_used": ranker.mlx_used,
        "graph": graph,  # <-- Added this line
        "next_n": [{"url": u, "score": round(float(s), 6)} for u, s in next_n],
        "simulation": {
            "start_url": start,
            "steps": args.sim_steps,
            "path": simulated_path,
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.debug("[main] next_n_count=%d sim_path_len=%d", len(next_n), len(simulated_path))
    logger.info("Crawled %d pages. MLX used=%s. Wrote %s", len(pages), ranker.mlx_used, args.out_json)


if __name__ == "__main__":
    main()
