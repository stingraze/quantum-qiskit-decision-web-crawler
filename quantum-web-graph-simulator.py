#!/usr/bin/env python3
"""
(C)Tsubasa Kato - Inspire Search Corp. 2026/5/2
Created with help from Chat GPT, Gemini and Codex.

Quantum web crawler simulator from crawler JSON output.

Modes:
- train: trains a lightweight transition model from crawled graph JSON
- infer: loads trained model and simulates quantum-biased web hops

Need to improve: sites may cycle through same sites. Need to fix.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("quantum_web_graph_sim")


def load_graph_json(path: str) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports crawler6 output format + direct graph dumps.
    graph = data.get("graph") or data.get("adjacency") or {}
    if not graph:
        raise ValueError("Input JSON missing graph/adjacency object.")

    score_map = {}
    for row in data.get("next_n", []):
        if isinstance(row, dict) and "url" in row and "score" in row:
            score_map[row["url"]] = float(row["score"])
    return graph, score_map


def build_edge_dataset(graph: Dict[str, List[str]], score_map: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    indeg = defaultdict(int)
    outdeg = {u: len(v) for u, v in graph.items()}
    for u, vs in graph.items():
        for v in vs:
            indeg[v] += 1

    X, y = [], []
    for src, nbrs in graph.items():
        sdom = urlparse(src).netloc
        for dst in nbrs:
            ddom = urlparse(dst).netloc
            feat = [
                1.0 if sdom == ddom else 0.0,
                min(indeg[dst] / max(1, max(indeg.values(), default=1)), 1.0),
                min(outdeg.get(dst, 0) / max(1, max(outdeg.values(), default=1)), 1.0),
                score_map.get(dst, 0.5),
            ]
            # pseudo target: stronger if same-domain and destination has popularity
            target = min(1.0, 0.4 * feat[0] + 0.3 * feat[1] + 0.3 * feat[3])
            X.append(feat)
            y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class TinyEdgeModel:
    def __init__(self, lr: float = 0.08, epochs: int = 180):
        self.lr = lr
        self.epochs = epochs
        self.w = np.zeros(4, dtype=np.float32)
        self.b = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0:
            return
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (p - y) / len(X))
            self.b -= self.lr * float(np.mean(p - y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return self._sigmoid(X @ self.w + self.b).astype(np.float32)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"w": self.w.tolist(), "b": self.b}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TinyEdgeModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = cls()
        m.w = np.asarray(data["w"], dtype=np.float32)
        m.b = float(data["b"])
        return m


class QuantumGraphSimulator:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha

    def simulate(self, graph: Dict[str, List[str]], model: TinyEdgeModel, start: str, steps: int, score_map: Dict[str, float]) -> List[str]:
        if start not in graph:
            raise ValueError("start URL not in graph")
        path = [start]
        cur = start

        for _ in range(steps):
            nbrs = graph.get(cur, [])
            if not nbrs:
                break
            probs = self._edge_probs(cur, nbrs, graph, model, score_map)
            cur = self._quantum_sample(nbrs, probs)
            path.append(cur)
        return path

    def _edge_probs(self, src: str, nbrs: List[str], graph: Dict[str, List[str]], model: TinyEdgeModel, score_map: Dict[str, float]) -> np.ndarray:
        indeg = defaultdict(int)
        for _, vs in graph.items():
            for v in vs:
                indeg[v] += 1
        outdeg = {u: len(v) for u, v in graph.items()}
        sdom = urlparse(src).netloc
        max_in = max(indeg.values(), default=1)
        max_out = max(outdeg.values(), default=1)

        feats = []
        for dst in nbrs:
            ddom = urlparse(dst).netloc
            feats.append([
                1.0 if sdom == ddom else 0.0,
                min(indeg[dst] / max(1, max_in), 1.0),
                min(outdeg.get(dst, 0) / max(1, max_out), 1.0),
                score_map.get(dst, 0.5),
            ])
        p = model.predict(np.asarray(feats, dtype=np.float32)).astype(np.float64)
        p = np.clip(p, 1e-4, None)
        p /= p.sum()
        return p

    def _quantum_sample(self, nbrs: List[str], probs: np.ndarray) -> str:
        if len(nbrs) <= 4:
            return self._sample_bucket(nbrs, probs)

        idx = np.argsort(-probs)
        head, tail = idx[:4], idx[4:]
        head_probs = probs[head] / probs[head].sum()
        quantum = self._bucket_probs(head_probs)

        mixed = np.zeros_like(probs)
        mass = min(0.85, max(0.55, float(probs[head].sum())))
        mixed[head] = mass * (quantum / max(quantum.sum(), 1e-9))
        if len(tail) > 0:
            t = probs[tail]
            mixed[tail] = (1 - mass) * (t / max(t.sum(), 1e-9))
        mixed /= max(mixed.sum(), 1e-9)
        return np.random.choice(nbrs, p=mixed)

    def _sample_bucket(self, nbrs: List[str], probs: np.ndarray) -> str:
        w = self._bucket_probs(probs)
        w /= max(w.sum(), 1e-9)
        return np.random.choice(nbrs, p=w)

    def _bucket_probs(self, probs: np.ndarray) -> np.ndarray:
        n = len(probs)
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        for i, p in enumerate(probs[:4]):
            theta = 2 * math.asin(min(0.999, max(0.001, math.sqrt(float(p)))))
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quantum web graph simulator from JSON")
    ap.add_argument("--mode", choices=["train", "infer"], required=True)
    ap.add_argument("--graph-json", required=True)
    ap.add_argument("--model-path", default="quantum_graph_model.json")
    ap.add_argument("--start-url", default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out-json", default="quantum_graph_simulation.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    graph, score_map = load_graph_json(args.graph_json)

    if args.mode == "train":
        X, y = build_edge_dataset(graph, score_map)
        model = TinyEdgeModel()
        model.fit(X, y)
        model.save(args.model_path)
        logger.info("trained model saved to %s (samples=%d)", args.model_path, len(X))
        return

    model = TinyEdgeModel.load(args.model_path)
    start = args.start_url or random.choice(list(graph.keys()))
    sim = QuantumGraphSimulator()
    path = sim.simulate(graph=graph, model=model, start=start, steps=args.steps, score_map=score_map)
    result = {"start_url": start, "steps": args.steps, "path": path, "nodes": len(graph)}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("inference simulation saved to %s", args.out_json)


if __name__ == "__main__":
    main()
