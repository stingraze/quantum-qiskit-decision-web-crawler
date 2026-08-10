#!/usr/bin/env python3
"""
Quantum Hybrid Web Crawler
==========================

Pipeline:
  seeds.txt -> polite crawl / sitemap discovery / optional JS rendering
  -> BeautifulSoup link extraction -> decision-tree scoring
  -> Grover-amplified selection of next URLs.

Install:
  pip install requests beautifulsoup4 numpy scikit-learn qiskit qiskit-aer

Optional JavaScript rendering:
  pip install playwright
  playwright install chromium

Example:
  python quantum_hybrid_crawler.py \
      --seeds-file seeds.txt --max-pages 50 --max-depth 3 \
      --top-k-per-page 8 --buffer-size 32 --n-next-steps 5 \
      --render-javascript --debug-html-dir debug-html

seeds.txt format:
  # one HTTP(S) URL per line
  https://example.org/
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse, urldefrag
from urllib.robotparser import RobotFileParser

import numpy as np
from bs4 import BeautifulSoup

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:
    raise SystemExit("Missing dependency. Install with: pip install requests") from exc

try:
    from sklearn.tree import DecisionTreeClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


Link = namedtuple("Link", "url text depth same_domain")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
KEYWORDS = ("quantum", "qiskit", "apl", "mainframe", "docs", "middleware", "blog")
TRACKING_PARAMETERS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "_ga", "_gl", "ref", "ref_"
}
SKIP_EXTENSION_RE = re.compile(
    r"\.(?:7z|avi|bin|csv|dmg|docx?|eot|exe|gif|gz|ico|iso|jpe?g|m4[av]|mp[34]|"
    r"odp|ods|odt|otf|pdf|png|pptx?|rar|tar|tgz|ttf|wav|webm|woff2?|xls[xm]?|zip)$",
    re.IGNORECASE,
)


@dataclass
class CrawlConfig:
    seeds_file: str = "seeds.txt"
    max_pages: int = 50
    max_depth: int = 3
    top_k_per_page: int = 8
    buffer_size: int = 32
    n_next_steps: int = 5
    score_threshold: float = 0.5
    shots: int = 4096
    allow_external: bool = False
    respect_robots: bool = True
    request_timeout: float = 15.0
    min_host_delay: float = 1.0
    user_agent: str = DEFAULT_USER_AGENT
    render_javascript: bool = False
    render_wait_ms: int = 750
    discover_sitemaps: bool = True
    sitemap_url_limit: int = 5000
    debug_html_dir: str | None = None
    jsonl_output: str | None = "crawl_pages.jsonl"


@dataclass
class CrawlReport:
    buffer: list[tuple[Link, float]]
    pages_fetched: int
    visited_urls: set[str]
    errors: list[dict]
    discovered_urls: int


def parse_args() -> CrawlConfig:
    parser = argparse.ArgumentParser(description="Polite decision-tree and Grover-assisted web crawler")
    parser.add_argument("--seeds-file", default="seeds.txt")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--top-k-per-page", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=32)
    parser.add_argument("--n-next-steps", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--ignore-robots", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--min-host-delay", type=float, default=1.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--render-javascript", action="store_true")
    parser.add_argument("--render-wait-ms", type=int, default=750)
    parser.add_argument("--no-sitemaps", action="store_true")
    parser.add_argument("--debug-html-dir", default=None)
    parser.add_argument("--jsonl-output", default="crawl_pages.jsonl")
    args = parser.parse_args()
    return CrawlConfig(
        seeds_file=args.seeds_file,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        top_k_per_page=args.top_k_per_page,
        buffer_size=args.buffer_size,
        n_next_steps=args.n_next_steps,
        score_threshold=args.score_threshold,
        shots=args.shots,
        allow_external=args.allow_external,
        respect_robots=not args.ignore_robots,
        request_timeout=args.timeout,
        min_host_delay=args.min_host_delay,
        user_agent=args.user_agent,
        render_javascript=args.render_javascript,
        render_wait_ms=args.render_wait_ms,
        discover_sitemaps=not args.no_sitemaps,
        debug_html_dir=args.debug_html_dir,
        jsonl_output=args.jsonl_output,
    )


def validate_config(cfg: CrawlConfig) -> None:
    if cfg.max_pages < 1:
        raise ValueError("--max-pages must be at least 1")
    if cfg.max_depth < 0:
        raise ValueError("--max-depth must be non-negative")
    if cfg.top_k_per_page < 1:
        raise ValueError("--top-k-per-page must be at least 1")
    if cfg.buffer_size < 1:
        raise ValueError("--buffer-size must be at least 1")
    if cfg.n_next_steps < 1:
        raise ValueError("--n-next-steps must be at least 1")
    if not 0.0 <= cfg.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1")
    if cfg.min_host_delay < 0:
        raise ValueError("--min-host-delay must be non-negative")


def canonicalize(url: str) -> str | None:
    """Return a deduplication-friendly HTTP(S) URL, or None if invalid."""
    url = urldefrag(url).url
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None
    host = (parsed.hostname or "").lower()
    netloc = host
    if parsed.port and not ((parsed.scheme == "http" and parsed.port == 80) or
                            (parsed.scheme == "https" and parsed.port == 443)):
        netloc = f"{host}:{parsed.port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    query_pairs = [
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMETERS and not key.lower().startswith("utm_")
    ]
    return urlunparse((parsed.scheme.lower(), netloc, path, "", urlencode(sorted(query_pairs)), ""))


def read_seeds(path: str) -> list[str]:
    seeds, seen = [], set()
    with open(path, "r", encoding="utf-8") as file:
        for number, raw in enumerate(file, 1):
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            url = canonicalize(value)
            if not url:
                print(f"[warn] invalid seed skipped at {path}:{number}: {value!r}")
                continue
            if url not in seen:
                seen.add(url)
                seeds.append(url)
    if not seeds:
        raise ValueError(f"No valid HTTP(S) seed URLs in {path!r}")
    return seeds


# ---------------------------------------------------------------------------
# Decision-tree scorer
# ---------------------------------------------------------------------------

def featurize(link: Link) -> list[int]:
    parsed = urlparse(link.url)
    path = parsed.path.lower()
    text = link.text.lower()
    return [
        path.strip("/").count("/") + int(bool(path.strip("/"))),
        sum(word in path or word in text for word in KEYWORDS),
        min(len(link.text), 200),
        int(link.same_domain),
        int(bool(SKIP_EXTENSION_RE.search(path))),
    ]


def synthetic_training_set() -> tuple[list[list[int]], list[int]]:
    """Replace with labels gathered from your own crawl history in production."""
    X = [
        [1, 3, 20, 1, 0], [1, 2, 25, 1, 0], [1, 0, 8, 1, 0],
        [0, 0, 9, 0, 0], [1, 2, 28, 1, 0], [1, 0, 14, 1, 0],
        [0, 1, 5, 0, 1],
    ]
    y = [1, 1, 0, 0, 1, 0, 0]
    return X, y


def rule_based_score(features: list[int]) -> float:
    _, keyword_hits, anchor_length, same_domain, binary = features
    return float(np.clip(
        0.40 * same_domain + 0.15 * min(keyword_hits, 3) +
        0.10 * int(anchor_length > 12) - 0.50 * binary,
        0.0, 1.0,
    ))


def score_links(links: list[Link]) -> list[float]:
    """Return one score per link. Empty input is valid and returns []."""
    if not links:
        return []
    features = [featurize(link) for link in links]
    if not HAS_SKLEARN:
        return [rule_based_score(row) for row in features]

    X_train, y_train = synthetic_training_set()
    classifier = DecisionTreeClassifier(max_depth=3, random_state=0)
    classifier.fit(X_train, y_train)
    if 1 not in classifier.classes_:
        return [0.0] * len(links)
    positive_column = list(classifier.classes_).index(1)
    return classifier.predict_proba(features)[:, positive_column].tolist()


# ---------------------------------------------------------------------------
# Crawl engine
# ---------------------------------------------------------------------------

class PersistentRenderer:
    def __init__(self, user_agent: str, timeout_ms: int, wait_ms: int):
        self.user_agent = user_agent
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
        self.playwright = None
        self.browser = None

    def content(self, url: str) -> tuple[str, str]:
        if not HAS_PLAYWRIGHT:
            raise RuntimeError(
                "Install JavaScript rendering support with: "
                "pip install playwright && playwright install chromium"
            )
        if self.browser is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
        context = self.browser.new_context(user_agent=self.user_agent)
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            page.wait_for_timeout(self.wait_ms)
            return page.content(), page.url
        finally:
            context.close()

    def close(self) -> None:
        if self.browser is not None:
            self.browser.close()
            self.browser = None
        if self.playwright is not None:
            self.playwright.stop()
            self.playwright = None


class PoliteCrawler:
    def __init__(self, cfg: CrawlConfig):
        self.cfg = cfg
        self.session = self._make_session()
        self.robots: dict[str, RobotFileParser] = {}
        self.host_last_request: dict[str, float] = defaultdict(float)
        self.errors: list[dict] = []
        self.renderer = (
            PersistentRenderer(cfg.user_agent, int(cfg.request_timeout * 1000), cfg.render_wait_ms)
            if cfg.render_javascript else None
        )
        self.output = None

    def _make_session(self) -> requests.Session:
        retries = Retry(
            total=3, connect=3, read=3, status=3, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET", "HEAD")),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.cfg.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def host_key(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _wait_for_host(self, url: str) -> None:
        host = self.host_key(url)
        wait = self.cfg.min_host_delay - (time.monotonic() - self.host_last_request[host])
        if wait > 0:
            time.sleep(wait)
        self.host_last_request[host] = time.monotonic()

    def _robots_for(self, url: str) -> RobotFileParser:
        origin = self.host_key(url)
        if origin in self.robots:
            return self.robots[origin]
        parser = RobotFileParser()
        robots_url = f"{origin}/robots.txt"
        try:
            self._wait_for_host(robots_url)
            response = self.session.get(robots_url, timeout=self.cfg.request_timeout)
            if response.ok:
                parser.parse(response.text.splitlines())
            else:
                parser.allow_all = True
        except requests.RequestException:
            parser.allow_all = True
        self.robots[origin] = parser
        return parser

    def allowed_by_robots(self, url: str) -> bool:
        return not self.cfg.respect_robots or self._robots_for(url).can_fetch(self.cfg.user_agent, url)

    def fetch_html(self, url: str) -> tuple[str, str]:
        self._wait_for_host(url)
        response = self.session.get(url, timeout=self.cfg.request_timeout, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and "xhtml" not in content_type:
            raise ValueError(f"not HTML: {content_type or 'missing Content-Type'}")
        return response.text, canonicalize(response.url) or response.url

    @staticmethod
    def extract_links(html: str, base_url: str, depth: int) -> list[Link]:
        soup = BeautifulSoup(html, "html.parser")
        base_host = (urlparse(base_url).hostname or "").lower()
        output, seen = [], set()
        for element in soup.select("a[href], area[href]"):
            href = element.get("href", "").strip()
            if not href or href.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
                continue
            url = canonicalize(urljoin(base_url, href))
            if not url or url in seen or SKIP_EXTENSION_RE.search(urlparse(url).path):
                continue
            seen.add(url)
            output.append(Link(
                url=url,
                text=element.get_text(" ", strip=True) or element.get("aria-label", "") or element.get("alt", ""),
                depth=depth,
                same_domain=((urlparse(url).hostname or "").lower() == base_host),
            ))
        return output

    def sitemap_urls(self, seed: str) -> list[str]:
        if not self.cfg.discover_sitemaps:
            return []
        parser = self._robots_for(seed)
        pending = list(set((parser.site_maps() or []) + [f"{self.host_key(seed)}/sitemap.xml"]))
        visited, result = set(), []
        while pending and len(result) < self.cfg.sitemap_url_limit:
            sitemap = pending.pop()
            if sitemap in visited:
                continue
            visited.add(sitemap)
            try:
                self._wait_for_host(sitemap)
                response = self.session.get(sitemap, timeout=self.cfg.request_timeout)
                if not response.ok:
                    continue
                root = ET.fromstring(response.content)
            except Exception:
                continue
            document_type = root.tag.rsplit("}", 1)[-1].lower()
            locations = [node.text.strip() for node in root.findall(".//{*}loc") if node.text]
            if document_type == "sitemapindex":
                pending.extend(locations)
            else:
                for location in locations:
                    url = canonicalize(location)
                    if url:
                        result.append(url)
                        if len(result) >= self.cfg.sitemap_url_limit:
                            break
        return result

    def _write_log(self, record: dict) -> None:
        if self.output:
            self.output.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.output.flush()

    def _save_debug_html(self, url: str, html: str, ordinal: int) -> None:
        if not self.cfg.debug_html_dir:
            return
        directory = Path(self.cfg.debug_html_dir)
        directory.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", urlparse(url).netloc)
        (directory / f"no_links_{ordinal:05d}_{name}.html").write_text(html, encoding="utf-8")

    def crawl(self, seeds: Iterable[str]) -> CrawlReport:
        seeds = [canonicalize(seed) for seed in seeds]
        seeds = [seed for seed in seeds if seed]
        allowed_hosts = {(urlparse(seed).hostname or "").lower() for seed in seeds}
        frontier: list[tuple[float, int, int, str]] = []
        queued, visited = set(), set()
        candidate_scores: dict[str, tuple[Link, float]] = {}
        sequence = 0

        def enqueue(url: str, depth: int, score: float = 0.0) -> None:
            nonlocal sequence
            url = canonicalize(url)
            if not url or depth > self.cfg.max_depth or url in queued or url in visited:
                return
            host = (urlparse(url).hostname or "").lower()
            if not self.cfg.allow_external and host not in allowed_hosts:
                return
            if SKIP_EXTENSION_RE.search(urlparse(url).path):
                return
            queued.add(url)
            heapq.heappush(frontier, (-score, depth, sequence, url))
            sequence += 1

        for seed in seeds:
            enqueue(seed, 0, 1.0)
            for sitemap_url in self.sitemap_urls(seed):
                enqueue(sitemap_url, 1, 0.05)

        if self.cfg.jsonl_output:
            self.output = open(self.cfg.jsonl_output, "w", encoding="utf-8")

        pages_fetched = 0
        try:
            while frontier and pages_fetched < self.cfg.max_pages:
                _, depth, _, requested_url = heapq.heappop(frontier)
                queued.discard(requested_url)
                if requested_url in visited:
                    continue
                if not self.allowed_by_robots(requested_url):
                    visited.add(requested_url)
                    self._write_log({"url": requested_url, "depth": depth, "status": "robots_disallowed"})
                    continue

                try:
                    html, final_url = self.fetch_html(requested_url)
                    pages_fetched += 1
                    visited.update((requested_url, final_url))
                except Exception as exc:
                    visited.add(requested_url)
                    error = {"url": requested_url, "depth": depth, "error": str(exc)}
                    self.errors.append(error)
                    self._write_log({"status": "fetch_error", **error})
                    print(f"[warn] fetch failed: {requested_url} ({exc})")
                    continue

                links = self.extract_links(html, final_url, depth + 1)
                source = "static"
                if not links and self.renderer:
                    try:
                        rendered_html, rendered_url = self.renderer.content(final_url)
                        rendered_url = canonicalize(rendered_url) or final_url
                        links = self.extract_links(rendered_html, rendered_url, depth + 1)
                        html, final_url, source = rendered_html, rendered_url, "rendered"
                    except Exception as exc:
                        self.errors.append({"url": final_url, "depth": depth, "error": f"render error: {exc}"})
                        print(f"[warn] JavaScript rendering failed: {final_url} ({exc})")

                self._write_log({
                    "url": final_url, "requested_url": requested_url, "depth": depth,
                    "status": "ok", "source": source, "link_count": len(links),
                })
                if not links:
                    print(f"[info] no discoverable links: {final_url}")
                    self._save_debug_html(final_url, html, pages_fetched)
                    continue

                scores = score_links(links)
                if len(scores) != len(links):
                    raise ValueError("score_links must produce one score for every Link")
                ranked = sorted(zip(links, scores), key=lambda pair: pair[1], reverse=True)
                for link, score in ranked:
                    old = candidate_scores.get(link.url)
                    if old is None or score > old[1]:
                        candidate_scores[link.url] = (link, float(score))
                if depth < self.cfg.max_depth:
                    for link, score in ranked[:self.cfg.top_k_per_page]:
                        enqueue(link.url, depth + 1, float(score))
        finally:
            if self.output:
                self.output.close()
            if self.renderer:
                self.renderer.close()
            self.session.close()

        ranked_buffer = sorted(candidate_scores.values(), key=lambda pair: pair[1], reverse=True)
        return CrawlReport(ranked_buffer[:self.cfg.buffer_size], pages_fetched, visited, self.errors, len(candidate_scores))


# ---------------------------------------------------------------------------
# Grover selection
# ---------------------------------------------------------------------------

def next_power_of_two(value: int) -> int:
    return 1 << max(1, (value - 1).bit_length())


def marked_indices(scores: list[float], threshold: float) -> list[int]:
    marked = [index for index, score in enumerate(scores) if score >= threshold]
    return marked if marked else [int(np.argmax(scores))]


def optimal_iterations(state_count: int, marked_count: int) -> int:
    if not 0 < marked_count < state_count:
        return 0
    return max(1, round(math.pi * math.sqrt(state_count / marked_count) / 4))


def grover_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name="priority_oracle")
    for index in marked:
        little_endian_bits = format(index, f"0{n_qubits}b")[::-1]
        zeros = [qubit for qubit, bit in enumerate(little_endian_bits) if bit == "0"]
        circuit.x(zeros)
        if n_qubits == 1:
            circuit.z(0)
        else:
            target = n_qubits - 1
            circuit.h(target)
            circuit.mcx(list(range(target)), target)
            circuit.h(target)
        circuit.x(zeros)
    return circuit


def grover_diffuser(n_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, name="diffuser")
    circuit.h(range(n_qubits))
    circuit.x(range(n_qubits))
    if n_qubits == 1:
        circuit.z(0)
    else:
        target = n_qubits - 1
        circuit.h(target)
        circuit.mcx(list(range(target)), target)
        circuit.h(target)
    circuit.x(range(n_qubits))
    circuit.h(range(n_qubits))
    return circuit


def build_grover_circuit(n_qubits: int, marked: list[int], iterations: int) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(range(n_qubits))
    oracle, diffuser = grover_oracle(n_qubits, marked), grover_diffuser(n_qubits)
    for _ in range(iterations):
        circuit.compose(oracle, inplace=True)
        circuit.compose(diffuser, inplace=True)
    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def numpy_grover_probabilities(n_qubits: int, marked: list[int], iterations: int) -> np.ndarray:
    state_count = 2 ** n_qubits
    state = np.ones(state_count, dtype=complex) / math.sqrt(state_count)
    oracle_mask = np.zeros(state_count, dtype=bool)
    oracle_mask[marked] = True
    for _ in range(iterations):
        state[oracle_mask] *= -1
        state = 2 * state.mean() - state
    return np.abs(state) ** 2


def select_next_steps(buffer: list[tuple[Link, float]], cfg: CrawlConfig) -> dict:
    if not buffer:
        return {"backend": None, "n_qubits": 0, "marked": [], "iterations": 0, "next_steps": []}
    links = [link for link, _ in buffer]
    scores = [score for _, score in buffer]
    state_count = next_power_of_two(len(links))
    n_qubits = state_count.bit_length() - 1
    padded_scores = scores + [0.0] * (state_count - len(scores))
    marked = marked_indices(padded_scores, cfg.score_threshold)
    iterations = optimal_iterations(state_count, len(marked))

    if HAS_QISKIT:
        circuit = build_grover_circuit(n_qubits, marked, iterations)
        result = AerSimulator().run(circuit, shots=cfg.shots).result()
        probabilities = {int(bitstring, 2): count / cfg.shots for bitstring, count in result.get_counts().items()}
        backend = "qiskit_aer"
    else:
        probabilities = dict(enumerate(numpy_grover_probabilities(n_qubits, marked, iterations)))
        backend = "numpy_fallback"

    choices = []
    for index, probability in sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True):
        if index < len(links):
            choices.append((links[index], probability))
        if len(choices) >= cfg.n_next_steps:
            break
    return {
        "backend": backend, "n_qubits": n_qubits, "marked": marked,
        "iterations": iterations, "next_steps": choices,
    }


def main() -> None:
    cfg = parse_args()
    validate_config(cfg)
    seeds = read_seeds(cfg.seeds_file)
    report = PoliteCrawler(cfg).crawl(seeds)
    selected = select_next_steps(report.buffer, cfg)

    print(f"Seeds: {len(seeds)} | pages fetched: {report.pages_fetched} | visited URLs: {len(report.visited_urls)}")
    print(f"Discovered candidates: {report.discovered_urls} | retained buffer: {len(report.buffer)} | errors: {len(report.errors)}")
    if not report.buffer:
        print("No candidate URLs were discovered; no Grover circuit was run.")
        return

    print(f"Backend: {selected['backend']} | qubits: {selected['n_qubits']} | Grover iterations: {selected['iterations']}")
    print(f"Marked decision-tree indices: {selected['marked']}")
    print("\nCandidate buffer:")
    for index, (link, score) in enumerate(report.buffer):
        print(f"  [{index:>3}] score={score:.3f} depth={link.depth} {link.url}")
    print("\nNext URLs selected by Grover-amplified sampling:")
    for link, probability in selected["next_steps"]:
        print(f"  p={probability:.3f}  {link.text!r} -> {link.url}")


if __name__ == "__main__":
    main()
