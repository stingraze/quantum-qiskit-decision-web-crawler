#!/usr/bin/env python3
"""Polite hybrid web crawler with a classical decision tree, Qiskit-sampled
probabilistic leaf scores, and Grover-amplified next-URL selection.

Install:
  pip install requests beautifulsoup4 numpy scikit-learn qiskit qiskit-aer
Optional JS rendering:
  pip install playwright && playwright install chromium
  
How to run:

uv run quantum_probabilistic_hybrid_crawler.py \
  --seeds-file seeds.txt \
  --max-pages 20 \
  --max-depth 2 \
  --top-k-per-page 5 \
  --buffer-size 16 \
  --n-next-steps 5 \
  --tree-shots 512 \
  --tree-prior-alpha 1.0 \
  --shots 4096 \
  --min-host-delay 1.5
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
    raise SystemExit("Install requests: pip install requests") from exc
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
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; QuantumHybridCrawler/1.0)"
KEYWORDS = ("quantum", "qiskit", "apl", "mainframe", "docs", "middleware", "blog")
TRACKING_PARAMETERS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "_ga", "_gl", "ref", "ref_"}
SKIP_EXTENSION_RE = re.compile(r"\.(?:7z|avi|bin|csv|dmg|docx?|eot|exe|gif|gz|ico|iso|jpe?g|m4[av]|mp[34]|odp|ods|odt|otf|pdf|png|pptx?|rar|tar|tgz|ttf|wav|webm|woff2?|xls[xm]?|zip)$", re.I)

@dataclass
class CrawlConfig:
    seeds_file: str = "seeds.txt"
    max_pages: int = 50
    max_depth: int = 3
    top_k_per_page: int = 8
    buffer_size: int = 32
    n_next_steps: int = 5
    score_threshold: float = 0.5
    shots: int = 4096                 # Grover shots
    tree_shots: int = 512             # Qiskit Bernoulli-score shots per link
    tree_prior_alpha: float = 1.0     # Beta(alpha, alpha) leaf smoothing
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
    p = argparse.ArgumentParser(description="Polite Qiskit-probabilistic hybrid web crawler")
    p.add_argument("--seeds-file", default="seeds.txt"); p.add_argument("--max-pages", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=3); p.add_argument("--top-k-per-page", type=int, default=8)
    p.add_argument("--buffer-size", type=int, default=32); p.add_argument("--n-next-steps", type=int, default=5)
    p.add_argument("--score-threshold", type=float, default=0.5); p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--tree-shots", type=int, default=512); p.add_argument("--tree-prior-alpha", type=float, default=1.0)
    p.add_argument("--allow-external", action="store_true"); p.add_argument("--ignore-robots", action="store_true")
    p.add_argument("--timeout", type=float, default=15.0); p.add_argument("--min-host-delay", type=float, default=1.0)
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT); p.add_argument("--render-javascript", action="store_true")
    p.add_argument("--render-wait-ms", type=int, default=750); p.add_argument("--no-sitemaps", action="store_true")
    p.add_argument("--debug-html-dir"); p.add_argument("--jsonl-output", default="crawl_pages.jsonl")
    a = p.parse_args()
    return CrawlConfig(a.seeds_file, a.max_pages, a.max_depth, a.top_k_per_page, a.buffer_size,
        a.n_next_steps, a.score_threshold, a.shots, a.tree_shots, a.tree_prior_alpha,
        a.allow_external, not a.ignore_robots, a.timeout, a.min_host_delay, a.user_agent,
        a.render_javascript, a.render_wait_ms, not a.no_sitemaps, 5000, a.debug_html_dir, a.jsonl_output)

def validate_config(c: CrawlConfig) -> None:
    if min(c.max_pages, c.top_k_per_page, c.buffer_size, c.n_next_steps, c.shots, c.tree_shots) < 1: raise ValueError("page, buffer, step, and shot counts must be at least 1")
    if c.max_depth < 0 or c.min_host_delay < 0: raise ValueError("depth and delay must be non-negative")
    if not 0 <= c.score_threshold <= 1: raise ValueError("--score-threshold must be in [0, 1]")
    if c.tree_prior_alpha <= 0: raise ValueError("--tree-prior-alpha must be positive")

def canonicalize(url: str) -> str | None:
    p = urlparse(urldefrag(url).url)
    if p.scheme.lower() not in ("http", "https") or not p.netloc: return None
    host = (p.hostname or "").lower(); netloc = host
    if p.port and not ((p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443)): netloc = f"{host}:{p.port}"
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMETERS and not k.lower().startswith("utm_")]
    return urlunparse((p.scheme.lower(), netloc, re.sub(r"/{2,}", "/", p.path or "/"), "", urlencode(sorted(query)), ""))

def read_seeds(path: str) -> list[str]:
    seen, seeds = set(), []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if value and not value.startswith("#"):
            url = canonicalize(value)
            if not url: print(f"[warn] invalid seed at {path}:{n}: {value!r}")
            elif url not in seen: seen.add(url); seeds.append(url)
    if not seeds: raise ValueError(f"No valid HTTP(S) seed URLs in {path!r}")
    return seeds

# --- Decision tree -> smoothed probability -> Qiskit Bernoulli sampling ---
def featurize(link: Link) -> list[int]:
    p = urlparse(link.url); path, text = p.path.lower(), link.text.lower()
    return [path.strip("/").count("/") + int(bool(path.strip("/"))), sum(w in path or w in text for w in KEYWORDS), min(len(link.text), 200), int(link.same_domain), int(bool(SKIP_EXTENSION_RE.search(path)))]

def synthetic_training_set() -> tuple[list[list[int]], list[int]]:
    return ([[1,3,20,1,0],[1,2,25,1,0],[1,0,8,1,0],[0,0,9,0,0],[1,2,28,1,0],[1,0,14,1,0],[0,1,5,0,1]], [1,1,0,0,1,0,0])

def rule_based_score(x: list[int]) -> float:
    _, hits, length, same, binary = x
    return float(np.clip(.40*same + .15*min(hits,3) + .10*(length > 12) - .50*binary, 0, 1))

def smoothed_tree_probabilities(tree: DecisionTreeClassifier, X: list[list[int]], alpha: float) -> np.ndarray:
    classes = list(tree.classes_)
    if 1 not in classes: return np.zeros(len(X))
    positive = classes.index(1); leaves = tree.apply(X); out = np.empty(len(X))
    for i, leaf in enumerate(leaves):
        counts = tree.tree_.value[leaf][0]; out[i] = (counts[positive] + alpha) / (counts.sum() + 2 * alpha)
    return out

def qiskit_bernoulli_estimate(p: float, shots: int) -> float:
    p = float(np.clip(p, 0, 1))
    if not HAS_QISKIT: return p
    qc = QuantumCircuit(1, 1)
    qc.ry(2 * math.asin(math.sqrt(p)), 0) # P(measure 1) = p
    qc.measure(0, 0)
    return AerSimulator().run(qc, shots=shots).result().get_counts().get("1", 0) / shots

def score_links(links: list[Link], cfg: CrawlConfig) -> list[float]:
    if not links: return []
    X = [featurize(x) for x in links]
    if not HAS_SKLEARN: return [rule_based_score(x) for x in X]
    train_X, y = synthetic_training_set()
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=0).fit(train_X, y)
    return [qiskit_bernoulli_estimate(p, cfg.tree_shots) for p in smoothed_tree_probabilities(tree, X, cfg.tree_prior_alpha)]

class PersistentRenderer:
    def __init__(self, ua: str, timeout_ms: int, wait_ms: int): self.ua, self.timeout_ms, self.wait_ms, self.pw, self.browser = ua, timeout_ms, wait_ms, None, None
    def content(self, url: str) -> tuple[str, str]:
        if not HAS_PLAYWRIGHT: raise RuntimeError("Install playwright and run: playwright install chromium")
        if self.browser is None: self.pw = sync_playwright().start(); self.browser = self.pw.chromium.launch(headless=True)
        context = self.browser.new_context(user_agent=self.ua); page = context.new_page()
        try: page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms); page.wait_for_timeout(self.wait_ms); return page.content(), page.url
        finally: context.close()
    def close(self) -> None:
        if self.browser: self.browser.close()
        if self.pw: self.pw.stop()

class PoliteCrawler:
    def __init__(self, cfg: CrawlConfig):
        self.cfg, self.robots, self.last, self.errors, self.output = cfg, {}, defaultdict(float), [], None
        self.session = self._session(); self.renderer = PersistentRenderer(cfg.user_agent, int(cfg.request_timeout*1000), cfg.render_wait_ms) if cfg.render_javascript else None
    def _session(self):
        s = requests.Session(); retry = Retry(total=3, connect=3, read=3, status=3, backoff_factor=.5, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset(("GET","HEAD")))
        a = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10); s.mount("http://", a); s.mount("https://", a); s.headers.update({"User-Agent":self.cfg.user_agent,"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}); return s
    @staticmethod
    def host_key(url):
        p=urlparse(url); return f"{p.scheme}://{p.netloc}"
    def _wait(self, url):
        host=self.host_key(url); delay=self.cfg.min_host_delay-(time.monotonic()-self.last[host])
        if delay>0: time.sleep(delay)
        self.last[host]=time.monotonic()
    def _robots(self, url):
        origin=self.host_key(url)
        if origin not in self.robots:
            rp=RobotFileParser()
            try: self._wait(origin); r=self.session.get(origin+"/robots.txt", timeout=self.cfg.request_timeout); rp.parse(r.text.splitlines()) if r.ok else setattr(rp,"allow_all",True)
            except requests.RequestException: rp.allow_all=True
            self.robots[origin]=rp
        return self.robots[origin]
    def allowed(self, url): return not self.cfg.respect_robots or self._robots(url).can_fetch(self.cfg.user_agent,url)
    def fetch_html(self, url):
        self._wait(url); r=self.session.get(url, timeout=self.cfg.request_timeout, allow_redirects=True); r.raise_for_status()
        if "html" not in r.headers.get("Content-Type", "").lower() and "xhtml" not in r.headers.get("Content-Type", "").lower(): raise ValueError("not HTML")
        return r.text, canonicalize(r.url) or r.url
    @staticmethod
    def extract_links(html, base, depth):
        soup=BeautifulSoup(html,"html.parser"); host=(urlparse(base).hostname or "").lower(); out=[]; seen=set()
        for e in soup.select("a[href], area[href]"):
            h=e.get("href","").strip()
            if not h or h.lower().startswith(("javascript:","mailto:","tel:","data:")): continue
            u=canonicalize(urljoin(base,h))
            if u and u not in seen and not SKIP_EXTENSION_RE.search(urlparse(u).path): seen.add(u); out.append(Link(u,e.get_text(" ",strip=True) or e.get("aria-label","") or e.get("alt", ""),depth,(urlparse(u).hostname or "").lower()==host))
        return out
    def sitemap_urls(self, seed):
        if not self.cfg.discover_sitemaps:return []
        pending=list(set((self._robots(seed).site_maps() or [])+[self.host_key(seed)+"/sitemap.xml"])); seen=set(); out=[]
        while pending and len(out)<self.cfg.sitemap_url_limit:
            sm=pending.pop()
            if sm in seen:continue
            seen.add(sm)
            try:
                self._wait(sm); r=self.session.get(sm,timeout=self.cfg.request_timeout)
                if not r.ok:continue
                root=ET.fromstring(r.content); loc=[x.text.strip() for x in root.findall(".//{*}loc") if x.text]
                if root.tag.rsplit("}",1)[-1].lower()=="sitemapindex":pending.extend(loc)
                else: out.extend(u for x in loc if (u:=canonicalize(x)))
            except Exception: pass
        return out[:self.cfg.sitemap_url_limit]
    def log(self, d):
        if self.output:self.output.write(json.dumps(d,ensure_ascii=False)+"\n");self.output.flush()
    def crawl(self,seeds:Iterable[str])->CrawlReport:
        seeds=[x for x in (canonicalize(s) for s in seeds) if x]; allowed_hosts={(urlparse(s).hostname or "").lower() for s in seeds}; frontier=[]; queued=set();visited=set(); candidates={};seq=0
        def enqueue(url,depth,score=0.0):
            nonlocal seq
            url=canonicalize(url); host=(urlparse(url).hostname or "").lower() if url else ""
            if not url or depth>self.cfg.max_depth or url in queued|visited or (not self.cfg.allow_external and host not in allowed_hosts) or SKIP_EXTENSION_RE.search(urlparse(url).path):return
            queued.add(url);heapq.heappush(frontier,(-score,depth,seq,url));seq+=1
        for s in seeds:
            enqueue(s,0,1.0)
            for u in self.sitemap_urls(s):enqueue(u,1,.05)
        if self.cfg.jsonl_output:self.output=open(self.cfg.jsonl_output,"w",encoding="utf-8")
        pages=0
        try:
            while frontier and pages<self.cfg.max_pages:
                _,depth,_,requested=heapq.heappop(frontier);queued.discard(requested)
                if requested in visited:continue
                if not self.allowed(requested):visited.add(requested);self.log({"url":requested,"depth":depth,"status":"robots_disallowed"});continue
                try: html,final=self.fetch_html(requested);pages+=1;visited.update((requested,final))
                except Exception as e: visited.add(requested);err={"url":requested,"depth":depth,"error":str(e)};self.errors.append(err);self.log({"status":"fetch_error",**err});continue
                links=self.extract_links(html,final,depth+1);source="static"
                if not links and self.renderer:
                    try: html,final=self.renderer.content(final);final=canonicalize(final) or final;links=self.extract_links(html,final,depth+1);source="rendered"
                    except Exception as e:self.errors.append({"url":final,"depth":depth,"error":f"render error: {e}"})
                self.log({"url":final,"requested_url":requested,"depth":depth,"status":"ok","source":source,"link_count":len(links)})
                ranked=sorted(zip(links,score_links(links,self.cfg)),key=lambda x:x[1],reverse=True)
                for link,score in ranked:
                    if link.url not in candidates or score>candidates[link.url][1]:candidates[link.url]=(link,float(score))
                if depth<self.cfg.max_depth:
                    for link,score in ranked[:self.cfg.top_k_per_page]:enqueue(link.url,depth+1,score)
        finally:
            if self.output:self.output.close()
            if self.renderer:self.renderer.close()
            self.session.close()
        ranked=sorted(candidates.values(),key=lambda x:x[1],reverse=True)
        return CrawlReport(ranked[:self.cfg.buffer_size],pages,visited,self.errors,len(candidates))

# --- Grover selection ---
def next_power_of_two(n): return 1<<max(1,(n-1).bit_length())
def marked_indices(scores,threshold):
    marked=[i for i,x in enumerate(scores) if x>=threshold];return marked or [int(np.argmax(scores))]
def optimal_iterations(n,m): return 0 if not 0<m<n else max(1,round(math.pi*math.sqrt(n/m)/4))
def oracle(n,marked):
    qc=QuantumCircuit(n,name="priority_oracle")
    for index in marked:
        zeros=[q for q,b in enumerate(format(index,f"0{n}b")[::-1]) if b=="0"];qc.x(zeros)
        if n==1:qc.z(0)
        else: target=n-1;qc.h(target);qc.mcx(list(range(target)),target);qc.h(target)
        qc.x(zeros)
    return qc
def diffuser(n):
    qc=QuantumCircuit(n,name="diffuser");qc.h(range(n));qc.x(range(n))
    if n==1:qc.z(0)
    else: target=n-1;qc.h(target);qc.mcx(list(range(target)),target);qc.h(target)
    qc.x(range(n));qc.h(range(n));return qc
def select_next_steps(buffer,cfg):
    if not buffer:return {"backend":None,"n_qubits":0,"marked":[],"iterations":0,"next_steps":[]}
    links=[x for x,_ in buffer];scores=[x for _,x in buffer];states=next_power_of_two(len(links));n=states.bit_length()-1;marked=marked_indices(scores+[0.]*(states-len(scores)),cfg.score_threshold);iters=optimal_iterations(states,len(marked))
    if HAS_QISKIT:
        qc=QuantumCircuit(n,n);qc.h(range(n));o,d=oracle(n,marked),diffuser(n)
        for _ in range(iters):qc.compose(o,inplace=True);qc.compose(d,inplace=True)
        qc.measure(range(n),range(n));counts=AerSimulator().run(qc,shots=cfg.shots).result().get_counts();probs={int(k,2):v/cfg.shots for k,v in counts.items()};backend="qiskit_aer"
    else:
        state=np.ones(states,dtype=complex)/math.sqrt(states);mask=np.zeros(states,dtype=bool);mask[marked]=True
        for _ in range(iters):state[mask]*=-1;state=2*state.mean()-state
        probs=dict(enumerate(np.abs(state)**2));backend="numpy_fallback"
    choices=[]
    for i,p in sorted(probs.items(),key=lambda x:x[1],reverse=True):
        if i<len(links):choices.append((links[i],p))
        if len(choices)>=cfg.n_next_steps:break
    return {"backend":backend,"n_qubits":n,"marked":marked,"iterations":iters,"next_steps":choices}

def main():
    cfg=parse_args();validate_config(cfg);seeds=read_seeds(cfg.seeds_file);report=PoliteCrawler(cfg).crawl(seeds);selected=select_next_steps(report.buffer,cfg)
    print(f"Seeds: {len(seeds)} | pages fetched: {report.pages_fetched} | visited URLs: {len(report.visited_urls)}")
    print(f"Discovered candidates: {report.discovered_urls} | retained buffer: {len(report.buffer)} | errors: {len(report.errors)}")
    if not report.buffer:print("No candidate URLs were discovered; no Grover circuit was run.");return
    print(f"Backend: {selected['backend']} | qubits: {selected['n_qubits']} | Grover iterations: {selected['iterations']}")
    print(f"Marked probabilistic-tree indices: {selected['marked']}")
    print("\nCandidate buffer:")
    for i,(link,score) in enumerate(report.buffer):print(f"  [{i:>3}] score={score:.3f} depth={link.depth} {link.url}")
    print("\nNext URLs selected by Grover-amplified sampling:")
    for link,p in selected["next_steps"]:print(f"  p={p:.3f}  {link.text!r} -> {link.url}")
if __name__=="__main__":main()
