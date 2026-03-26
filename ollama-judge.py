#!/usr/bin/env python3
"""
(C)Tsubasa Kato - Inspire Search Corp.
Created with help of AI assistants.
Contact: tsubasa@inspiresearch.io

ollama-judge.py -- AI judge powered by a local Ollama model.

Reads the ``comparison.json`` file produced by the ``--compare-algorithms``
mode of quantum-decision-crawler4.py / quantum-decision-crawler5.py and uses
a local Ollama model to analyse and compare the crawl results across the three
algorithms: quantum, BFS, and DFS.

Judging criteria
----------------
1. URL Uniqueness      -- domain / path diversity, same-host bias
2. Data Importance     -- content quality signals from titles and snippets
3. Link Variety        -- hub pages vs. leaf pages, broad vs. narrow discovery
4. Crawl Efficiency    -- success rate, avg fetch time, depth reached, throughput

Usage
-----
python ollama-judge.py --input comparison.json --output judge-report.md --model llama3.1
python ollama-judge.py --input comparison.json --verbose --model mistral
"""

import argparse
import datetime
import json
import logging
import sys
from collections import Counter
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ollama_judge")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Pages with at least this many outbound links are classified as "hub" pages.
HUB_PAGE_THRESHOLD: int = 20

#: Pages with fewer than this many outbound links are classified as "leaf" pages.
LEAF_PAGE_THRESHOLD: int = 5

#: Maximum characters of Ollama error response text to include in log messages.
MAX_ERROR_TEXT_LEN: int = 500

# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _host(url: str) -> str:
    """Return the netloc (host) portion of *url*, or empty string on error."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _path(url: str) -> str:
    """Return the path portion of *url*, or empty string on error."""
    try:
        return urlparse(url).path
    except Exception:
        return ""


def compute_stats(
    records: List[Dict[str, Any]],
    summaries: Dict[str, Any],
    algorithm: str,
    seeds: List[str],
    max_title_samples: int,
    max_snippet_samples: int,
    snippet_max_len: int,
) -> Dict[str, Any]:
    """Compute per-algorithm statistics from the raw records list."""
    algo_records = [r for r in records if r.get("algorithm") == algorithm]
    ok_records = [r for r in algo_records if r.get("status") == "ok"]
    skip_records = [r for r in algo_records if r.get("status") == "skip"]

    total = len(algo_records)
    visited = len(ok_records)
    skipped = len(skip_records)
    success_rate = visited / total if total else 0.0

    all_urls = [r.get("url", "") or r.get("final_url", "") for r in algo_records]
    all_urls = [u for u in all_urls if u]
    unique_urls = set(all_urls)
    hosts = [_host(u) for u in all_urls if _host(u)]
    host_counts: Counter[str] = Counter(hosts)
    unique_domains = len(host_counts)
    top_domains = host_counts.most_common(10)

    seed_hosts = {_host(s) for s in seeds if _host(s)}
    same_host_count = sum(1 for h in hosts if h in seed_hosts)
    same_host_ratio = same_host_count / len(hosts) if hosts else 0.0

    unique_paths = {_path(u) for u in all_urls if _path(u)}
    path_diversity = len(unique_paths) / len(unique_urls) if unique_urls else 0.0

    out_links_found = [r.get("out_links_found", 0) or 0 for r in ok_records]
    out_links_enqueued = [r.get("out_links_enqueued", 0) or 0 for r in ok_records]
    avg_out_links_found = (
        sum(out_links_found) / len(out_links_found) if out_links_found else 0.0
    )
    avg_out_links_enqueued = (
        sum(out_links_enqueued) / len(out_links_enqueued) if out_links_enqueued else 0.0
    )
    max_out_links_found = max(out_links_found) if out_links_found else 0
    hub_pages = sum(1 for v in out_links_found if v >= HUB_PAGE_THRESHOLD)
    leaf_pages = sum(1 for v in out_links_found if v < LEAF_PAGE_THRESHOLD)

    fetch_times = [r.get("fetch_seconds", None) for r in ok_records]
    fetch_times = [f for f in fetch_times if f is not None]
    avg_fetch_time = sum(fetch_times) / len(fetch_times) if fetch_times else 0.0

    depths = [r.get("depth", 0) or 0 for r in ok_records]
    max_depth = max(depths) if depths else 0

    # Duration from summaries (may not always be present)
    summary = summaries.get(algorithm, {})
    duration = None  # not directly in records; caller may pass via summary

    titles: List[str] = []
    for r in ok_records:
        t = r.get("title", "") or ""
        t = t.strip()
        if t and len(titles) < max_title_samples:
            titles.append(t)

    snippets: List[str] = []
    for r in ok_records:
        s = r.get("snippet", "") or ""
        s = s.strip()[:snippet_max_len]
        if s and len(snippets) < max_snippet_samples:
            snippets.append(s)

    return {
        "algorithm": algorithm,
        "total_attempts": total,
        "pages_visited": visited,
        "pages_skipped": skipped,
        "success_rate": success_rate,
        "unique_urls": len(unique_urls),
        "unique_domains": unique_domains,
        "same_host_ratio": same_host_ratio,
        "path_diversity_ratio": path_diversity,
        "top_domains": top_domains,
        "avg_out_links_found": avg_out_links_found,
        "avg_out_links_enqueued": avg_out_links_enqueued,
        "max_out_links_found": max_out_links_found,
        "hub_pages_count": hub_pages,
        "leaf_pages_count": leaf_pages,
        "avg_fetch_time_s": avg_fetch_time,
        "max_depth_reached": max_depth,
        "title_samples": titles,
        "snippet_samples": snippets,
        # Pass through pre-computed summary fields when available
        "summary_pages_visited": summary.get("pages_visited"),
        "summary_total_links_found": summary.get("total_links_found"),
        "summary_avg_fetch_seconds": summary.get("avg_fetch_seconds"),
        "summary_max_depth_reached": summary.get("max_depth_reached"),
        "summary_avg_priority_score": summary.get("avg_priority_score"),
    }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _stats_block(stats: Dict[str, Any]) -> str:
    """Return a human-readable block for one algorithm's statistics."""
    top_domains_str = ", ".join(
        f"{d}({c})" for d, c in stats["top_domains"]
    )
    titles_str = "\n".join(f"  - {t}" for t in stats["title_samples"]) or "  (none)"
    snippets_str = "\n".join(
        f"  [{i+1}] {s}" for i, s in enumerate(stats["snippet_samples"])
    ) or "  (none)"

    return f"""### Algorithm: {stats['algorithm'].upper()}
- Total attempts: {stats['total_attempts']}
- Pages visited (ok): {stats['pages_visited']}
- Pages skipped/failed: {stats['pages_skipped']}
- Success rate: {stats['success_rate']:.1%}
- Unique URLs: {stats['unique_urls']}
- Unique domains: {stats['unique_domains']}
- Same-host ratio: {stats['same_host_ratio']:.1%}  (fraction of URLs on seed domain(s))
- Path diversity ratio: {stats['path_diversity_ratio']:.2f}  (unique paths / unique URLs)
- Top 10 domains: {top_domains_str or '(none)'}
- Avg out-links found per page: {stats['avg_out_links_found']:.1f}
- Avg out-links enqueued per page: {stats['avg_out_links_enqueued']:.1f}
- Max out-links found on a single page: {stats['max_out_links_found']}
- Hub pages (≥20 out-links): {stats['hub_pages_count']}
- Leaf pages (<5 out-links): {stats['leaf_pages_count']}
- Avg fetch time per page: {stats['avg_fetch_time_s']:.3f}s
- Max depth reached: {stats['max_depth_reached']}

Page title samples (up to {len(stats['title_samples'])}):
{titles_str}

Snippet samples (up to {len(stats['snippet_samples'])}):
{snippets_str}
"""


def build_prompt(all_stats: List[Dict[str, Any]]) -> str:
    """Build the structured prompt to send to Ollama."""
    stats_sections = "\n".join(_stats_block(s) for s in all_stats)
    algo_names = ", ".join(s["algorithm"].upper() for s in all_stats)

    return f"""You are an expert web-crawl analyst. You have been given statistics and content samples from a three-way comparison of web-crawling algorithms ({algo_names}). Your task is to act as an impartial judge and produce a detailed, structured analysis.

## Crawl Statistics

{stats_sections}

## Judging Instructions

Analyse the results across the following four criteria. For each criterion, do NOT produce simple "good/bad" or "positive/negative" verdicts — instead describe **which direction each algorithm is biased toward** and explain why, using the statistics above as evidence.

### Criterion 1: URL Uniqueness
- How many unique domains/hosts were visited by each algorithm?
- How diverse are the URL paths (not just domains, but path diversity within domains)?
- Does the algorithm tend to stay on the same host (same-host bias) or explore widely across different hosts?
- What is the ratio of unique URLs to total URLs attempted for each algorithm?

### Criterion 2: Data Importance / Valuable Content
- Based on the page titles and snippets, does each algorithm tend to find pages with valuable news, research, documentation, or substantive information?
- Does it find more informational pages vs. boilerplate/navigation/error pages?
- Analyse the titles and snippets for signals of substantive content.

### Criterion 3: Link Variety
- How many outbound links did pages found by each algorithm contain on average?
- Are the discovered pages biased toward "hub" pages (many links, navigation/directory) or "leaf" pages (few links, content-focused)?
- What does the distribution of out_links_found and out_links_enqueued tell us about each algorithm's discovery pattern?
- Again: this is NOT about positive vs. negative — just describe which way each algorithm is biased.

### Criterion 4: Crawl Efficiency
- How many pages were successfully visited vs. skipped/failed?
- What is the average fetch time per page for each algorithm?
- How deep did each algorithm reach?
- Which algorithm appears to discover new content most quickly?

### Overall Comparative Summary
Provide a final comparative summary that:
1. Names a winner (or declares a tie) for each of the four criteria above.
2. Gives an overall assessment of which algorithm performed best overall, and under what circumstances each algorithm would be preferable.

Format your entire response in Markdown with clear section headers matching the four criteria and the overall summary section.
"""


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------

def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float,
    timeout: int = 120,
) -> str:
    """Send *prompt* to Ollama and return the full response text."""
    api_url = ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    logger.info("[JUDGE] Sending request to Ollama (%s) model=%s ...", api_url, model)
    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        logger.error(
            "[JUDGE] Cannot connect to Ollama at %s — is Ollama running? (%s)",
            ollama_url,
            exc,
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        logger.error(
            "[JUDGE] Ollama request timed out after %d seconds.", timeout
        )
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        logger.error("[JUDGE] Unexpected request error calling Ollama: %s", exc)
        sys.exit(1)

    if resp.status_code == 404:
        logger.error(
            "[JUDGE] Model '%s' not found in Ollama (HTTP 404). "
            "Pull it first with: ollama pull %s",
            model,
            model,
        )
        sys.exit(1)

    if not resp.ok:
        logger.error(
            "[JUDGE] Ollama returned HTTP %d: %s",
            resp.status_code,
            resp.text[:MAX_ERROR_TEXT_LEN],
        )
        sys.exit(1)

    try:
        data = resp.json()
    except Exception as exc:
        logger.error("[JUDGE] Failed to parse Ollama JSON response: %s", exc)
        sys.exit(1)

    response_text = data.get("response", "")
    if not response_text:
        logger.warning("[JUDGE] Ollama returned an empty response.")
    return response_text


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    all_stats: List[Dict[str, Any]],
    analysis: str,
    model: str,
    input_file: str,
) -> str:
    """Compose the final Markdown report."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    stats_md_parts = []
    for s in all_stats:
        top_domains_md = "\n".join(
            f"  - `{d}` ({c} visits)" for d, c in s["top_domains"]
        ) or "  - (none)"
        stats_md_parts.append(
            f"""### Algorithm: {s['algorithm'].upper()}
- Pages visited: {s['pages_visited']}
- Pages skipped/failed: {s['pages_skipped']}
- Total attempts: {s['total_attempts']}
- Success rate: {s['success_rate']:.1%}
- Unique URLs: {s['unique_urls']}
- Unique domains: {s['unique_domains']}
- Same-host ratio: {s['same_host_ratio']:.1%}
- Path diversity ratio: {s['path_diversity_ratio']:.2f}
- Avg out-links found per page: {s['avg_out_links_found']:.1f}
- Avg out-links enqueued per page: {s['avg_out_links_enqueued']:.1f}
- Hub pages (≥20 out-links): {s['hub_pages_count']}
- Leaf pages (<5 out-links): {s['leaf_pages_count']}
- Avg fetch time: {s['avg_fetch_time_s']:.3f}s
- Max depth reached: {s['max_depth_reached']}

Top 10 domains:
{top_domains_md}
"""
        )

    stats_section = "\n".join(stats_md_parts)

    return f"""# Quantum Crawler AI Judge Report

> Input: `{input_file}`  
> Model: `{model}`  
> Generated: {timestamp}

---

## Computed Statistics

{stats_section}

---

## AI Judge Analysis

{analysis}

---

*Generated by ollama-judge.py | Model: {model} | Timestamp: {timestamp}*
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, load data, compute stats, call Ollama, write report."""
    parser = argparse.ArgumentParser(
        description="AI judge for quantum-crawler comparison results using a local Ollama model."
    )
    parser.add_argument(
        "--input",
        default="comparison.json",
        metavar="FILE",
        help="Path to comparison.json (default: comparison.json)",
    )
    parser.add_argument(
        "--output",
        default="judge-report.md",
        metavar="FILE",
        help="Path to output report (default: judge-report.md)",
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        metavar="MODEL",
        help="Ollama model name (default: llama3.1)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--max-title-samples",
        type=int,
        default=20,
        metavar="N",
        help="Maximum title samples per algorithm (default: 20)",
    )
    parser.add_argument(
        "--max-snippet-samples",
        type=int,
        default=10,
        metavar="N",
        help="Maximum snippet samples per algorithm (default: 10)",
    )
    parser.add_argument(
        "--snippet-max-len",
        type=int,
        default=200,
        metavar="N",
        help="Max characters per snippet sample (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="LLM sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full prompt sent to Ollama",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Timeout for Ollama API calls in seconds (default: 120)",
    )

    args = parser.parse_args()

    # ---- Load comparison.json ----
    logger.info("[JUDGE] Loading input file: %s", args.input)
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        logger.error("[JUDGE] Input file not found: %s", args.input)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logger.error("[JUDGE] Failed to parse JSON from %s: %s", args.input, exc)
        sys.exit(1)

    # ---- Validate structure ----
    records: List[Dict[str, Any]] = data.get("records", [])
    summaries: Dict[str, Any] = data.get("summaries", {})
    comparison_run: Dict[str, Any] = data.get("comparison_run", {})

    if not records:
        logger.error(
            "[JUDGE] 'records' key is missing or empty in %s — nothing to judge.",
            args.input,
        )
        sys.exit(1)

    algorithms: List[str] = comparison_run.get(
        "algorithms_run", list(summaries.keys())
    )
    if not algorithms:
        # Fall back to unique algorithm values in records
        algorithms = sorted({r.get("algorithm", "") for r in records if r.get("algorithm")})
    if not algorithms:
        logger.error("[JUDGE] Cannot determine algorithms from input file.")
        sys.exit(1)

    seeds: List[str] = comparison_run.get("seeds", [])
    logger.info("[JUDGE] Algorithms detected: %s", algorithms)
    logger.info("[JUDGE] Total records: %d", len(records))

    # ---- Compute statistics ----
    all_stats: List[Dict[str, Any]] = []
    for algo in algorithms:
        logger.info("[JUDGE] Computing statistics for algorithm: %s", algo)
        stats = compute_stats(
            records=records,
            summaries=summaries,
            algorithm=algo,
            seeds=seeds,
            max_title_samples=args.max_title_samples,
            max_snippet_samples=args.max_snippet_samples,
            snippet_max_len=args.snippet_max_len,
        )
        all_stats.append(stats)
        logger.info(
            "[JUDGE]   %s: visited=%d unique_domains=%d success_rate=%.1f%%",
            algo,
            stats["pages_visited"],
            stats["unique_domains"],
            stats["success_rate"] * 100,
        )

    # ---- Build prompt ----
    prompt = build_prompt(all_stats)

    if args.verbose:
        print("\n" + "=" * 70)
        print("PROMPT SENT TO OLLAMA:")
        print("=" * 70)
        print(prompt)
        print("=" * 70 + "\n")

    # ---- Call Ollama ----
    analysis = call_ollama(
        prompt=prompt,
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    logger.info("[JUDGE] Received analysis (%d characters).", len(analysis))

    # ---- Compose report ----
    report = format_report(
        all_stats=all_stats,
        analysis=analysis,
        model=args.model,
        input_file=args.input,
    )

    # ---- Print to stdout ----
    print(report)

    # ---- Write report file ----
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("[JUDGE] Report written to: %s", args.output)
    except OSError as exc:
        logger.error("[JUDGE] Failed to write report to %s: %s", args.output, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
