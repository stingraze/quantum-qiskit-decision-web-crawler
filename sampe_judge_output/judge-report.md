# Quantum Crawler AI Judge Report

> Input: `comparison.json`  
> Model: `llama3.1`  
> Generated: 2026-03-26T07:33:55Z

---

## Computed Statistics

### Algorithm: QUANTUM
- Pages visited: 11
- Pages skipped/failed: 2
- Total attempts: 13
- Success rate: 84.6%
- Unique URLs: 13
- Unique domains: 1
- Same-host ratio: 100.0%
- Path diversity ratio: 0.92
- Avg out-links found per page: 119.3
- Avg out-links enqueued per page: 33.9
- Hub pages (≥20 out-links): 10
- Leaf pages (<5 out-links): 1
- Avg fetch time: 0.866s
- Max depth reached: 0

Top 10 domains:
  - `www.dwavequantum.com` (13 visits)

### Algorithm: DFS
- Pages visited: 13
- Pages skipped/failed: 5
- Total attempts: 18
- Success rate: 72.2%
- Unique URLs: 18
- Unique domains: 2
- Same-host ratio: 5.6%
- Path diversity ratio: 1.00
- Avg out-links found per page: 50.3
- Avg out-links enqueued per page: 42.2
- Hub pages (≥20 out-links): 2
- Leaf pages (<5 out-links): 0
- Avg fetch time: 0.200s
- Max depth reached: 2

Top 10 domains:
  - `www.youtube.com` (17 visits)
  - `www.dwavequantum.com` (1 visits)

### Algorithm: BFS
- Pages visited: 11
- Pages skipped/failed: 2
- Total attempts: 13
- Success rate: 84.6%
- Unique URLs: 13
- Unique domains: 1
- Same-host ratio: 100.0%
- Path diversity ratio: 0.92
- Avg out-links found per page: 119.3
- Avg out-links enqueued per page: 33.9
- Hub pages (≥20 out-links): 10
- Leaf pages (<5 out-links): 1
- Avg fetch time: 0.998s
- Max depth reached: 0

Top 10 domains:
  - `www.dwavequantum.com` (13 visits)


---

## AI Judge Analysis

**URL Uniqueness**

### QUANTUM
- Unique domains: 1
- Same-host ratio: 100.0% (fraction of URLs on seed domain(s))
- Path diversity ratio: 0.92 (unique paths / unique URLs)
- Ratio of unique URLs to total URLs attempted: 84.6%

The QUANTUM algorithm tends towards **same-host bias**, as it only visited one domain and stayed within the same host throughout its crawl.

### DFS
- Unique domains: 2
- Same-host ratio: 5.6% (fraction of URLs on seed domain(s))
- Path diversity ratio: 1.00 (unique paths / unique URLs)
- Ratio of unique URLs to total URLs attempted: 72.2%

The DFS algorithm is biased towards **exploring widely across different hosts**, as it visited two domains and had a relatively low same-host ratio.

### BFS
- Unique domains: 1
- Same-host ratio: 100.0% (fraction of URLs on seed domain(s))
- Path diversity ratio: 0.92 (unique paths / unique URLs)
- Ratio of unique URLs to total URLs attempted: 84.6%

The BFS algorithm also tends towards **same-host bias**, similar to QUANTUM, as it only visited one domain and stayed within the same host.

**Data Importance / Valuable Content**

### QUANTUM
- Page titles and snippets suggest a focus on substantive content related to quantum computing.
- Many page titles are informative and descriptive of the content, while snippet samples often contain links to learn more or explore products.

The QUANTUM algorithm tends towards **finding valuable news, research, documentation, and substantive information**.

### DFS
- Page titles and snippets show a mix of informative and boilerplate/navigation/error pages.
- Snippet samples often contain error messages or navigation links rather than substantive content.

The DFS algorithm is biased towards **finding more informational pages vs. boilerplate/navigation/error pages**, but with less success in finding valuable content compared to QUANTUM.

### BFS
- Page titles and snippets are similar to those of the QUANTUM algorithm, suggesting a focus on substantive content related to quantum computing.
- Many page titles are informative and descriptive of the content, while snippet samples often contain links to learn more or explore products.

The BFS algorithm also tends towards **finding valuable news, research, documentation, and substantive information**, similar to QUANTUM.

**Link Variety**

### QUANTUM
- Avg out-links found per page: 119.3
- Max out-links found on a single page: 414
- Hub pages (≥20 out-links): 10

The QUANTUM algorithm tends towards **discovering "hub" pages with many links**, as it finds an average of 119.3 out-links per page and has a high number of hub pages.

### DFS
- Avg out-links found per page: 50.3
- Max out-links found on a single page: 511
- Hub pages (≥20 out-links): 2

The DFS algorithm is biased towards **discovering "leaf" pages with few links**, as it finds an average of 50.3 out-links per page and has relatively few hub pages.

### BFS
- Avg out-links found per page: 119.3
- Max out-links found on a single page: 414
- Hub pages (≥20 out-links): 10

The BFS algorithm also tends towards **discovering "hub" pages with many links**, similar to QUANTUM.

**Crawl Efficiency**

### QUANTUM
- Pages visited (ok): 11
- Success rate: 84.6%
- Avg fetch time per page: 0.866s
- Max depth reached: 0

The QUANTUM algorithm has a relatively high success rate and efficient average fetch time, but does not reach great depths.

### DFS
- Pages visited (ok): 13
- Success rate: 72.2%
- Avg fetch time per page: 0.200s
- Max depth reached: 2

The DFS algorithm is biased towards **discovering new content quickly**, as it has a relatively high success rate and efficient average fetch time, but does not reach great depths.

### BFS
- Pages visited (ok): 11
- Success rate: 84.6%
- Avg fetch time per page: 0.998s
- Max depth reached: 0

The BFS algorithm also has a relatively high success rate and efficient average fetch time, but does not reach great depths.

**Overall Comparative Summary**

Based on the analysis above:

* **QUANTUM wins in URL Uniqueness**, as it tends towards same-host bias.
* **DFS wins in Crawl Efficiency**, as it discovers new content quickly with a relatively high success rate and efficient average fetch time.
* **BFS ties with QUANTUM in Data Importance / Valuable Content** and **Link Variety**, as both algorithms tend towards finding valuable news, research, documentation, and substantive information, and discovering "hub" pages with many links.

Overall, the **QUANTUM algorithm performed best overall**, due to its ability to find valuable content and discover new links efficiently. However, the DFS algorithm excels in crawl efficiency, making it a preferable choice when speed is crucial. The BFS algorithm is also a strong contender, but its performance is more similar to QUANTUM's.

---

*Generated by ollama-judge.py | Model: llama3.1 | Timestamp: 2026-03-26T07:33:55Z*
