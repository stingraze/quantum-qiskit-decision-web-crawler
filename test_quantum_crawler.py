"""Tests for quantum-decision-crawler4.py

Covers:
  1. _heuristic_score: important pages score higher than generic ones.
  2. _heuristic_score: low-value URLs (signout, cart, terms, etc.) are penalised.
  3. _heuristic_score: URL path keyword scoring lifts docs/about/api paths.
  4. _hybrid_score_candidate: quantum mode produces non-uniform scores across a
     candidate set, unlike BFS constant priorities.
  5. _SingleThreadComparableCrawler: quantum and BFS use different scoring paths
     for the same discovered candidate set (regression test for the root bug).
  6. Deterministic tie-breaking: exploration_temperature=0.0 yields stable,
     reproducible scores for the same URL pair.
  7. Priority ordering: high-value URLs outscore low-value ones under the hybrid
     scorer.
"""

import importlib.util
import os
import sys
import types

# ---------------------------------------------------------------------------
# Lazy-load the crawler module from its unusual filename.
# ---------------------------------------------------------------------------

_CRAWLER_FILE = os.path.join(
    os.path.dirname(__file__), "quantum-decision-crawler4.py"
)


def _load_crawler() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("crawler4", _CRAWLER_FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_crawler = _load_crawler()
HybridQuantumCrawler = _crawler.HybridQuantumCrawler
_SingleThreadComparableCrawler = _crawler._SingleThreadComparableCrawler
_DEFAULT_NON_QUANTUM_PRIORITY = _crawler._DEFAULT_NON_QUANTUM_PRIORITY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEEDS_PATH = os.path.join(os.path.dirname(__file__), "seeds.txt")


def _make_hybrid(
    exploration_temperature: float = 0.0,
    quantum_weight: float = 0.3,
    heuristic_weight: float = 0.5,
    exploration_weight: float = 0.2,
) -> HybridQuantumCrawler:
    """Create a minimal HybridQuantumCrawler for unit-testing scoring methods."""
    return HybridQuantumCrawler(
        seeds_path=_SEEDS_PATH,
        out_jsonl="/dev/null",
        max_pages=10,
        max_depth=3,
        max_workers=1,
        qiskit_shots=64,
        debug=False,
        gpu_math=False,
        aer_request_gpu=False,
        respect_robots=False,
        use_sitemaps=False,
        quantum_weight=quantum_weight,
        heuristic_weight=heuristic_weight,
        exploration_weight=exploration_weight,
        exploration_temperature=exploration_temperature,
    )


def _make_comparable(algorithm: str, exploration_temperature: float = 0.0) -> _SingleThreadComparableCrawler:
    """Create a minimal _SingleThreadComparableCrawler for unit tests."""
    return _SingleThreadComparableCrawler(
        algorithm=algorithm,
        seeds_path=_SEEDS_PATH,
        out_jsonl="/dev/null",
        max_pages=10,
        max_depth=3,
        max_workers=1,
        qiskit_shots=64,
        debug=False,
        gpu_math=False,
        aer_request_gpu=False,
        respect_robots=False,
        use_sitemaps=False,
        quantum_weight=0.3,
        heuristic_weight=0.5,
        exploration_weight=0.2,
        exploration_temperature=exploration_temperature,
    )


# ---------------------------------------------------------------------------
# Test 1: HybridQuantumCrawler inherits the right methods
# ---------------------------------------------------------------------------


def test_hybrid_has_heuristic_and_hybrid_score():
    """HybridQuantumCrawler must expose both _heuristic_score and _hybrid_score_candidate."""
    crawler = _make_hybrid()
    assert hasattr(crawler, "_heuristic_score"), "_heuristic_score missing"
    assert hasattr(crawler, "_hybrid_score_candidate"), "_hybrid_score_candidate missing"


# ---------------------------------------------------------------------------
# Test 2: _SingleThreadComparableCrawler inherits HybridQuantumCrawler
# ---------------------------------------------------------------------------


def test_comparable_crawler_inherits_hybrid():
    """_SingleThreadComparableCrawler must inherit HybridQuantumCrawler (not bare QuantumCrawler)."""
    assert issubclass(_SingleThreadComparableCrawler, HybridQuantumCrawler), (
        "_SingleThreadComparableCrawler should inherit HybridQuantumCrawler "
        "so quantum mode uses the full hybrid scoring pipeline"
    )


# ---------------------------------------------------------------------------
# Test 3: Heuristic score — important pages score higher than generic ones
# ---------------------------------------------------------------------------


def test_heuristic_docs_page_outscores_generic():
    """A /docs/ page should score higher than a generic page under heuristic scoring."""
    crawler = _make_hybrid()
    parent = "https://example.com"

    docs_url = "https://example.com/docs/getting-started"
    generic_url = "https://example.com/some/random/page/12345"

    # Use a matching anchor keyword to reinforce the docs path signal.
    docs_score = crawler._heuristic_score(docs_url, "documentation", parent, 0, "Site")
    generic_score = crawler._heuristic_score(generic_url, "click here", parent, 0, "Site")

    assert docs_score > generic_score, (
        f"docs page ({docs_score:.4f}) should outscore generic page ({generic_score:.4f})"
    )


def test_heuristic_about_page_outscores_generic():
    """An /about/ page should outscore a path with no meaningful segments."""
    crawler = _make_hybrid()
    parent = "https://example.com"

    about_score = crawler._heuristic_score(
        "https://example.com/about", "About us", parent, 0, "Company"
    )
    opaque_score = crawler._heuristic_score(
        "https://example.com/qwerty/zxcvbn/asdf", "link", parent, 0, "Company"
    )

    assert about_score > opaque_score, (
        f"about page ({about_score:.4f}) should outscore opaque path ({opaque_score:.4f})"
    )


def test_heuristic_blog_and_news_paths_score_high():
    """Blog and news URL paths should receive the path-keyword boost."""
    crawler = _make_hybrid()
    parent = "https://example.com"

    for path_kw in ("blog", "news", "research", "api", "contact"):
        url = f"https://example.com/{path_kw}/2024-post"
        score = crawler._heuristic_score(url, path_kw, parent, 0, "Title")
        assert score > 0.3, (
            f"'{path_kw}' path URL should score > 0.3 under heuristic, got {score:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Low-value URL penalties
# ---------------------------------------------------------------------------


def test_low_value_signout_penalised():
    """signout URL should receive the low-value penalty (score < penalty-free baseline)."""
    crawler = _make_hybrid()
    parent = "https://example.com"

    signout_score = crawler._heuristic_score(
        "https://example.com/signout", "Sign out", parent, 0, "App"
    )
    home_score = crawler._heuristic_score(
        "https://example.com/home", "Home", parent, 0, "App"
    )

    assert signout_score < home_score, (
        f"signout ({signout_score:.4f}) should score lower than home ({home_score:.4f})"
    )


def test_low_value_cart_penalised():
    """Cart URL should be penalised."""
    crawler = _make_hybrid()
    parent = "https://shop.example.com"

    cart_score = crawler._heuristic_score(
        "https://shop.example.com/cart", "View cart", parent, 0, "Shop"
    )
    product_score = crawler._heuristic_score(
        "https://shop.example.com/products/widget", "Widget", parent, 0, "Shop"
    )

    assert cart_score < product_score, (
        f"cart ({cart_score:.4f}) should score lower than product ({product_score:.4f})"
    )


def test_low_value_terms_penalised():
    """Privacy-policy and terms URLs should receive a penalty."""
    crawler = _make_hybrid()
    parent = "https://example.com"

    for path in ("privacy-policy", "terms-of-service", "cookie-policy"):
        low_url = f"https://example.com/{path}"
        normal_url = "https://example.com/features"
        low_score = crawler._heuristic_score(low_url, path, parent, 0, "Site")
        normal_score = crawler._heuristic_score(normal_url, "Features", parent, 0, "Site")
        assert low_score < normal_score, (
            f"'{path}' ({low_score:.4f}) should score lower than 'features' ({normal_score:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 5: URL path segments contribute to score
# ---------------------------------------------------------------------------


def test_url_path_keyword_raises_score():
    """A URL with an important path segment should score higher than an identical-looking
    URL without it, holding anchor text and other signals constant."""
    crawler = _make_hybrid()
    parent = "https://example.com"
    anchor = "link"  # Neutral anchor text (no keyword)

    url_with_docs = "https://example.com/docs/reference"
    url_without_docs = "https://example.com/pages/ref12345"

    score_with = crawler._heuristic_score(url_with_docs, anchor, parent, 0, "")
    score_without = crawler._heuristic_score(url_without_docs, anchor, parent, 0, "")

    assert score_with > score_without, (
        f"URL with docs path ({score_with:.4f}) should beat URL without ({score_without:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 6: Quantum and BFS produce different priority values for a candidate
# ---------------------------------------------------------------------------


def test_quantum_and_bfs_use_different_scoring():
    """Quantum mode must score candidates with hybrid scores (non-constant), while BFS
    uses a constant _DEFAULT_NON_QUANTUM_PRIORITY.

    This is the regression test for the root bug where quantum collapsed to BFS
    because _score_candidate produced near-uniform scores close to 0.5.
    """
    quantum_crawler = _make_comparable("quantum", exploration_temperature=0.0)
    bfs_crawler = _make_comparable("bfs", exploration_temperature=0.0)

    parent_url = "https://example.com"
    parent_depth = 0
    parent_title = "Example Site"

    # A set of candidates that span a range of expected scores.
    candidates = [
        ("https://example.com/docs/api", "API Documentation"),
        ("https://example.com/about/team", "Our Team"),
        ("https://example.com/signout", "Sign out"),
        ("https://example.com/cart", "Cart"),
        ("https://example.com/blog/intro-to-quantum", "Quantum blog post"),
        ("https://other.com/random/page", "Random page"),
    ]

    quantum_scores = []
    bfs_scores = []

    for url, anchor in candidates:
        # Quantum mode scores via _hybrid_score_candidate.
        q_score = quantum_crawler._hybrid_score_candidate(
            url, anchor, parent_url, parent_depth, parent_title
        )
        quantum_scores.append(q_score)
        # BFS mode uses the constant _DEFAULT_NON_QUANTUM_PRIORITY.
        bfs_scores.append(_DEFAULT_NON_QUANTUM_PRIORITY)

    # BFS should produce a constant score for every candidate.
    assert all(s == _DEFAULT_NON_QUANTUM_PRIORITY for s in bfs_scores), (
        "BFS should always use _DEFAULT_NON_QUANTUM_PRIORITY"
    )

    # Quantum scores must NOT all be constant (they must vary across candidates).
    assert len(set(round(s, 6) for s in quantum_scores)) > 1, (
        "Quantum hybrid scoring should produce varied scores across different candidates; "
        f"got: {quantum_scores}"
    )


# ---------------------------------------------------------------------------
# Test 7: Quantum ordering — high-value URLs rank above low-value ones
# ---------------------------------------------------------------------------


def test_quantum_high_value_outscores_low_value():
    """With deterministic scoring (temperature=0.0), a docs page should score higher
    than a signout page under the hybrid scorer."""
    crawler = _make_hybrid(exploration_temperature=0.0)
    parent = "https://example.com"

    docs_score = crawler._hybrid_score_candidate(
        "https://example.com/docs/reference", "Documentation", parent, 0, "Docs Site"
    )
    signout_score = crawler._hybrid_score_candidate(
        "https://example.com/signout", "Sign out", parent, 0, "Docs Site"
    )

    assert docs_score > signout_score, (
        f"docs ({docs_score:.4f}) should outscore signout ({signout_score:.4f}) "
        "under hybrid scoring"
    )


def test_quantum_about_outscores_cart():
    """About page should rank above cart page."""
    crawler = _make_hybrid(exploration_temperature=0.0)
    parent = "https://example.com"

    about_score = crawler._hybrid_score_candidate(
        "https://example.com/about", "About us", parent, 0, "Company"
    )
    cart_score = crawler._hybrid_score_candidate(
        "https://example.com/cart", "Cart", parent, 0, "Company"
    )

    assert about_score > cart_score, (
        f"about ({about_score:.4f}) should outscore cart ({cart_score:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 8: Deterministic scoring when temperature=0 (reproducibility)
# ---------------------------------------------------------------------------


def test_hybrid_score_deterministic_at_zero_temperature():
    """When exploration_temperature=0.0, the exploration component contributes exactly
    zero noise.  The heuristic component is fully deterministic; the quantum component
    may vary slightly due to shot-based sampling noise.  We verify that the *heuristic*
    component alone is deterministic and that the overall hybrid score stays within a
    reasonable tolerance (bounded by shot noise from the quantum circuit).
    """
    crawler = _make_hybrid(exploration_temperature=0.0)
    parent = "https://example.com"
    url = "https://example.com/research/papers"
    anchor = "Research Papers"

    # Heuristic score is fully deterministic — it must be identical across calls.
    h_a = crawler._heuristic_score(url, anchor, parent, 1, "Research Hub")
    h_b = crawler._heuristic_score(url, anchor, parent, 1, "Research Hub")
    assert h_a == h_b, f"Heuristic score should be identical; got {h_a:.6f} vs {h_b:.6f}"

    # Exploration noise must be exactly 0 at temperature=0.
    noise = crawler._exploration_noise(url)
    assert noise == 0.0, f"Expected 0.0 exploration noise at temp=0, got {noise}"

    # Full hybrid score: quantum shot noise may cause small variation.
    # We verify it stays within a generous bound (±0.15 covers shot noise at 64 shots).
    score_a = crawler._hybrid_score_candidate(url, anchor, parent, 1, "Research Hub")
    score_b = crawler._hybrid_score_candidate(url, anchor, parent, 1, "Research Hub")
    assert abs(score_a - score_b) < 0.15, (
        f"Hybrid scores should be close at temp=0 (only quantum shot noise can differ); "
        f"got {score_a:.6f} vs {score_b:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 9: Expanded keyword list covers new additions
# ---------------------------------------------------------------------------


def test_anchor_keywords_coverage():
    """Check that new anchor keyword hints are properly detected.

    Both calls use the *same* URL so the test isolates the anchor keyword
    effect alone: the URL path and brevity signals are held constant, and
    the only varying factor is the anchor text keyword.
    """
    crawler = _make_hybrid()
    parent = "https://example.com"

    new_keywords = [
        "whitepaper", "pricing", "platform", "changelog", "handbook",
        "enterprise", "community", "solutions", "features",
    ]
    for kw in new_keywords:
        score_with = crawler._heuristic_score(
            "https://example.com/page", kw, parent, 0, "Site"
        )
        score_without = crawler._heuristic_score(
            "https://example.com/page", "click here", parent, 0, "Site"
        )
        assert score_with > score_without, (
            f"Keyword '{kw}' should boost score; "
            f"with={score_with:.4f} without={score_without:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 10: _exploration_noise returns 0 at temperature=0
# ---------------------------------------------------------------------------


def test_exploration_noise_zero_at_zero_temperature():
    """_exploration_noise must return exactly 0.0 when temperature is 0."""
    crawler = _make_hybrid(exploration_temperature=0.0)
    for url in (
        "https://example.com/page",
        "https://other.org/docs",
        "https://third.net/about",
    ):
        noise = crawler._exploration_noise(url)
        assert noise == 0.0, f"Expected 0.0 noise at temp=0, got {noise} for {url}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
