"""
guitar-consolidator / scraper.py
─────────────────────────────────
Fetches chord tabs from Ultimate Guitar, saves them to a local cache,
runs the consolidation pipeline, and records everything back into the
knowledge base so future runs benefit from the data.

Usage
─────
  python scraper.py "Yesterday" "The Beatles" --key F
  python scraper.py "Creep" "Radiohead" --key G --n 5
  python scraper.py "Old Man" "Neil Young" --key D --out old_man.txt
  python scraper.py "Yesterday" "The Beatles" --key F --refresh
  python scraper.py "Yesterday" "The Beatles" --key F --compare

Options
───────
  --key KEY         Target consolidation key (required)
  --n N             Number of tabs to fetch (default: 5)
  --out FILE        Write consolidated output to FILE
  --compare         Print side-by-side source comparison instead
  --compare-out F   Write compare report to FILE
  --refresh         Ignore cache and re-fetch from UG
  --no-learn        Skip recording results into the knowledge base
  --verbose         Show fetch details and pipeline scoring
  --mock            Use bundled mock data instead of hitting the network
                    (useful for development / CI)

INSTALL DEPS (once, on your machine)
─────────────────────────────────────
  pip install cloudscraper beautifulsoup4

How it works
────────────
1. Search UG: search.php?search_type=title&value=...&type=300
2. Filter: Chords type only, ≥MIN_VOTES votes, sort by rating desc
3. Fetch each tab page with cloudscraper (Cloudflare bypass)
4. Extract content from js-store JSON blob, strip UG markup tags
5. Save to scraper_cache/{slug}/{song}_ug_{rating}_{votes}_{n}.txt
   (filename encodes metadata so main.py scoring engine reads it)
6. Run full consolidation pipeline on cached files
7. Feed result back into knowledge base — every scrape improves the model
8. do something stupid

UG page structure (as of 2024-2025)
────────────────────────────────────
UG is behind Cloudflare — use cloudscraper, not plain requests.

Both search pages and individual tab pages embed all data as JSON
in a data-content attribute on <div class="js-store">:
  <div class="js-store" data-content="{...HTML-escaped JSON...}">

Search page JSON path:
  .store.page.data.results[]
  Each result: id, song_name, artist_name, type, rating, votes,
               tab_url, username, version

Tab page JSON path:
  .store.page.data.tab_view.wiki_tab.content   ← raw chord text
  (UG markup [ch]Am[/ch], [tab]e|---[/tab] must be stripped)

If UG changes their structure, only two functions need updating:
  _parse_search_page()  and  _parse_tab_page()
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
import argparse
import textwrap
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

# ── Allow running from project root ──────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parser.chord_parser   import parse_tab
from parser.transposer      import transpose_tab
from scorer.scoring_engine  import rank_tabs
from consolidator.engine    import consolidate
from consolidator.formatter import (
    format_consolidated, save_text,
    format_compare,      save_compare,
)
from knowledge.corpus import Corpus

# ── Constants ─────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(ROOT, "scraper_cache")

# type=300 restricts results to Chords tabs server-side, cutting noise.
UG_SEARCH_URL = (
    "https://www.ultimate-guitar.com/search.php"
    "?search_type=title&value={query}&type=300"
)

# Tabs with fewer votes than this are skipped (filters out one-off junk)
MIN_VOTES = 10

# Accept only these UG type strings even after server-side filtering
ALLOWED_TYPES = {"Chords"}

# Polite delay between HTTP requests (seconds)
REQUEST_DELAY = 1.5

# cloudscraper browser profile — mimics a real Chrome on Windows.
# UG is behind Cloudflare; plain requests.get() returns 403.
_CLOUDSCRAPER_BROWSER = {"browser": "chrome", "platform": "windows", "mobile": False}


# ═══════════════════════════════════════════════════════════════════════════
# HTTP layer — the ONLY place that touches the network.
# Uses a persistent cloudscraper session so Cloudflare cookies are reused
# across all requests in one run (avoids repeated 5-second challenge waits).
# ═══════════════════════════════════════════════════════════════════════════

def _make_session():
    """
    Create and return a cloudscraper session that bypasses Cloudflare.
    Raises RuntimeError with install instructions if cloudscraper is missing.
    """
    try:
        import cloudscraper
    except ImportError:
        raise RuntimeError(
            "cloudscraper is not installed.\n"
            "Run:  pip install cloudscraper\n"
            "Then retry.  Use --mock for offline / CI testing."
        )
    return cloudscraper.create_scraper(browser=_CLOUDSCRAPER_BROWSER)


def _http_get(url: str, session=None, verbose: bool = False) -> str:
    """
    Fetch a URL and return the response body as a string.
    Raises RuntimeError on HTTP errors or network failure.

    Pass a session from _make_session() to reuse Cloudflare cookies
    across multiple requests in the same run.
    """
    if verbose:
        print(f"    GET {url}", file=sys.stderr)

    sess = session or _make_session()
    try:
        resp = sess.get(url, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        raise RuntimeError(f"Fetch failed for {url!r}: {e}") from e


# ═══════════════════════════════════════════════════════════════════════════
# UG HTML parsing — isolated here so one place needs updating if UG changes
# ═══════════════════════════════════════════════════════════════════════════

def _extract_js_store(html: str) -> dict:
    """
    UG embeds all page data as JSON in a data-content attribute on
    the element with class 'js-store'.  Extract and return that dict.
    """
    # Try the data-content approach first (most common)
    m = re.search(r'class="js-store"\s+data-content="([^"]+)"', html)
    if m:
        raw = m.group(1)
        # UG HTML-encodes the JSON value
        raw = (raw.replace("&quot;", '"')
                  .replace("&amp;", "&")
                  .replace("&#039;", "'")
                  .replace("&#39;", "'"))
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Fallback: look for a <script> block with the store data
    m = re.search(r'window\.UGAPP\.store\s*=\s*(\{.+?\});\s*</script>', html, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Could not extract JS store from UG page.  "
        "UG may have changed their page structure — update _extract_js_store()."
    )


def _parse_search_page(html: str) -> list[dict]:
    """
    Parse a UG search results page and return a list of tab metadata dicts:
        {
          'id':        int,
          'song':      str,
          'artist':    str,
          'type':      str,   # 'Chords', 'Tab', etc.
          'rating':    float,
          'votes':     int,
          'url':       str,
          'author':    str,
          'version':   int,
        }
    Returns empty list if no results found.
    """
    try:
        store = _extract_js_store(html)
        results = store["store"]["page"]["data"]["results"]
    except (KeyError, RuntimeError):
        return []

    tabs = []
    for r in results:
        try:
            raw_url = r.get("tab_url") or r.get("url") or ""
            # Normalise relative URLs to absolute
            if raw_url and raw_url.startswith("/"):
                raw_url = "https://www.ultimate-guitar.com" + raw_url
            # Skip entries without a usable URL
            if not raw_url or not raw_url.startswith("http"):
                continue
            tab = {
                "id":      r.get("id", 0),
                "song":    r.get("song_name", ""),
                "artist":  r.get("artist_name", ""),
                "type":    r.get("type", ""),
                "rating":  float(r.get("rating", 0.0)),
                "votes":   int(r.get("votes", 0)),
                "url":     raw_url,
                "author":  r.get("username", ""),
                "version": int(r.get("version", 1)),
            }
            tabs.append(tab)
        except (TypeError, ValueError):
            continue

    return tabs


def _parse_tab_page(html: str) -> str | None:
    """
    Parse a UG individual tab page and return the raw chord text.
    Returns None if the content can't be extracted.
    """
    try:
        store  = _extract_js_store(html)
        # Path varies slightly between tab types
        tab_view = store["store"]["page"]["data"]["tab_view"]
        content  = tab_view.get("wiki_tab", {}).get("content", "")
        if not content:
            # Some tabs use a different key
            content = tab_view.get("applicature", {}).get("content", "")
        return content if content else None
    except (KeyError, RuntimeError):
        return None


def _clean_tab_content(raw: str) -> str:
    """
    UG tab content contains markup like [ch]Am[/ch] and [tab]...[/tab].
    Strip these tags and normalise whitespace / unicode.
    """
    # Remove UG markup tags
    text = re.sub(r'\[/?(?:ch|tab|verse|chorus|bridge|intro|outro|pre-chorus|'
                  r'interlude|instrumental|solo|break|coda|outro|refrain)\]',
                  '', raw, flags=re.IGNORECASE)
    # Normalise unicode (smart quotes, en-dashes etc.)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Collapse runs of blank lines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════════════════

def _slug(song: str, artist: str) -> str:
    """URL-safe directory name for a song."""
    combined = f"{artist}_{song}".lower()
    combined = re.sub(r"[^\w\s-]", "", combined)
    combined = re.sub(r"[\s_-]+", "_", combined).strip("_")
    return combined[:60]


def _cache_dir(song: str, artist: str) -> Path:
    return Path(CACHE_DIR) / _slug(song, artist)


def _cache_filename(song: str, artist: str, rating: float, votes: int, idx: int) -> str:
    """
    Generate a filename that embeds metadata so main.py can parse it.
    e.g.  yesterday_ug_4.8_2341.txt
    Uses idx suffix when ratings collide.
    """
    song_slug = re.sub(r"[^\w]", "_", song.lower()).strip("_")
    return f"{song_slug}_ug_{rating:.1f}_{votes}_{idx}.txt"


def _cached_files(song: str, artist: str) -> list[Path]:
    """Return cached tab files for a song, sorted best-rated first.

    Sorts by (rating desc, votes desc) parsed from the filename so that
    '4.9' always beats '4.10' etc.  Falls back to reverse-alpha if a
    filename doesn't match the expected pattern.
    """
    d = _cache_dir(song, artist)
    if not d.exists():
        return []

    _pat = re.compile(r'_ug_([\d.]+)_(\d+)_\d+\.txt$')

    def _sort_key(p: Path):
        m = _pat.search(p.name)
        if m:
            return (float(m.group(1)), int(m.group(2)))
        return (0.0, 0)

    return sorted(d.glob("*.txt"), key=_sort_key, reverse=True)


def _save_to_cache(song: str, artist: str, content: str,
                   rating: float, votes: int, idx: int) -> Path:
    d = _cache_dir(song, artist)
    d.mkdir(parents=True, exist_ok=True)
    fname = _cache_filename(song, artist, rating, votes, idx)
    path  = d / fname
    path.write_text(content, encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Core fetch logic
# ═══════════════════════════════════════════════════════════════════════════

def fetch_tabs(
    song:      str,
    artist:    str,
    n:         int  = 5,
    refresh:   bool = False,
    verbose:   bool = False,
    debug_dir: str  = None,  # if set, save raw HTML responses here
    _session         = None,   # injectable for tests
) -> tuple:
    """
    Ensure we have N tabs for (song, artist) in the cache.
    If cache is warm and refresh=False, returns immediately.
    Returns (paths, urls) where paths is a list of Path objects pointing to
    the cached tab files and urls is a list of the corresponding UG URLs
    (empty list on a cache hit).

    debug_dir: path to a directory where raw HTML pages will be saved.
    Useful for diagnosing parse failures without re-fetching from UG.
    """
    cached = _cached_files(song, artist)
    if cached and not refresh:
        if verbose:
            print(f"  Cache hit — {len(cached)} file(s) in {_cache_dir(song, artist)}")
        return cached, []

    # Create one session for the whole run; Cloudflare challenge solved once.
    session = _session or _make_session()

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    print(f"  Searching UG: {song} — {artist}")
    query      = quote_plus(f"{song} {artist}")
    search_url = UG_SEARCH_URL.format(query=query)

    html    = _http_get(search_url, session=session, verbose=verbose)

    if debug_dir:
        slug = _slug(song, artist)
        debug_path = os.path.join(debug_dir, f"{slug}_search.html")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  [debug] Search HTML saved → {debug_path}", file=sys.stderr)

    results = _parse_search_page(html)

    if not results:
        raise RuntimeError(
            f"No results found on UG for '{song}' by '{artist}'.  "
            "Check spelling or try --mock for offline testing."
        )

    # Filter to allowed types with enough votes
    candidates = [
        r for r in results
        if r["type"] in ALLOWED_TYPES and r["votes"] >= MIN_VOTES
    ]

    # Filter by song title similarity.
    # UG search often returns the artist's highest-rated tabs regardless of
    # whether the title matches (e.g. "Blackbird" search returns "Hey Jude").
    # Keep results where the requested song words substantially overlap.
    def _title_match(result_song: str, req_song: str) -> bool:
        r = result_song.lower()
        q = req_song.lower()
        if q in r or r in q:
            return True
        q_words = set(re.sub(r"[^a-z0-9 ]", "", q).split())
        r_words = set(re.sub(r"[^a-z0-9 ]", "", r).split())
        if not q_words:
            return True
        return len(q_words & r_words) / len(q_words) >= 0.5

    title_matched = [r for r in candidates if _title_match(r["song"], song)]
    if title_matched:
        candidates = title_matched
    elif verbose:
        print(f"  [warn] No close title match for '{song}' — using best-rated results")

    # Sort: rating descending, then votes descending as tiebreaker
    candidates.sort(key=lambda r: (r["rating"], r["votes"]), reverse=True)

    if not candidates:
        raise RuntimeError(
            f"Found {len(results)} UG result(s) for \'{song}\' "
            f"but none are Chords tabs with ≥{MIN_VOTES} votes.\n"
            "Try a different spelling or --mock."
        )

    # Take top N
    to_fetch = candidates[:n]
    print(f"  Found {len(candidates)} chord tab(s), fetching top {len(to_fetch)}")

    # Clear stale cache if refreshing
    if refresh:
        d = _cache_dir(song, artist)
        if d.exists():
            for f in d.glob("*.txt"):
                f.unlink()

    fetched      = []
    fetched_urls = []
    for idx, meta in enumerate(to_fetch, 1):
        tab_url = meta["url"]
        print(f"  [{idx}/{len(to_fetch)}] ★{meta['rating']:.1f} "
              f"({meta['votes']:,} votes)  {tab_url}")

        try:
            time.sleep(REQUEST_DELAY)
            tab_html = _http_get(tab_url, session=session, verbose=verbose)

            if debug_dir:
                slug = _slug(song, artist)
                debug_path = os.path.join(debug_dir, f"{slug}_tab_{idx}.html")
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(tab_html)
                print(f"  [debug] Tab HTML saved → {debug_path}", file=sys.stderr)

            content  = _parse_tab_page(tab_html)
            if not content:
                print(f"    ⚠ Could not extract tab content — skipping")
                continue

            content = _clean_tab_content(content)
            path    = _save_to_cache(
                song, artist,
                content,
                meta["rating"],
                meta["votes"],
                idx,
            )
            fetched.append(path)
            fetched_urls.append(tab_url)
            if verbose:
                print(f"    Saved → {path.name}")
        except RuntimeError as e:
            print(f"    ⚠ {e} — skipping")
            continue

    if not fetched:
        raise RuntimeError("Could not fetch any tab content from UG.")

    return fetched, fetched_urls


# ═══════════════════════════════════════════════════════════════════════════
# Mock data — used when --mock is passed or network is unavailable
# Bundled copies of our test fixtures so the scraper can be fully tested
# without hitting UG.
# ═══════════════════════════════════════════════════════════════════════════

_MOCK_REGISTRY = {
    # (song_lower, artist_lower) → list of (fixture_filename, rating, votes)
    ("yesterday", "the beatles"): [
        ("yesterday_tab_a.txt", 4.8, 1000),
        ("yesterday_tab_b.txt", 4.6,  800),
        ("yesterday_tab_c.txt", 4.4,  600),
    ],
    ("creep", "radiohead"): [
        ("creep_tab_a.txt", 4.8, 900),
        ("creep_tab_b.txt", 4.5, 500),
    ],
    ("old man", "neil young"): [
        ("old_man_tab_a.txt", 4.7, 1580),
        ("old_man_tab_b.txt", 4.5,  720),
    ],
    ("josie", "steely dan"): [
        ("josie_tab_a.txt", 4.8, 940),
        ("josie_tab_b.txt", 4.6, 412),
    ],
}

def _mock_fetch(song: str, artist: str, n: int, verbose: bool) -> list[Path]:
    """
    Return paths to bundled test fixtures, pretending they were fetched.
    Writes them into the cache so the rest of the pipeline is identical.
    """
    key = (song.lower().strip(), artist.lower().strip())
    fixtures = _MOCK_REGISTRY.get(key)

    if not fixtures:
        available = [f"'{s}' by '{a}'" for s, a in _MOCK_REGISTRY]
        raise RuntimeError(
            f"No mock data for '{song}' by '{artist}'.  "
            f"Available: {', '.join(available)}"
        )

    tests_dir = Path(ROOT) / "tests"
    fetched   = []
    for idx, (fname, rating, votes) in enumerate(fixtures[:n], 1):
        src_path = tests_dir / fname
        if not src_path.exists():
            print(f"  ⚠ Mock fixture not found: {src_path}", file=sys.stderr)
            continue
        content = src_path.read_text(encoding="utf-8")
        path    = _save_to_cache(song, artist, content, rating, votes, idx)
        fetched.append(path)
        if verbose:
            print(f"  [mock {idx}] {fname} → {path.name}")

    if not fetched:
        raise RuntimeError("No mock fixtures could be loaded.")
    return fetched


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    tab_paths: list[Path],
    song:      str,
    artist:    str,
    key:       str,
    verbose:   bool = False,
) -> tuple[dict, str, list]:
    """
    Parse, transpose, rank, and consolidate the given tab files.
    Returns (consolidated_dict, formatted_output, ranked_tabs).
    """
    from main import metadata_from_filename

    parsed = []
    for path in tab_paths:
        raw  = path.read_text(encoding="utf-8")
        meta = metadata_from_filename(str(path))
        # Don't inject target key as source key — let the transposer infer
        meta.pop("key", None)
        p = parse_tab(raw, metadata=meta)
        if p.get("sections"):
            parsed.append(p)
        elif verbose:
            print(f"  ⚠ No sections parsed from {path.name} — skipping")

    if not parsed:
        raise RuntimeError("None of the fetched tabs produced parseable sections.")

    normalized = [transpose_tab(p, to_key=key) for p in parsed]
    
    # Filter out tabs that required extreme transposition (likely wrong key source)
    MAX_TRANSPOSE_SEMITONES = 3
    def _semitones_transposed(tab):
        warn = ' '.join(tab.get('warnings', []))
        import re
        m = re.search(r'\((\d+) semitones\)', warn)
        return int(m.group(1)) if m else 0
    normalized = [t for t in normalized if _semitones_transposed(t) <= MAX_TRANSPOSE_SEMITONES]    
    
    ranked = rank_tabs(normalized, consistency_weight=2.0)

    if verbose:
        print(f"\n  Scoring ({len(ranked)} tabs):")
        for i, t in enumerate(ranked, 1):
            m = t.get("metadata", {})
            print(f"    {i}. score={t.get('score', 0):.3f}  "
                  f"rating={m.get('rating','?')}  votes={m.get('vote_count','?')}")

    consolidated = consolidate(ranked[:1], song_title=song, artist=artist, target_key=key)
    output       = format_consolidated(consolidated)

    return consolidated, output, ranked


# ═══════════════════════════════════════════════════════════════════════════
# Learning loop
# ═══════════════════════════════════════════════════════════════════════════

def record_to_corpus(consolidated: dict, ranked: list, verbose: bool = False) -> dict:
    """
    Feed the consolidation result back into the knowledge base.
    Returns a summary dict of what was recorded.
    """
    corpus  = Corpus()
    summary = corpus.record(consolidated, ranked)

    if verbose and summary:
        print("\n  Knowledge base updated:")
        for k, v in summary.items():
            print(f"    {k}: {v}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scraper.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Fetch chord tabs from Ultimate Guitar, consolidate them,
            and record the results back into the knowledge base.
        """),
        epilog=textwrap.dedent("""\
            Examples:
              python scraper.py "Yesterday" "The Beatles" --key F
              python scraper.py "Creep" "Radiohead" --key G --n 3
              python scraper.py "Old Man" "Neil Young" --key D --mock --verbose
        """),
    )
    p.add_argument("song",   help="Song title")
    p.add_argument("artist", help="Artist name")
    p.add_argument("--key",  required=True, metavar="KEY",
                   help="Target consolidation key, e.g. F, Em, Bb")
    p.add_argument("--n",    type=int, default=5, metavar="N",
                   help="Number of tabs to fetch (default: 5)")
    p.add_argument("--out",  metavar="FILE",
                   help="Write consolidated output to FILE")
    p.add_argument("--compare", action="store_true",
                   help="Print side-by-side source comparison report")
    p.add_argument("--compare-out", metavar="FILE", dest="compare_out",
                   help="Write compare report to FILE")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore cache and re-fetch from UG")
    p.add_argument("--no-learn", action="store_true", dest="no_learn",
                   help="Skip recording results into the knowledge base")
    p.add_argument("--mock", action="store_true",
                   help="Use bundled test fixtures instead of hitting UG")
    p.add_argument("--debug", metavar="DIR", dest="debug_dir",
                   help="Save raw HTML responses to DIR for diagnosis")
    p.add_argument("--verbose", action="store_true",
                   help="Print fetch details and scoring breakdown")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    print(f"\n  Guitar Tab Scraper")
    print(f"  {args.song} — {args.artist}  (key: {args.key})")
    print(f"  {'─' * 60}")

    # ── 1. Fetch (or load from cache / mock) ─────────────────────────────
    try:
        if args.mock:
            print("  [mock mode — using bundled test fixtures]")
            tab_paths = _mock_fetch(args.song, args.artist, args.n, args.verbose)
        else:
            tab_paths, _ = fetch_tabs(
                args.song, args.artist,
                n=args.n,
                refresh=args.refresh,
                verbose=args.verbose,
                debug_dir=getattr(args, "debug_dir", None),
            )
    except RuntimeError as e:
        print(f"\n  ✗ Fetch failed: {e}", file=sys.stderr)
        return 1

    print(f"  Using {len(tab_paths)} tab(s):")
    for p in tab_paths:
        print(f"    {p.name}")

    # ── 2. Consolidate ───────────────────────────────────────────────────
    print(f"\n  Running consolidation pipeline…")
    try:
        consolidated, output, ranked = run_pipeline(
            tab_paths, args.song, args.artist, args.key,
            verbose=args.verbose,
        )
    except RuntimeError as e:
        print(f"\n  ✗ Pipeline failed: {e}", file=sys.stderr)
        return 2

    secs   = consolidated.get("structure", [])
    chords = consolidated.get("chord_glossary", [])
    disp   = len(consolidated.get("disputed_lines", []))
    print(f"  ✓ {len(secs)} sections  ·  {len(chords)} chords  ·  {disp} disputed line(s)")

    # ── 3. Output ────────────────────────────────────────────────────────
    if args.out:
        save_text(consolidated, args.out)
        print(f"  Saved → {args.out}")

    if args.compare or args.compare_out:
        compare_text = format_compare(consolidated, ranked)
        if args.compare_out:
            save_compare(consolidated, ranked, args.compare_out)
            print(f"  Compare report → {args.compare_out}")
        if args.compare:
            print()
            print(compare_text)
    elif not args.out:
        print()
        print(output)

    # ── 4. Learn ─────────────────────────────────────────────────────────
    if not args.no_learn:
        try:
            summary = record_to_corpus(consolidated, ranked, verbose=args.verbose)
            prog_added = summary.get("progressions_recorded", 0)
            obs_added  = summary.get("sources_updated", 0)
            print(f"\n  Knowledge base updated  "
                  f"(+{prog_added} progressions recorded, {obs_added} source(s) updated)")
        except Exception as e:
            # Learning failure should never block output
            print(f"\n  ⚠ Knowledge base update failed (non-fatal): {e}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
