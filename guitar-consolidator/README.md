# Guitar Tab Consolidator

Fetches multiple chord tab versions of a song from Ultimate Guitar, scores and ranks them, transposes to a common key, and consolidates them into a single best-of-breed chord chart using weighted voting across sources.

## How It Works

1. **Scrape** — Searches Ultimate Guitar for chord tabs, bypasses Cloudflare via `cloudscraper`, extracts tab content from the `js-store` JSON blob, and caches results locally
2. **Parse** — Detects section headers, chord lines, lyrics, performance directions, tuning/capo info; strips ASCII tab notation and sign-offs
3. **Transpose** — Infers source key via scale-fit analysis + section-boundary heuristics, transposes all tabs to a common target key
4. **Score & Rank** — Base score (rating, votes, source quality, completeness) + consistency score (Jaccard similarity of chord vocabularies across tabs)
5. **Consolidate** — Aligns lyric lines across sources using word-level Jaccard similarity, votes on chords per line weighted by tab score, resolves disputes via corpus data when available
6. **Format** — Outputs a clean chord chart with positional chord placement above lyrics, disputed-line annotations, and CGCGCD tuning voicing suggestions

The system maintains a **knowledge base** (`knowledge/data/`) that improves with each song processed — recording chord progressions, source reliability stats, and song structure priors.

## Install

```bash
make install
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+ required. Dependencies (`cloudscraper`, `beautifulsoup4`) are declared in `pyproject.toml`.

## Quick Start

```bash
make install   # create venv + install deps
make serve     # start web UI at http://localhost:8080
make test      # run test suite
```

## Usage

### Web UI (recommended)

```bash
make serve
# Open http://localhost:8080
```

Enter a song title and artist to generate a consolidated tab.

### CLI — Scraper (fetches from Ultimate Guitar)

```bash
# Basic usage
python scraper.py "Yesterday" "The Beatles" --key F

# Fetch 3 tabs, save output, verbose scoring
python scraper.py "Creep" "Radiohead" --key G --n 3 --out creep.txt --verbose

# Side-by-side source comparison
python scraper.py "Old Man" "Neil Young" --key D --compare

# Offline mode with bundled test fixtures
python scraper.py "Yesterday" "The Beatles" --key F --mock

# Re-fetch ignoring cache
python scraper.py "Yesterday" "The Beatles" --key F --refresh
```

### CLI — Batch mode (local files)

```bash
# Process a directory of tab files
python main.py --dir ./raw_tabs/ --song "Old Man" --artist "Neil Young" --key D

# Process specific files
python main.py tests/old_man_tab_a.txt tests/old_man_tab_b.txt \
    --song "Old Man" --artist "Neil Young" --key D --out consolidated.txt
```

## Tests

```bash
make test
```

## Project Structure

```
main.py              — Batch CLI entry point (local tab files)
scraper.py           — UG fetcher + full pipeline runner
server.py            — Local web server + HTML UI
renderer.py          — Single-tab plain-text renderer
comp.html            — Web UI frontend
parser/
  chord_parser.py    — Tab text → structured sections/blocks/chords
  transposer.py      — Key detection and chord transposition
  tunings.py         — Tuning registry (Standard, CGCGCD, Drop D, etc.)
  cgcgcd.py          — CGCGCD voicing engine + capo suggestions
scorer/
  scoring_engine.py  — Tab quality scoring and ranking
consolidator/
  engine.py          — Multi-source alignment, voting, consolidation
  formatter.py       — Consolidated output + compare report renderer
  chord_diagrams.py  — ASCII fretboard diagram generator
knowledge/
  corpus.py          — Coordinator for all knowledge stores
  chord_corpus.py    — Observed chord progression database
  source_stats.py    — Learned source reliability weights
  structure_priors.py — Section sequence transition data
  data/              — Persistent JSON knowledge files
scraper_cache/       — Cached tab downloads (by song)
tests/               — Test fixtures and test suite
```
