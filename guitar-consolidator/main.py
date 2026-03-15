"""
guitar-consolidator / main.py
─────────────────────────────
Batch ingestion entry point for the Guitar Tab Consolidator.

Usage
-----
Process a directory of tab files for one song:
    python main.py --dir ./raw_tabs/ --song "Old Man" --artist "Neil Young" --key D

Process individual files explicitly:
    python main.py tests/old_man_tab_a.txt tests/old_man_tab_b.txt \\
        --song "Old Man" --artist "Neil Young" --key D

Save output to a file:
    python main.py --dir ./raw_tabs/ --song "Josie" --artist "Steely Dan" \\
        --key Em --out josie_consolidated.txt

Options
-------
  --dir DIR         Directory containing raw tab .txt files (one song per run)
  --song TITLE      Song title (used in output header)
  --artist ARTIST   Artist name (used in output header)
  --key KEY         Target key for consolidation, e.g. D, Em, F# (required)
  --out FILE        Write consolidated output to FILE (default: stdout)
  --tuning TUNING   Source tuning if non-standard, e.g. CGCGCD (default: Standard)
  --verbose         Print per-file parse details and scoring breakdown
  --no-color        Disable Unicode box-drawing characters in output

Metadata hints in filenames
---------------------------
Files named with rating/vote hints are automatically detected:
    old_man_tab_ug_4.7_1580.txt  → source=ultimate_guitar, rating=4.7, votes=1580
    old_man_tab_echords.txt      → source=echords
    old_man_key_G.txt            → key hint G (overridden by --key if supplied)

Exit codes
----------
  0  Success
  1  No valid tab files found
  2  Consolidation produced zero sections
  3  Fatal error (exception)
"""

from __future__ import annotations
import os
import sys
import re
import argparse
import traceback

# ── Allow running from project root or from any subdirectory ──────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parser.chord_parser  import parse_tab
from parser.transposer    import transpose_tab
from scorer.scoring_engine import rank_tabs
from consolidator.engine  import consolidate
from consolidator.formatter import format_consolidated, save_text, format_compare, save_compare


# ─────────────────────────────────────────────────────────────────────────
# Filename metadata extraction
# ─────────────────────────────────────────────────────────────────────────

# Patterns for extracting hints from filenames like:
#   old_man_ug_4.7_1580.txt   → source=ultimate_guitar rating=4.7 votes=1580
#   creep_echords_3.2_45.txt  → source=echords rating=3.2 votes=45
#   josie_key_Em.txt          → key hint Em

_SOURCE_ALIASES = {
    'ug':              'ultimate_guitar',
    'ultimate_guitar': 'ultimate_guitar',
    'ultimateguitar':  'ultimate_guitar',
    'echords':         'echords',
    'chordie':         'chordie',
    'azchords':        'azchords',
    'tabs':            'unknown',
}

_RATING_VOTES_RE = re.compile(r'_(\d+\.\d+)_(\d+)(?:_|\.)')
_KEY_HINT_RE     = re.compile(r'_key_([A-Ga-g][#b]?m?)', re.IGNORECASE)


def metadata_from_filename(filename: str) -> dict:
    """
    Extracts source, rating, vote_count, and key hints from a filename.
    All values are best-effort; missing values use safe defaults.
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    meta = {
        'source':      'unknown',
        'rating':      3.0,
        'vote_count':  1,
        'key':         None,
    }

    # Source
    for alias, canonical in _SOURCE_ALIASES.items():
        if alias in stem:
            meta['source'] = canonical
            break

    # Rating + votes: e.g. _4.7_1580
    m = _RATING_VOTES_RE.search(stem + '_')
    if m:
        try:
            meta['rating']     = float(m.group(1))
            meta['vote_count'] = int(m.group(2))
        except ValueError:
            pass

    # Key hint: e.g. _key_Em
    m = _KEY_HINT_RE.search(filename)
    if m:
        meta['key'] = m.group(1)

    return meta


# ─────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────

def discover_tab_files(directory: str) -> list:
    """
    Returns a sorted list of .txt file paths in `directory`.
    Skips hidden files and known non-tab names (README, notes, etc.).
    """
    skip_prefixes = ('readme', 'notes', 'todo', 'output', 'consolidated')
    result = []
    try:
        for name in sorted(os.listdir(directory)):
            if name.startswith('.'):
                continue
            if not name.endswith('.txt'):
                continue
            if any(name.lower().startswith(p) for p in skip_prefixes):
                continue
            result.append(os.path.join(directory, name))
    except FileNotFoundError:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────
# Per-file parsing
# ─────────────────────────────────────────────────────────────────────────

def load_and_parse(filepath: str, key_hint: str | None, verbose: bool) -> dict | None:
    """
    Reads, parses, and returns a parsed tab dict.
    Returns None if the file can't be read or produces zero sections.
    """
    try:
        with open(filepath, encoding='utf-8') as f:
            raw = f.read()
    except OSError as e:
        print(f"  ✗ Cannot read {filepath}: {e}", file=sys.stderr)
        return None

    meta = metadata_from_filename(filepath)
    if key_hint:
        meta['key'] = key_hint

    parsed = parse_tab(raw, metadata=meta)

    if not parsed.get('sections'):
        print(f"  ✗ No sections found in {os.path.basename(filepath)} — skipping",
              file=sys.stderr)
        return None

    if verbose:
        _print_parse_detail(filepath, parsed)
    else:
        n_sections = len(parsed['sections'])
        n_chords   = len({c for s in parsed['sections']
                          for b in s['blocks'] for c in b.get('chords', [])})
        warns = len(parsed.get('warnings', []))
        warn_str = f"  ⚠ {warns} warning{'s' if warns != 1 else ''}" if warns else ""
        print(f"  ✓ {os.path.basename(filepath):40s}  "
              f"{n_sections} sections  {n_chords} chords{warn_str}")

    return parsed


def _print_parse_detail(filepath: str, parsed: dict):
    """Verbose per-file breakdown."""
    meta = parsed.get('metadata', {})
    print(f"\n  ── {os.path.basename(filepath)} ──")
    print(f"     source={meta.get('source','?')}  "
          f"rating={meta.get('rating','?')}  "
          f"votes={meta.get('vote_count','?')}  "
          f"key={meta.get('key','?')}  "
          f"tuning={parsed.get('tuning','Standard')}")
    for s in parsed['sections']:
        n_blocks = len(s['blocks'])
        chords = [c for b in s['blocks'] for c in b.get('chords', [])]
        print(f"     [{s['name']}]  {n_blocks} blocks  "
              f"chords: {' '.join(chords[:8])}{'…' if len(chords) > 8 else ''}")
    for w in parsed.get('warnings', []):
        print(f"     ⚠ {w}")


# ─────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────

def print_summary(result: dict, n_sources: int):
    """Prints a concise post-consolidation summary to stdout."""
    sections   = result.get('structure', [])
    glossary   = result.get('chord_glossary', {})
    disputes   = result.get('disputed_lines', [])
    warnings   = result.get('warnings', [])
    errors     = [w for w in warnings if 'error' in w.lower()]

    print()
    print(f"  Consolidated from {n_sources} source{'s' if n_sources != 1 else ''}")
    print(f"  Sections : {len(sections)}  "
          f"({', '.join(norm for norm, orig in sections)})")
    print(f"  Chords   : {len(glossary)}  "
          f"({', '.join(sorted(glossary)[:12])}"
          f"{'…' if len(glossary) > 12 else ''})")
    if disputes:
        print(f"  Disputes : {len(disputes)} chord disagreement"
              f"{'s' if len(disputes) != 1 else ''}")
    if errors:
        print(f"  Errors   : {len(errors)}", file=sys.stderr)
        for e in errors:
            print(f"    ✗ {e}", file=sys.stderr)
    elif warnings:
        processing = [w for w in warnings if 'error' not in w.lower()]
        if processing:
            print(f"  Notes    : {len(processing)} processing note"
                  f"{'s' if len(processing) != 1 else ''}")
    print()


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Consolidate multiple guitar tab sources for one song.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Exit codes')[0].strip(),
    )
    p.add_argument('files', nargs='*', metavar='FILE',
                   help='Individual tab .txt files to process')
    p.add_argument('--dir', metavar='DIR',
                   help='Directory containing raw tab .txt files')
    p.add_argument('--song', metavar='TITLE', default='',
                   help='Song title for output header')
    p.add_argument('--artist', metavar='ARTIST', default='',
                   help='Artist name for output header')
    p.add_argument('--key', metavar='KEY', default=None,
                   help='Target key, e.g. D, Em, F# (required)')
    p.add_argument('--out', metavar='FILE', default=None,
                   help='Write output to FILE instead of stdout')
    p.add_argument('--compare', action='store_true',
                   help='Output a side-by-side source comparison report instead of the consolidated tab')
    p.add_argument('--compare-out', metavar='FILE', default=None,
                   help='Write compare report to FILE (also writes consolidated to --out if given)')
    p.add_argument('--tuning', metavar='TUNING', default='Standard',
                   help='Source tuning if non-standard, e.g. CGCGCD')
    p.add_argument('--verbose', action='store_true',
                   help='Print per-file parse details and scoring breakdown')
    return p


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    # ── Collect file paths ────────────────────────────────────────────────
    filepaths = list(args.files)
    if args.dir:
        discovered = discover_tab_files(args.dir)
        if not discovered:
            print(f"  ✗ No .txt files found in {args.dir}", file=sys.stderr)
            return 1
        filepaths.extend(discovered)

    if not filepaths:
        ap.print_help()
        print("\n  ✗ No files specified. Use --dir or pass file paths directly.",
              file=__import__('sys').stderr)
        return 1

    # ── Validate key ──────────────────────────────────────────────────────
    key = args.key
    if not key:
        # Try to infer from first filename
        for fp in filepaths:
            m = _KEY_HINT_RE.search(fp)
            if m:
                key = m.group(1)
                print(f"  ℹ Key inferred from filename: {key}")
                break
    if not key:
        print("  ✗ --key is required (e.g. --key D)", file=sys.stderr)
        print("    If key hints are in filenames (e.g. song_key_D.txt), "
              "they are auto-detected.", file=sys.stderr)
        return 1

    # ── Print run header ──────────────────────────────────────────────────
    title  = args.song   or os.path.splitext(os.path.basename(filepaths[0]))[0]
    artist = args.artist or ''
    header = f"{title}" + (f" — {artist}" if artist else "")
    print()
    print(f"  Guitar Tab Consolidator")
    print(f"  {header}  (key: {key})")
    print(f"  Processing {len(filepaths)} file{'s' if len(filepaths) != 1 else ''}…")
    print()

    # ── Parse ─────────────────────────────────────────────────────────────
    parsed_tabs = []
    for fp in filepaths:
        parsed = load_and_parse(fp, key_hint=key, verbose=args.verbose)
        if parsed is not None:
            parsed_tabs.append(parsed)

    if not parsed_tabs:
        print("  ✗ No valid tab files could be parsed.", file=sys.stderr)
        return 1

    # ── Transpose to common key ───────────────────────────────────────────
    try:
        normalized = [transpose_tab(p, to_key=key) for p in parsed_tabs]
    except Exception as e:
        print(f"  ✗ Transposition error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 3

    # ── Rank ──────────────────────────────────────────────────────────────
    try:
        ranked = rank_tabs(normalized, consistency_weight=2.0)
    except Exception as e:
        print(f"  ✗ Scoring error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 3

    if args.verbose:
        print()
        print("  Scoring:")
        for i, tab in enumerate(ranked, 1):
            meta = tab.get('metadata', {})
            print(f"    {i}. score={tab.get('score', 0):.3f}  "
                  f"source={meta.get('source','?')}  "
                  f"rating={meta.get('rating','?')}")

    # ── Consolidate ───────────────────────────────────────────────────────
    try:
        result = consolidate(ranked, song_title=title, artist=artist, target_key=key)
    except Exception as e:
        print(f"  ✗ Consolidation error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 3

    if not result.get('structure'):
        print("  ✗ Consolidation produced zero sections.", file=sys.stderr)
        return 2

    # ── Format + output ───────────────────────────────────────────────────
    formatted = format_consolidated(result)

    if args.out:
        try:
            save_text(result, args.out)
            print(f"  ✓ Output written to {args.out}")
        except OSError as e:
            print(f"  ✗ Could not write to {args.out}: {e}", file=sys.stderr)
            return 3
    elif not args.compare:
        print(formatted)

    # ── Compare report ────────────────────────────────────────────────────
    if args.compare or args.compare_out:
        compare_text = format_compare(result, ranked)
        compare_out  = args.compare_out
        if compare_out:
            try:
                save_compare(result, ranked, compare_out)
                print(f"  ✓ Compare report written to {compare_out}")
            except OSError as e:
                print(f"  ✗ Could not write compare report to {compare_out}: {e}",
                      file=sys.stderr)
                return 3
        else:
            print(compare_text)

    print_summary(result, n_sources=len(parsed_tabs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
