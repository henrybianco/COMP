"""
scorer/scoring_engine.py
────────────────────────
Scores and ranks parsed tab dicts using three components:

  1. Base score    — rating, vote count, source quality, completeness
  2. Consistency   — how much this tab's chords agree with the other tabs
  3. Final score   — base + (consistency × weight)

All tabs should be transposed to the same key before scoring,
so that chord comparison is meaningful.
"""

import math

# ─────────────────────────────────────────────
# SOURCE WEIGHTS
# ─────────────────────────────────────────────
# Multipliers encoding our prior belief about source reliability.
# Values > 1.0 boost, values < 1.0 penalize.

SOURCE_WEIGHTS = {
    'ultimate_guitar': 1.0,
    'ultimate_guitar_official': 1.2,
    'songsterr': 1.25,
    'echords': 0.85,
    'chordie': 0.80,
    'azchords': 0.80,
    'unknown': 0.75
}


# ─────────────────────────────────────────────
# BASE SCORE
# ─────────────────────────────────────────────

def score_tab(parsed_tab):
    """
    Calculates a base quality score for a single parsed tab.

    COMPONENTS:
    ┌─────────────────┬────────┬──────────────────────────────────────────┐
    │ Component       │ Weight │ Why                                      │
    ├─────────────────┼────────┼──────────────────────────────────────────┤
    │ Rating          │  3.0   │ Direct signal from real players          │
    │ Vote count      │  1.5   │ Log-scaled — volume of evidence          │
    │ Completeness    │  1.0   │ More sections = more of the song         │
    │ Source quality  │  mult  │ Multiplier applied to entire sum         │
    └─────────────────┴────────┴──────────────────────────────────────────┘
    """
    meta = parsed_tab.get('metadata', {})

    rating = float(meta.get('rating', 3.0))
    rating = max(1.0, min(5.0, rating))
    rating_norm = (rating - 1.0) / 4.0

    vote_count = int(meta.get('vote_count', 0))
    vote_weight = math.log(1 + vote_count)

    source = meta.get('source', 'unknown').lower().replace(' ', '_')

    # Use corpus learned weight if available, fall back to hard-coded prior.
    try:
        from knowledge.corpus import corpus as _corpus
        source_weight = _corpus.source_weight(source)
        author = meta.get('author', '')
        if author:
            source_weight *= _corpus.author_weight(author)
    except ImportError:
        source_weight = SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS['unknown'])

    sections = parsed_tab.get('sections', [])
    completeness = min(len(sections) / 5.0, 1.0)

    raw_score = (
        (rating_norm * 3.0) +
        (vote_weight * 1.5) +
        (completeness * 1.0) +
        source_weight
    ) * source_weight

    return round(raw_score, 4)


# ─────────────────────────────────────────────
# CONSISTENCY SCORING
# ─────────────────────────────────────────────

def get_chord_set(parsed_tab):
    """
    Returns the set of unique chord identifiers (root + base quality) used
    across an entire tab.

    We use ROOT+QUALITY rather than the full chord name so that:
      - 'Am' and 'Am7' count as the same chord (same harmonic function,
        extensions are transcriber flavour)
      - 'Am' and 'A' count as DIFFERENT chords (parallel minor/major
        confusion is a genuine error signal worth preserving)
      - 'F#m' and 'Gbm' count as the same chord (enharmonic equivalents)

    Quality extraction rules:
      - Minor quality:   root starts 'm' and is NOT 'maj' → append 'm'
      - Diminished:      'dim' or 'b5' in quality string  → append 'dim'
      - Augmented:       'aug' or '#5' in quality string  → append 'aug'
      - Major (default): everything else                  → no suffix

    Returns a set like: {'F', 'Cm', 'Bb', 'Dm', 'Adim'}
    """
    chord_ids = set()
    for section in parsed_tab.get('sections', []):
        for block in section.get('blocks', []):
            for chord in block.get('chords', []):
                # Extract root
                if len(chord) >= 2 and chord[1] in ('#', 'b'):
                    root = chord[:2]
                    quality_str = chord[2:]
                elif chord:
                    root = chord[0]
                    quality_str = chord[1:]
                else:
                    continue

                # Strip slash-bass from quality (e.g. 'G/B' → quality of 'G')
                quality_str = quality_str.split('/')[0]

                # Classify base quality
                is_minor = quality_str.startswith('m') and not quality_str.startswith('maj')
                is_dim   = 'dim' in quality_str or 'b5' in quality_str
                is_aug   = 'aug' in quality_str or '#5' in quality_str

                if is_minor:
                    chord_id = root + 'm'
                elif is_dim:
                    chord_id = root + 'dim'
                elif is_aug:
                    chord_id = root + 'aug'
                else:
                    chord_id = root   # plain major

                chord_ids.add(chord_id)
    return chord_ids


def consistency_score(tab, all_tabs, base_scores):
    """
    Measures how much one tab's chord vocabulary agrees with the others,
    weighted by each other tab's base score.

    Uses Jaccard similarity: |intersection| / |union|
      → 1.0 means identical chord sets
      → 0.0 means no chords in common

    Tabs with higher base scores (more votes, better rating) get more
    influence over the consensus — their agreement matters more.

    Returns a float 0.0–1.0.
    """
    my_roots = get_chord_set(tab)
    if not my_roots:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for other_tab, other_score in zip(all_tabs, base_scores):
        if other_tab is tab:
            continue
        other_roots = get_chord_set(other_tab)
        if not other_roots:
            continue

        intersection = len(my_roots & other_roots)
        union        = len(my_roots | other_roots)
        jaccard      = intersection / union if union > 0 else 0.0

        weighted_sum += jaccard * other_score
        total_weight += other_score

    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


# ─────────────────────────────────────────────
# RANK TABS (full pipeline)
# ─────────────────────────────────────────────

def rank_tabs(parsed_tabs, consistency_weight=2.0):
    """
    Scores and ranks a list of parsed tabs.

    Each tab gets three score fields:
      'score_base'        — rating + votes + source + completeness
      'score_consistency' — chord agreement with other tabs (0–1)
      'score'             — final = base + (consistency × weight)

    consistency_weight=2.0 means a perfectly consistent tab gets +2.0
    added to its base score — meaningful but not overwhelming.
    """
    if not parsed_tabs:
        return []

    base_scores = [score_tab(t) for t in parsed_tabs]

    for i, tab in enumerate(parsed_tabs):
        base        = base_scores[i]
        consistency = consistency_score(tab, parsed_tabs, base_scores)
        final       = round(base + (consistency * consistency_weight), 4)

        tab['score_base']        = base
        tab['score_consistency'] = consistency
        tab['score']             = final

    parsed_tabs.sort(key=lambda t: t['score'], reverse=True)
    return parsed_tabs


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────

def rank_tabs_with_report(parsed_tabs, consistency_weight=2.0):
    """
    Ranks tabs and prints a readable score breakdown.
    Returns the ranked list.
    """
    ranked = rank_tabs(parsed_tabs, consistency_weight)

    print("\n" + "═" * 62)
    print("  TAB SCORING RESULTS  (base + consistency)")
    print("═" * 62)

    for i, tab in enumerate(ranked):
        meta   = tab.get('metadata', {})
        source = meta.get('source', 'unknown')
        rating = meta.get('rating', 'N/A')
        votes  = meta.get('vote_count', 0)
        vtype  = meta.get('version_type', 'unspecified')
        key    = meta.get('transposed_to') or meta.get('key', 'unknown')
        c_bonus = round(tab['score_consistency'] * consistency_weight, 4)

        print(f"\n  Rank #{i + 1}")
        print(f"  {'Final score:':<22} {tab['score']}")
        print(f"  {'  Base score:':<22} {tab['score_base']}")
        print(f"  {'  Consistency (0-1):':<22} {tab['score_consistency']}  → +{c_bonus}")
        print(f"  {'Source:':<22} {source}")
        print(f"  {'Version type:':<22} {vtype}")
        print(f"  {'Key (normalized):':<22} {key}")
        print(f"  {'Rating:':<22} {rating} ({votes} votes)")
        print(f"  {'Sections found:':<22} {len(tab.get('sections', []))}")
        print(f"  {'Chord vocabulary:':<22} {sorted(get_chord_set(tab))}")

        if tab.get('warnings'):
            for w in tab['warnings']:
                print(f"  ⚠  {w}")

    print("\n" + "═" * 62 + "\n")
    return ranked

