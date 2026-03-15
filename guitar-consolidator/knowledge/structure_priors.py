"""
knowledge/structure_priors.py
──────────────────────────────
Tracks observed section sequences across all processed songs.

After N songs, the system knows that:
  - A section after a chorus is most likely another verse or an outro
  - A section after a bridge is almost always a chorus
  - A standalone 8-block unlabeled section early in a song is probably a verse

This knowledge is used in two places:
  1. Resolving ambiguous section labels ('unlabeled', 'hook', 'refrain'
     that weren't caught by detect_section) by looking at what surrounds them
  2. Flagging structural anomalies in incoming tabs as a quality signal

STORAGE FORMAT
──────────────
JSON at knowledge/data/structure_priors.json

{
    "version": 1,
    "total_songs": 42,
    "section_counts": {
        "verse":   { "count": 89, "avg_blocks": 4.2 },
        "chorus":  { "count": 76, "avg_blocks": 3.1 },
        ...
    },
    "transitions": {
        "intro -> verse":        34,
        "verse -> chorus":       71,
        "chorus -> verse":       58,
        "chorus -> bridge":      29,
        "bridge -> chorus":      28,
        "verse -> outro":        18,
        ...
    },
    "common_structures": {
        "verse chorus verse chorus bridge chorus": 21,
        "verse chorus verse chorus":              14,
        ...
    }
}

QUERY INTERFACE
───────────────
priors.likely_next(current_section)
    → list of (section_name, probability) sorted by probability
      e.g. [('chorus', 0.73), ('verse', 0.18), ('outro', 0.09)]

priors.score_structure(structure)
    → float 0.0–1.0: how common is this exact section sequence.
      Used to rank competing structures when tabs disagree.

priors.infer_label(blocks, preceding_section, following_section)
    → best guess for an unlabeled section's type, or None if uncertain.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'structure_priors.json')

# Sections we consider structurally "known" — map aliases to canonical names
CANONICAL = {
    'verse':       'verse',
    'chorus':      'chorus',
    'bridge':      'bridge',
    'intro':       'intro',
    'outro':       'outro',
    'pre-chorus':  'pre-chorus',
    'prechorus':   'pre-chorus',
    'interlude':   'interlude',
    'solo':        'solo',
    'hook':        'chorus',    # treat hook as chorus for structure purposes
    'refrain':     'chorus',
    'tag':         'outro',
    'coda':        'outro',
}

# Minimum transition observations before we use them for inference
MIN_TRANSITION_OBS = 3


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        return {
            'version': 1,
            'total_songs': 0,
            'section_counts': {},
            'transitions': {},
            'common_structures': {}
        }
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _canonical(name: str) -> str:
    """Normalize a section name to its canonical form, stripping numbers."""
    base = name.rsplit('_', 1)[0] if (
        name.rsplit('_', 1)[-1].isdigit() and '_' in name
    ) else name
    return CANONICAL.get(base.lower(), base.lower())


def record_song(consolidated: dict) -> None:
    """
    Record the section structure of a consolidated song.

    Extracts the normalized sequence of section types, records:
      - How often each section type appears (with block counts)
      - Every adjacent-section transition
      - The full structure as a canonical sequence string
    """
    structure = consolidated.get('structure', [])
    sections  = consolidated.get('sections', {})

    if not structure:
        return

    data = _load()

    # Build canonical section sequence
    canonical_seq = []
    for norm_name, orig_name in structure:
        cn = _canonical(norm_name)
        canonical_seq.append(cn)

        # Record section type count and average block count
        template   = sections.get(orig_name) or sections.get(norm_name, [])
        block_count = len(template)

        if cn not in data['section_counts']:
            data['section_counts'][cn] = {'count': 0, 'block_count_sum': 0}
        data['section_counts'][cn]['count']           += 1
        data['section_counts'][cn]['block_count_sum'] += block_count

    # Record pairwise transitions
    for i in range(len(canonical_seq) - 1):
        transition = f"{canonical_seq[i]} -> {canonical_seq[i+1]}"
        data['transitions'][transition] = data['transitions'].get(transition, 0) + 1

    # Record full structure as a string
    structure_str = ' '.join(canonical_seq)
    data['common_structures'][structure_str] = (
        data['common_structures'].get(structure_str, 0) + 1
    )

    data['total_songs'] += 1
    _save(data)


def likely_next(current_section: str) -> List[Tuple[str, float]]:
    """
    Given the current section type, returns the likely next sections
    sorted by probability.

    Returns [] if no data exists for this section.
    Each entry is (section_name, probability) where probabilities sum to 1.0.

    Example:
        likely_next('chorus') → [('verse', 0.52), ('bridge', 0.29), ('outro', 0.19)]
    """
    data = _load()
    cn   = _canonical(current_section)

    # Find all transitions starting from this section
    matching = {
        k: v for k, v in data.get('transitions', {}).items()
        if k.startswith(f"{cn} -> ") and v >= MIN_TRANSITION_OBS
    }
    if not matching:
        return []

    total = sum(matching.values())
    result = []
    for transition, count in matching.items():
        next_section = transition.split(' -> ', 1)[1]
        result.append((next_section, round(count / total, 4)))

    return sorted(result, key=lambda x: -x[1])


def score_structure(structure: List[Tuple[str, str]]) -> float:
    """
    Returns a score 0.0–1.0 for how common this section sequence is.

    1.0 = this is the single most common structure seen.
    0.0 = never seen.
    0.5 = no data (neutral).

    Used to rank competing structures when tabs disagree on song layout.
    """
    data = _load()
    if not data.get('common_structures'):
        return 0.5

    canonical_seq = [_canonical(norm) for norm, orig in structure]
    structure_str = ' '.join(canonical_seq)

    count     = data['common_structures'].get(structure_str, 0)
    max_count = max(data['common_structures'].values())

    if count == 0:
        return 0.0
    return round(count / max_count, 4)


def infer_label(
    block_count: int,
    preceding: Optional[str],
    following: Optional[str],
) -> Optional[str]:
    """
    Given an unlabeled section's block count and its neighbors,
    returns the most likely section type, or None if uncertain.

    Logic:
      1. Use transition data: what section typically follows `preceding`
         AND precedes `following`?
      2. Use block count as a secondary signal: intros/outros are short
         (1-2 blocks), verses/choruses are longer (3-6 blocks).
      3. Return None if the two signals conflict or confidence is low.
    """
    data = _load()

    candidates = {}

    # Signal 1: what typically follows the preceding section?
    if preceding:
        for section, prob in likely_next(preceding):
            candidates[section] = candidates.get(section, 0) + prob * 2.0

    # Signal 2: what typically precedes the following section?
    if following:
        cn_following = _canonical(following)
        # Find all transitions that lead INTO following section
        for transition, count in data.get('transitions', {}).items():
            if transition.endswith(f" -> {cn_following}") and count >= MIN_TRANSITION_OBS:
                prev_section = transition.split(' -> ')[0]
                total_from_prev = sum(
                    v for k, v in data['transitions'].items()
                    if k.startswith(f"{prev_section} -> ")
                )
                prob = count / total_from_prev if total_from_prev > 0 else 0
                candidates[prev_section] = candidates.get(prev_section, 0) + prob

    if not candidates:
        return None

    # Signal 3: block count heuristic
    # Intro/outro: ≤2 blocks; verse/chorus: 3-6; bridge: 2-4
    block_hints = {}
    if block_count <= 2:
        block_hints = {'intro': 0.5, 'outro': 0.5}
    elif block_count <= 6:
        block_hints = {'verse': 0.5, 'chorus': 0.4, 'bridge': 0.1}
    else:
        block_hints = {'verse': 0.8, 'bridge': 0.2}

    # Combine with weak weight
    for section, hint in block_hints.items():
        candidates[section] = candidates.get(section, 0) + hint * 0.5

    best      = max(candidates, key=lambda k: candidates[k])
    best_score = candidates[best]
    second     = sorted(candidates.values(), reverse=True)
    second_score = second[1] if len(second) > 1 else 0

    # Only return a label if we have meaningful confidence above runner-up
    if best_score - second_score < 0.20:
        return None

    return best


def stats() -> dict:
    """Returns a summary of learned structure priors."""
    data = _load()

    section_avgs = {}
    for name, entry in data.get('section_counts', {}).items():
        n = entry['count']
        section_avgs[name] = {
            'appearances': n,
            'avg_blocks': round(entry['block_count_sum'] / n, 1) if n else 0
        }

    top_structures = sorted(
        data.get('common_structures', {}).items(),
        key=lambda x: -x[1]
    )[:10]

    top_transitions = sorted(
        data.get('transitions', {}).items(),
        key=lambda x: -x[1]
    )[:10]

    return {
        'total_songs':      data.get('total_songs', 0),
        'section_types':    section_avgs,
        'top_structures':   top_structures,
        'top_transitions':  top_transitions,
    }
