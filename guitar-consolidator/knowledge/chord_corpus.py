"""
knowledge/chord_corpus.py
─────────────────────────
Persistent store of observed chord progressions.

After each consolidation, the system records every chord sequence it
produced (weighted by the confidence of that line). Future consolidations
query this corpus to resolve disputed lines: if two tabs disagree on a
chord and the corpus shows one option appearing frequently in similar
harmonic contexts while the other has never appeared, that's real signal.

STORAGE FORMAT
──────────────
JSON file at knowledge/data/chord_corpus.json

{
    "version": 1,
    "total_songs": 42,
    "progressions": {
        "G": {
            "G Em C D": { "count": 38, "confidence_sum": 31.2 },
            "G Am7 Em7 D": { "count": 12, "confidence_sum": 10.8 },
            ...
        },
        "D": { ... },
        ...
    },
    "bigrams": {
        "G": {
            "D -> G": 94,
            "G -> Em": 87,
            ...
        },
        ...
    }
}

WHAT GETS RECORDED
──────────────────
For each consolidated song line that has chords and a lyric:
  - The chord sequence as a space-joined string (normalized to the song key)
  - The key it was recorded in
  - The confidence score of that line

For each adjacent pair of chord sequences in the song:
  - A bigram: "predecessor_chords -> successor_chords"
  - Used to score how natural a transition is

QUERY INTERFACE
───────────────
corpus.score_progression(chords, key)
    → float 0.0–1.0: how often has this exact chord sequence appeared
      in this key, relative to the most common sequence in that key.
      Returns 0.5 (neutral) if the key has no data yet.

corpus.score_transition(prev_chords, next_chords, key)
    → float 0.0–1.0: how common is this chord-to-chord transition.
      Returns 0.5 if no data.

corpus.resolve_dispute(options, prev_chords, key)
    → the option from `options` with the highest combined
      progression + transition score. Returns None if no data.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'chord_corpus.json')

# Minimum number of observations before corpus scores override neutral 0.5
MIN_OBSERVATIONS = 3


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        return {
            'version': 1,
            'total_songs': 0,
            'progressions': {},
            'bigrams': {}
        }
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _chord_key(chords: List[str]) -> str:
    """Canonical string representation of a chord list."""
    return ' '.join(chords)


def record_song(consolidated: dict) -> int:
    """
    Record all chord sequences from a consolidated song into the corpus.

    Walks every section template, recording:
      - Each chord sequence (as a progression entry)
      - Each adjacent pair of sequences (as a bigram)

    Returns the number of new progression entries recorded.
    """
    key = consolidated.get('key', '').strip()
    if not key:
        return 0

    data = _load()

    if key not in data['progressions']:
        data['progressions'][key] = {}
    if key not in data['bigrams']:
        data['bigrams'][key] = {}

    prog_store   = data['progressions'][key]
    bigram_store = data['bigrams'][key]

    recorded = 0
    prev_chord_key = None

    # Walk structure in order so bigrams are positionally meaningful
    structure        = consolidated.get('structure', [])
    section_templates = consolidated.get('sections', {})

    for norm_name, orig_name in structure:
        template = section_templates.get(orig_name) or section_templates.get(norm_name, [])
        for line in template:
            chords     = line.get('chords', [])
            confidence = line.get('confidence', 0.5)

            if not chords:
                continue

            ck = _chord_key(chords)

            # Record progression
            if ck not in prog_store:
                prog_store[ck] = {'count': 0, 'confidence_sum': 0.0}
            prog_store[ck]['count']          += 1
            prog_store[ck]['confidence_sum'] += round(confidence, 4)
            recorded += 1

            # Record bigram (transition from previous line)
            if prev_chord_key is not None:
                bigram = f"{prev_chord_key} -> {ck}"
                bigram_store[bigram] = bigram_store.get(bigram, 0) + 1

            prev_chord_key = ck

    data['total_songs'] += 1
    _save(data)
    return recorded


def score_progression(chords: List[str], key: str) -> float:
    """
    Returns a score 0.0–1.0 for how common this chord sequence is in
    this key, relative to the most-seen sequence in that key.

    Returns 0.5 (neutral) if:
      - No data exists for this key
      - This sequence has fewer than MIN_OBSERVATIONS

    A score of 1.0 means this is the most commonly seen sequence.
    A score of 0.0 means it has never been seen.
    """
    data = _load()
    prog_store = data.get('progressions', {}).get(key, {})
    if not prog_store:
        return 0.5

    ck = _chord_key(chords)
    entry = prog_store.get(ck)
    if entry is None or entry['count'] < MIN_OBSERVATIONS:
        return 0.5

    max_count = max(e['count'] for e in prog_store.values())
    if max_count == 0:
        return 0.5

    return round(entry['count'] / max_count, 4)


def score_transition(prev_chords: List[str], next_chords: List[str], key: str) -> float:
    """
    Returns a score 0.0–1.0 for how common the transition
    prev_chords → next_chords is in this key.

    Returns 0.5 (neutral) if no data.
    """
    data = _load()
    bigram_store = data.get('bigrams', {}).get(key, {})
    if not bigram_store:
        return 0.5

    bigram    = f"{_chord_key(prev_chords)} -> {_chord_key(next_chords)}"
    count     = bigram_store.get(bigram, 0)
    if count < MIN_OBSERVATIONS:
        return 0.5

    max_count = max(bigram_store.values()) if bigram_store else 1
    return round(count / max_count, 4)


def resolve_dispute(
    options: List[List[str]],
    key: str,
    prev_chords: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Given a list of chord options for a disputed line, returns the option
    the corpus considers most likely given the key and preceding context.

    Scoring: 60% progression score + 40% transition score (if prev available)

    Returns None if the corpus has no data that distinguishes the options
    (all scores equal, or all neutral 0.5).
    """
    if not options:
        return None

    scores = []
    for option in options:
        prog_score = score_progression(option, key)

        if prev_chords:
            trans_score = score_transition(prev_chords, option, key)
            combined = (prog_score * 0.6) + (trans_score * 0.4)
        else:
            combined = prog_score

        scores.append(combined)

    # Only act if there's a meaningful difference (avoid acting on noise)
    best_score = max(scores)
    worst_score = min(scores)
    if best_score - worst_score < 0.10:
        return None  # corpus can't distinguish — leave vote_on_chords to decide

    best_idx = scores.index(best_score)
    return options[best_idx]


def stats(key: Optional[str] = None) -> dict:
    """Returns summary statistics about the corpus."""
    data = _load()
    result = {
        'total_songs': data.get('total_songs', 0),
        'keys': {}
    }
    for k, progs in data.get('progressions', {}).items():
        if key and k != key:
            continue
        total_observations = sum(e['count'] for e in progs.values())
        result['keys'][k] = {
            'unique_progressions': len(progs),
            'total_observations': total_observations,
            'top_5': sorted(
                [{'chords': ck, 'count': e['count']} for ck, e in progs.items()],
                key=lambda x: -x['count']
            )[:5]
        }
    return result
