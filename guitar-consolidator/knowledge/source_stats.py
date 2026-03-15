"""
knowledge/source_stats.py
─────────────────────────
Tracks observed consistency scores per source domain, replacing the
hand-coded SOURCE_WEIGHTS dict in scoring_engine.py once enough data
has accumulated.

DESIGN
──────
The hard-coded constants in scoring_engine.py serve as PRIORS — our
best guess before we've seen any data. As the system processes songs,
it records the consistency score each source achieved, and the live
weight converges toward the empirical mean.

Convergence formula:
    weight = prior_weight * (1 - alpha) + observed_mean * alpha

where alpha = min(n_observations / CONVERGENCE_THRESHOLD, 1.0)

At 0 observations: weight = prior (100% prior)
At CONVERGENCE_THRESHOLD observations: weight = observed_mean (100% data)
In between: blend proportionally

This means the hard-coded constants remain fully in effect for new sources
until we have enough evidence to trust the data.

STORAGE FORMAT
──────────────
JSON at knowledge/data/source_stats.json

{
    "version": 1,
    "sources": {
        "ultimate_guitar": {
            "observations": 47,
            "consistency_sum": 38.2,
            "consistency_mean": 0.812,
            "prior_weight": 1.0,
            "live_weight": 0.987
        },
        ...
    },
    "authors": {
        "tab_author_username": {
            "observations": 3,
            "consistency_sum": 2.8,
            "consistency_mean": 0.933
        }
    }
}

CONVERGENCE THRESHOLD
─────────────────────
We need at least this many observations before the data starts
overriding the prior. Set conservatively — 20 songs from a source
is meaningful; 3 is noise.
"""

import json
import os
from typing import Optional

DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'source_stats.json')

# Prior weights — same as the hand-coded constants in scoring_engine.py
# These are the starting point before any data is accumulated.
PRIOR_WEIGHTS = {
    'ultimate_guitar':          1.00,
    'ultimate_guitar_official': 1.20,
    'songsterr':                1.25,
    'echords':                  0.85,
    'chordie':                  0.80,
    'azchords':                 0.80,
    'unknown':                  0.75,
}
DEFAULT_PRIOR = 0.80

# How many observations before the data fully overrides the prior
CONVERGENCE_THRESHOLD = 20


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        return {'version': 1, 'sources': {}, 'authors': {}}
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _compute_live_weight(prior: float, observed_mean: float, n: int) -> float:
    """
    Blend prior and observed mean based on how much data we have.
    Alpha rises linearly from 0 → 1 as observations approach CONVERGENCE_THRESHOLD.
    """
    alpha = min(n / CONVERGENCE_THRESHOLD, 1.0)
    return round(prior * (1.0 - alpha) + observed_mean * alpha, 4)


def record_song(consolidated: dict, ranked_tabs: list) -> None:
    """
    Record source consistency scores from a completed consolidation run.

    For each tab that contributed to the consolidation:
      - Look up its source and author from metadata
      - Record its consistency score (score_consistency field)
    """
    data = _load()

    for tab in ranked_tabs:
        meta        = tab.get('metadata', {})
        source      = meta.get('source', 'unknown').lower().replace(' ', '_')
        author      = meta.get('author', '').strip()
        consistency = tab.get('score_consistency', None)

        if consistency is None:
            continue

        # ── Record by source ────────────────────────────────────────────────
        if source not in data['sources']:
            prior = PRIOR_WEIGHTS.get(source, DEFAULT_PRIOR)
            data['sources'][source] = {
                'observations':      0,
                'consistency_sum':   0.0,
                'consistency_mean':  prior,  # start at prior until data arrives
                'prior_weight':      prior,
                'live_weight':       prior,
            }

        entry = data['sources'][source]
        entry['observations']    += 1
        entry['consistency_sum'] += round(consistency, 4)
        entry['consistency_mean'] = round(
            entry['consistency_sum'] / entry['observations'], 4
        )
        entry['live_weight'] = _compute_live_weight(
            entry['prior_weight'],
            entry['consistency_mean'],
            entry['observations']
        )

        # ── Record by author (if named) ─────────────────────────────────────
        if author:
            key = author.lower()
            if key not in data['authors']:
                data['authors'][key] = {
                    'observations':     0,
                    'consistency_sum':  0.0,
                    'consistency_mean': 0.5,
                }
            a = data['authors'][key]
            a['observations']    += 1
            a['consistency_sum'] += round(consistency, 4)
            a['consistency_mean'] = round(
                a['consistency_sum'] / a['observations'], 4
            )

    _save(data)


def get_source_weight(source: str) -> float:
    """
    Returns the current best weight for a source domain.

    If we have enough observations, returns the data-driven live_weight.
    Otherwise returns the prior (hard-coded constant).

    This is a drop-in replacement for the SOURCE_WEIGHTS dict lookup
    in scoring_engine.py.
    """
    source_key = source.lower().replace(' ', '_')
    data       = _load()
    entry      = data.get('sources', {}).get(source_key)

    if entry is None:
        # Never seen this source — use default prior
        return PRIOR_WEIGHTS.get(source_key, DEFAULT_PRIOR)

    return entry['live_weight']


def get_author_weight(author: str) -> float:
    """
    Returns a small multiplicative bonus/penalty for a named author,
    based on their observed mean consistency.

    Scaled around 1.0:
      - mean consistency 1.0 → weight 1.10  (10% bonus)
      - mean consistency 0.5 → weight 1.00  (neutral)
      - mean consistency 0.0 → weight 0.90  (10% penalty)

    Only applied when the author has MIN_AUTHOR_OBSERVATIONS entries.
    Returns 1.0 (neutral) otherwise.
    """
    MIN_AUTHOR_OBSERVATIONS = 3
    if not author:
        return 1.0

    data  = _load()
    entry = data.get('authors', {}).get(author.lower())
    if not entry or entry['observations'] < MIN_AUTHOR_OBSERVATIONS:
        return 1.0

    mean   = entry['consistency_mean']
    weight = 1.0 + (mean - 0.5) * 0.20  # ±10% max adjustment
    return round(weight, 4)


def stats() -> dict:
    """Returns a summary of learned source weights vs priors."""
    data = _load()
    result = {}
    for source, entry in data.get('sources', {}).items():
        result[source] = {
            'observations': entry['observations'],
            'prior':        entry['prior_weight'],
            'observed_mean': entry['consistency_mean'],
            'live_weight':  entry['live_weight'],
            'delta':        round(entry['live_weight'] - entry['prior_weight'], 4),
        }
    return result
