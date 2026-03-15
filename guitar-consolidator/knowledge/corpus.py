"""
knowledge/corpus.py
───────────────────
Coordinator for all three knowledge stores.

This is the single public interface the rest of the pipeline uses.
Import this module; don't import chord_corpus, source_stats, or
structure_priors directly from outside the knowledge package.

USAGE
─────
At the end of every successful consolidation run:

    from knowledge.corpus import Corpus
    corpus = Corpus()
    corpus.record(consolidated, ranked_tabs)

During consolidation (in engine.py), to resolve a disputed line:

    suggestion = corpus.resolve_chord_dispute(options, key, prev_chords)

During scoring (in scoring_engine.py), to get a learned source weight:

    weight = corpus.source_weight(source_name)

During structure extraction, to infer an unlabeled section:

    label = corpus.infer_section_label(block_count, preceding, following)

COLD START BEHAVIOUR
────────────────────
All three stores return neutral values (0.5, prior weights, None) when
no data exists. The pipeline behaves identically to its pre-corpus state
until enough songs have been processed to generate meaningful signal.

The corpus starts influencing output meaningfully at approximately:
  - Chord corpus:    ~10 songs (enough progression variety)
  - Source stats:    ~20 songs per source (convergence threshold)
  - Structure priors: ~15 songs (enough transition data)
"""

import os
from typing import Dict, List, Optional, Tuple

from knowledge import chord_corpus, source_stats, structure_priors


class Corpus:
    """
    Coordinator for all knowledge stores.

    Instantiate once per session (or use the module-level singleton below).
    All I/O is through JSON files in knowledge/data/ — no database required.
    """

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, consolidated: dict, ranked_tabs: list) -> dict:
        """
        Record everything learnable from a completed consolidation.

        Called once per song after a successful consolidation run.
        Returns a summary of what was recorded.

        Parameters:
            consolidated   The output of consolidate() from engine.py
            ranked_tabs    The ranked tab list from rank_tabs() — needed
                           for per-tab source/consistency data
        """
        n_progressions = chord_corpus.record_song(consolidated)
        source_stats.record_song(consolidated, ranked_tabs)
        structure_priors.record_song(consolidated)

        return {
            'progressions_recorded': n_progressions,
            'sources_updated':       len(ranked_tabs),
            'structure_recorded':    True,
        }

    # ── Chord corpus queries ─────────────────────────────────────────────────

    def resolve_chord_dispute(
        self,
        options: List[List[str]],
        key: str,
        prev_chords: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Given a list of chord options for a disputed line, returns the
        corpus-preferred option, or None if the corpus can't distinguish.

        This is called from vote_on_chords() in engine.py when confidence
        is below the dispute threshold.
        """
        return chord_corpus.resolve_dispute(options, key, prev_chords)

    def score_progression(self, chords: List[str], key: str) -> float:
        """Corpus familiarity score for a chord sequence in a given key."""
        return chord_corpus.score_progression(chords, key)

    # ── Source stats queries ─────────────────────────────────────────────────

    def source_weight(self, source: str) -> float:
        """
        Returns the current best weight for a source domain.

        Initially returns the hard-coded prior from source_stats.PRIOR_WEIGHTS.
        Converges toward the observed empirical mean as data accumulates.

        Drop-in replacement for the SOURCE_WEIGHTS dict in scoring_engine.py.
        """
        return source_stats.get_source_weight(source)

    def author_weight(self, author: str) -> float:
        """
        Returns a small bonus/penalty multiplier for a named author.
        Returns 1.0 (neutral) until MIN_AUTHOR_OBSERVATIONS are recorded.
        """
        return source_stats.get_author_weight(author)

    # ── Structure prior queries ───────────────────────────────────────────────

    def likely_next_section(self, current_section: str) -> List[Tuple[str, float]]:
        """
        Returns the likely next sections after current_section,
        sorted by probability. Returns [] if no data.
        """
        return structure_priors.likely_next(current_section)

    def score_structure(self, structure: List[Tuple[str, str]]) -> float:
        """
        Returns how common this section sequence is (0.0–1.0).
        0.5 = no data (neutral).
        """
        return structure_priors.score_structure(structure)

    def infer_section_label(
        self,
        block_count: int,
        preceding: Optional[str] = None,
        following: Optional[str] = None,
    ) -> Optional[str]:
        """
        Infers the most likely label for an unlabeled section.
        Returns None if the corpus can't make a confident determination.
        """
        return structure_priors.infer_label(block_count, preceding, following)

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> str:
        """
        Returns a human-readable summary of all three knowledge stores.
        """
        lines = ['═' * 62, '  CORPUS REPORT', '═' * 62]

        # Chord corpus
        chord_stats = chord_corpus.stats()
        lines.append(f"\n  Songs processed: {chord_stats['total_songs']}")
        lines.append(  "  Chord corpus:")
        for key, info in sorted(chord_stats['keys'].items()):
            lines.append(
                f"    Key of {key}: {info['unique_progressions']} unique progressions, "
                f"{info['total_observations']} observations"
            )
            for entry in info['top_5'][:3]:
                lines.append(f"      [{entry['count']:3}x]  {entry['chords']}")

        # Source stats
        src_stats = source_stats.stats()
        if src_stats:
            lines.append("\n  Source weights (prior → learned):")
            for source, info in sorted(src_stats.items()):
                delta_str = f"{info['delta']:+.3f}" if info['delta'] != 0 else "  ——  "
                lines.append(
                    f"    {source:<30} {info['prior']:.3f} → {info['live_weight']:.3f}  "
                    f"({delta_str})  n={info['observations']}"
                )

        # Structure priors
        struct_stats = structure_priors.stats()
        if struct_stats['top_transitions']:
            lines.append("\n  Top section transitions:")
            for transition, count in struct_stats['top_transitions'][:8]:
                lines.append(f"    [{count:3}x]  {transition}")
        if struct_stats['top_structures']:
            lines.append("\n  Most common song structures:")
            for structure, count in struct_stats['top_structures'][:5]:
                lines.append(f"    [{count:3}x]  {structure}")

        lines.append('\n' + '═' * 62)
        return '\n'.join(lines)

    def data_path(self) -> str:
        """Returns the path to the knowledge data directory."""
        return os.path.join(os.path.dirname(__file__), 'data')


# ── Module-level singleton ───────────────────────────────────────────────────
# Use this in most cases rather than instantiating Corpus() yourself.
# It's stateless (all state is in JSON files) so sharing it is safe.

corpus = Corpus()
