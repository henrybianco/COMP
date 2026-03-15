"""
consolidator/chord_diagrams.py
───────────────────────────────
Generates ASCII chord diagrams for guitar, given a list of chord names.

THREE-TIER VOICING LOOKUP
──────────────────────────
Tier 1 — Source voicings: if the source tab included voicing annotations
  like "G  3-5-5-4-3-3" in its notes section, those are used verbatim.
  The transcriber already did the work; we honour it.

Tier 2 — Curated database: ~130 standard chord shapes covering all common
  open chords, barre chords, and 7th/sus/add variants.  These are the
  voicings a guitarist would actually look up in a chord book.

Tier 3 — Algorithmic fallback: for exotic chords not in the database
  (altered dominants, polychords, etc.), we compute a playable voicing
  by searching the fretboard for a fingering that contains all required
  chord tones, stays within a 4-fret window, and starts on the root.

DIAGRAM FORMAT
──────────────
Each chord is rendered as a 10-line ASCII block:

    Cmaj7
    x
    ╔═╦═╦═╦═╦═╦═╗
    ║ ║ ║ ║ ║●║ ║   ← fret 1
    ╠═╬═╬═╬═╬═╬═╣
    ║ ║ ║ ║ ║ ║ ║   ← fret 2
    ╠═╬═╬═╬═╬═╬═╣
    ║ ║ ║●║ ║ ║ ║   ← fret 3
    ╠═╬═╬═╬═╬═╬═╣
    ║ ║●║ ║ ║ ║ ║   ← fret 4
    ╚═╩═╩═╩═╩═╩═╝

  (E A D G B e  — low string left)

For barre chords at high positions, a fret number appears to the right of
the top border:

    Bm
    x
    ╔═╦═╦═╦═╦═╦═╗  2fr
    ║●║ ║ ║ ║ ║●║   ← fret 2
    ...

LAYOUT
──────
format_chord_section() lays diagrams side-by-side in rows of 3,
fitting within LINE_WIDTH=72.  It returns the full multi-row string
ready to drop into the formatter.

NOTES-SECTION VOICING PARSING
──────────────────────────────
parse_source_voicings(notes_list) extracts voicing annotations from
the 'notes' field of a parsed tab.  UG tabs often include lines like:
    G     3-5-5-4-3-3
    Cm    8-10-10-8-8-8
These are parsed into the same format as the curated database entries.
"""

import re
from typing import Dict, List, Optional, Tuple

# Voicing format throughout: list of 6 ints or None (muted).
# Index 0 = string 6 (low E), index 5 = string 1 (high e).
Voicing = List[Optional[int]]


# ─────────────────────────────────────────────────────────────────────────
# TIER 2 — CURATED VOICING DATABASE
# ─────────────────────────────────────────────────────────────────────────
# Notation: list of 6 values, None = muted string.
# All standard open-position chords plus the most common barre variants.
# Keys use the same spelling as the transposer outputs (sharps preferred
# in sharp keys, flats in flat keys).

_VOICING_DB: Dict[str, Voicing] = {
    # ── Open major chords ────────────────────────────────────────────────
    'C':   [None, 3, 2, 0, 1, 0],
    'D':   [None, None, 0, 2, 3, 2],
    'E':   [0, 2, 2, 1, 0, 0],
    'F':   [1, 3, 3, 2, 1, 1],        # barre 1
    'G':   [3, 2, 0, 0, 0, 3],
    'A':   [None, 0, 2, 2, 2, 0],
    'B':   [None, 2, 4, 4, 4, 2],     # barre 2
    'Bb':  [None, 1, 3, 3, 3, 1],     # barre 1
    'Eb':  [None, 6, 8, 8, 8, 6],     # barre 6
    'Ab':  [None, None, 6, 5, 4, 4],  # barre 4 (mini)
    'Db':  [None, 4, 6, 6, 6, 4],     # barre 4
    'Gb':  [None, 9, 11, 11, 11, 9],  # barre 9 (enharmonic F#)
    'F#':  [None, 9, 11, 11, 11, 9],  # barre 9
    'C#':  [None, 4, 6, 6, 6, 4],     # barre 4 (enharmonic Db)

    # ── Open minor chords ────────────────────────────────────────────────
    'Am':  [None, 0, 2, 2, 1, 0],
    'Bm':  [None, 2, 4, 4, 3, 2],     # barre 2
    'Cm':  [None, 3, 5, 5, 4, 3],     # barre 3
    'Dm':  [None, None, 0, 2, 3, 1],
    'Em':  [0, 2, 2, 0, 0, 0],
    'Fm':  [1, 3, 3, 1, 1, 1],        # barre 1
    'Gm':  [3, 5, 5, 3, 3, 3],        # barre 3
    'F#m': [None, None, 4, 6, 7, 5],  # open position
    'C#m': [None, 4, 6, 6, 5, 4],     # barre 4
    'G#m': [None, None, 6, 8, 9, 7],  # barre pos

    # ── Dominant 7th chords ──────────────────────────────────────────────
    'C7':  [None, 3, 2, 3, 1, 0],
    'D7':  [None, None, 0, 2, 1, 2],
    'E7':  [0, 2, 0, 1, 0, 0],
    'F7':  [1, 3, 1, 2, 1, 1],        # barre 1
    'G7':  [3, 2, 0, 0, 0, 1],
    'A7':  [None, 0, 2, 0, 2, 0],
    'B7':  [None, 2, 1, 2, 0, 2],

    # ── Major 7th chords ────────────────────────────────────────────────
    'Cmaj7':  [None, 3, 2, 0, 0, 0],
    'Dmaj7':  [None, None, 0, 2, 2, 2],
    'Emaj7':  [0, 2, 1, 1, 0, 0],
    'Fmaj7':  [None, None, 3, 2, 1, 0],
    'Gmaj7':  [3, 2, 0, 0, 0, 2],
    'Amaj7':  [None, 0, 2, 1, 2, 0],
    'Bbmaj7': [None, 1, 3, 2, 3, 1],  # barre 1
    'Bmaj7':  [None, 2, 4, 3, 4, 2],  # barre 2

    # ── Minor 7th chords ────────────────────────────────────────────────
    'Am7':  [None, 0, 2, 0, 1, 0],
    'Bm7':  [None, 2, 4, 2, 3, 2],    # barre 2
    'Cm7':  [None, 3, 5, 3, 4, 3],    # barre 3
    'Dm7':  [None, None, 0, 2, 1, 1],
    'Em7':  [0, 2, 0, 0, 0, 0],
    'Fm7':  [1, 3, 1, 1, 1, 1],       # barre 1
    'Gm7':  [3, 5, 3, 3, 3, 3],       # barre 3
    'F#m7': [None, None, 4, 4, 5, 2],
    'C#m7': [None, 4, 6, 4, 5, 4],    # barre 4

    # ── Minor/major 7th ─────────────────────────────────────────────────
    'Ammaj7': [None, 0, 2, 1, 1, 0],
    'Dmmaj7': [None, None, 0, 2, 3, 1],  # same as Dm with maj7 = Dm shape

    # ── sus2 ────────────────────────────────────────────────────────────
    'Asus2': [None, 0, 2, 2, 0, 0],
    'Dsus2': [None, None, 0, 2, 3, 0],
    'Esus2': [0, 2, 4, 4, 0, 0],
    'Gsus2': [3, 2, 0, 0, 3, 3],
    'Csus2': [None, 3, 3, 0, 1, 1],

    # ── sus4 ────────────────────────────────────────────────────────────
    'Asus4': [None, 0, 2, 2, 3, 0],
    'Dsus4': [None, None, 0, 2, 3, 3],
    'Esus4': [0, 2, 2, 2, 0, 0],
    'Gsus4': [3, 3, 0, 0, 1, 3],
    'Csus4': [None, 3, 3, 0, 1, 1],
    'Fsus4': [None, None, 3, 3, 6, 6],  # barred
    'D7sus4': [None, None, 0, 2, 1, 3],
    'A7sus4': [None, 0, 2, 0, 3, 0],

    # ── add9 ────────────────────────────────────────────────────────────
    'Cadd9':  [None, 3, 2, 0, 3, 0],
    'Dadd9':  [None, None, 0, 2, 3, 0],
    'Gadd9':  [3, 2, 0, 2, 0, 3],
    'Aadd9':  [None, 0, 2, 4, 2, 0],
    'Emadd9': [0, 2, 2, 0, 0, 2],
    'Amadd9': [None, 0, 2, 2, 1, 3],

    # ── 6th chords ───────────────────────────────────────────────────────
    'C6':   [None, 3, 2, 2, 1, 0],
    'D6':   [None, None, 0, 2, 0, 2],
    'G6':   [3, 2, 0, 0, 0, 0],
    'A6':   [None, 0, 2, 2, 2, 2],
    'E6':   [0, 2, 2, 1, 2, 0],
    'Am6':  [None, 0, 2, 2, 1, 2],

    # ── diminished ───────────────────────────────────────────────────────
    'Bdim':  [None, 2, 0, 3, 3, None],
    'Cdim':  [None, 3, 4, 3, 2, None],
    'Adim':  [None, 0, 1, 2, 1, None],
    'Edim':  [0, 1, 2, 3, None, None],
    'Ddim':  [None, None, 0, 1, 0, 1],
    'F#dim': [None, None, 4, 5, 4, 5],
    'Gdim':  [3, None, 0, 1, 0, None],

    # ── augmented ────────────────────────────────────────────────────────
    'Caug':  [None, 3, 2, 1, 1, 0],
    'Daug':  [None, None, 0, 3, 3, 2],
    'Eaug':  [0, 3, 2, 1, 1, 0],
    'Faug':  [1, 0, 3, 2, 2, 1],
    'Gaug':  [3, 2, 1, 0, 0, 3],
    'Aaug':  [None, 0, 3, 2, 2, 1],

    # ── Power chords (5th) ───────────────────────────────────────────────
    'A5':  [None, 0, 2, 2, None, None],
    'E5':  [0, 2, 2, None, None, None],
    'G5':  [3, 5, 5, None, None, None],
    'D5':  [None, None, 0, 2, 3, None],

    # ── Common slash chords ──────────────────────────────────────────────
    'C/E':  [0, 3, 2, 0, 1, 0],
    'C/G':  [3, 3, 2, 0, 1, 0],
    'D/F#': [2, 0, 0, 2, 3, 2],
    'E/B':  [None, 2, 2, 1, 0, 0],
    'F/C':  [None, 3, 3, 2, 1, 1],
    'G/B':  [None, 2, 0, 0, 0, 3],
    'G/F#': [2, 2, 0, 0, 0, 3],
    'Am/E': [0, 0, 2, 2, 1, 0],
    'Am/G': [3, 0, 2, 2, 1, 0],
    'Am/F#': [2, 0, 2, 2, 1, 0],
    'G/D':  [None, None, 0, 0, 0, 3],
    'Dm/C': [None, 3, 0, 2, 3, 1],
    'Bb/D': [None, 1, 3, 3, 3, None],  # 1st inversion Bb

    # ── Common 9th chords ────────────────────────────────────────────────
    'C9':  [None, 3, 2, 3, 3, 3],
    'D9':  [None, None, 0, 2, 1, 0],
    'G9':  [3, 0, 0, 0, 0, 1],
    'A9':  [None, 0, 2, 4, 2, 3],
    'E9':  [0, 2, 0, 1, 3, 2],
    'Dm9': [None, None, 0, 2, 1, 0],   # common simplified voicing
    'Am9': [None, 0, 2, 4, 1, 0],
    'Em9': [0, 2, 0, 2, 0, 0],
    'Cmaj9': [None, 3, 2, 0, 3, 0],
    'Gmaj9': [3, 2, 0, 2, 0, 2],

    # ── Common major 9th (Xmaj9 = Xmaj7 + 9) ────────────────────────────
    'Fmaj9':  [None, None, 3, 0, 1, 0],
    'Amaj9':  [None, 0, 2, 1, 0, 0],
    'Bmaj9':  [None, 2, 4, 3, 2, 2],
    'Dbmaj9': [None, 4, 3, 3, 3, None],

    # ── Dominant 9th / 13th / altered ────────────────────────────────────
    'G13': [3, 2, 0, 0, 0, 0],  # simplified
    'A13': [None, 0, 2, 0, 2, 2],

    # ── Half-diminished (m7b5) ───────────────────────────────────────────
    'Bm7b5':  [None, 2, 3, 2, 3, None],
    'Em7b5':  [0, 1, 2, 0, 3, None],
    'Am7b5':  [None, 0, 1, 2, 1, None],
    'Dm7b5':  [None, None, 0, 1, 0, 1],
    'F#m7b5': [None, None, 4, 5, 5, None],

    # ── Dim7 ─────────────────────────────────────────────────────────────
    'Bdim7':  [None, 2, 3, 2, 3, None],
    'Ddim7':  [None, None, 0, 1, 0, 1],
    'Fdim7':  [1, 2, 3, 1, None, None],
}

# Enharmonic aliases — map to the stored key
_ENHARMONIC: Dict[str, str] = {
    # Flat ↔ sharp major
    'Db':  'C#',  'C#':  'Db',
    'Eb':  'D#',  'D#':  'Eb',
    'Gb':  'F#',  'F#':  'Gb',
    'Ab':  'G#',  'G#':  'Ab',
    'Bb':  'A#',  'A#':  'Bb',
    # Flat ↔ sharp minor
    'Dbm': 'C#m', 'C#m': 'Dbm',
    'Ebm': 'D#m', 'D#m': 'Ebm',
    'Gbm': 'F#m', 'F#m': 'Gbm',
    'Abm': 'G#m', 'G#m': 'Abm',
    'Bbm': 'A#m', 'A#m': 'Bbm',
}


# ─────────────────────────────────────────────────────────────────────────
# TIER 1 — SOURCE VOICING PARSER
# ─────────────────────────────────────────────────────────────────────────

def parse_source_voicings(notes: List[str]) -> Dict[str, Voicing]:
    """
    Extracts chord voicings from a tab's 'notes' list.

    Handles three common UG formats:
        G     3-5-5-4-3-3        (dash-separated)
        Cm    8-10-10-8-8-8      (dash-separated, multi-digit frets)
        Dm9:  xx0560             (compact 6-char, no separators, colon after name)
        Fmaj7 x-x-3-2-1-0       (x values with dashes)

    Returns a dict of {chord_name: voicing}.
    """
    voicings: Dict[str, Voicing] = {}
    for note in notes:
        stripped = note.strip()

        # Try dash/space-separated format first: "Chord[:] N-N-N-N-N-N"
        m = re.match(
            r'^([A-G][b#]?[^\s:]*):?\s+((?:[\dx]+[-\s]){5}[\dx]+)\s*$',
            stripped,
            re.IGNORECASE,
        )
        if m:
            chord_name = m.group(1)
            fret_str   = m.group(2).strip()
            parts = re.split(r'[-\s]+', fret_str)
            if len(parts) == 6:
                frets: Voicing = []
                valid = True
                for p in parts:
                    if p.lower() == 'x':
                        frets.append(None)
                    elif p.isdigit():
                        frets.append(int(p))
                    else:
                        valid = False
                        break
                if valid:
                    voicings[chord_name] = frets
            continue

        # Try compact 6-char format: "Chord[:] xxNNNN" (single digits only)
        m2 = re.match(
            r'^([A-G][b#]?[^\s:]*):?\s+([x\d]{6})\s*$',
            stripped,
            re.IGNORECASE,
        )
        if m2:
            chord_name = m2.group(1)
            fret_str   = m2.group(2)
            frets = [None if c.lower() == 'x' else int(c) for c in fret_str]
            voicings[chord_name] = frets

    return voicings


# ─────────────────────────────────────────────────────────────────────────
# TIER 3 — ALGORITHMIC VOICING FINDER
# ─────────────────────────────────────────────────────────────────────────

# Standard tuning: pitch classes (mod 12) of each open string.
# Index 0 = low E, index 5 = high e.
_STRING_OPEN_PC = [4, 9, 2, 7, 11, 4]   # E A D G B E

# Core interval sets for quality parsing.
_QUALITY_INTERVALS: Dict[str, List[int]] = {
    '':       [0, 4, 7],
    'm':      [0, 3, 7],
    '7':      [0, 4, 7, 10],
    'maj7':   [0, 4, 7, 11],
    'm7':     [0, 3, 7, 10],
    'sus2':   [0, 2, 7],
    'sus4':   [0, 5, 7],
    'dim':    [0, 3, 6],
    'dim7':   [0, 3, 6, 9],
    'aug':    [0, 4, 8],
    'add9':   [0, 2, 4, 7],
    '6':      [0, 4, 7, 9],
    'm6':     [0, 3, 7, 9],
    '9':      [0, 4, 7, 10, 2],
    'm9':     [0, 3, 7, 10, 2],
    'maj9':   [0, 4, 7, 11, 2],
    'maj6':   [0, 4, 7, 9],
    '5':      [0, 7],
    '11':     [0, 4, 7, 10, 2, 5],
    'm7b5':   [0, 3, 6, 10],
    'mmaj7':  [0, 3, 7, 11],
}


def _chord_required_notes(chord_name: str) -> Tuple[Optional[int], int, Optional[int]]:
    """
    Returns (root_st, required_pitch_classes, bass_st_or_None).
    required_pitch_classes is a frozenset of 0-11 values.
    """
    from parser.transposer import NOTE_TO_SEMITONE

    # Handle slash chord
    slash_bass: Optional[int] = None
    name = chord_name
    if '/' in chord_name:
        parts = chord_name.split('/', 1)
        name  = parts[0]
        bm    = re.match(r'^([A-G][b#]?)', parts[1])
        if bm:
            slash_bass = NOTE_TO_SEMITONE.get(bm.group(1))

    m = re.match(r'^([A-G][b#]?)(.*)', name)
    if not m:
        return None, frozenset(), None

    root   = m.group(1)
    quality = m.group(2)
    root_st = NOTE_TO_SEMITONE.get(root)
    if root_st is None:
        return None, frozenset(), None

    # Match quality to interval template (longest match wins)
    intervals = [0, 4, 7]  # default major
    for tq in sorted(_QUALITY_INTERVALS.keys(), key=len, reverse=True):
        if quality.startswith(tq):
            intervals = _QUALITY_INTERVALS[tq]
            break

    required = frozenset((root_st + i) % 12 for i in intervals)
    if slash_bass is not None:
        required = required | frozenset([slash_bass])

    bass = slash_bass if slash_bass is not None else root_st
    return root_st, required, bass


def _find_voicing_algorithmic(chord_name: str) -> Optional[Voicing]:
    """
    Finds a playable guitar voicing by brute-force search over the fretboard.

    Search strategy:
      - For each starting fret position 0–8, look for a 6-string combination
        where all required chord tones are present, the span is ≤ 4 frets,
        the lowest sounding string plays the bass note, and at least 3 strings
        are sounding.
      - Score by: lowest fret position (prefer open/low), open string count
        (prefer ringing open strings), and sounding string count (prefer full).
      - Return the highest-scoring voicing found, or None if no valid voicing
        exists within the search range.
    """
    from itertools import product

    root_st, required, bass = _chord_required_notes(chord_name)
    if root_st is None or not required:
        return None

    best: Optional[Voicing] = None
    best_score: float = -999.0

    for start in range(0, 9):
        end = start + 4  # max 4-fret window

        # Per-string options: frets in window that produce a required note,
        # plus 'mute' (represented as sentinel value -1 here).
        string_opts: List[List[int]] = []
        for s in range(6):
            fret_range = range(0, end + 1) if start == 0 else range(start, end + 1)
            opts: List[int] = [-1]  # -1 = mute
            for fret in fret_range:
                pc = (_STRING_OPEN_PC[s] + fret) % 12
                if pc in required:
                    opts.append(fret)
            string_opts.append(opts)

        for combo in product(*string_opts):
            sounding = [(i, f) for i, f in enumerate(combo) if f != -1]
            if len(sounding) < 3:
                continue

            # All required tones must be present
            sounding_pcs = frozenset((_STRING_OPEN_PC[i] + f) % 12 for i, f in sounding)
            if not required.issubset(sounding_pcs):
                continue

            # Lowest sounding string must play the bass note
            lowest_pc = (_STRING_OPEN_PC[sounding[0][0]] + sounding[0][1]) % 12
            if lowest_pc != bass % 12:
                continue

            # Span check (ignoring open strings in span calculation)
            non_open = [f for _, f in sounding if f > 0]
            if non_open:
                if max(non_open) - min(non_open) > 3:
                    continue

            # Score
            lowest_fret  = min(non_open) if non_open else 0
            open_count   = sum(1 for _, f in sounding if f == 0)
            muted_count  = 6 - len(sounding)
            # Strong penalty for high positions; reward open strings and full voicings
            score = (
                -lowest_fret * 3
                + open_count * 2
                + len(sounding) * 1
                - muted_count * 0.5
            )

            if score > best_score:
                best_score = score
                best = [f if f != -1 else None for f in combo]

    return best


# ─────────────────────────────────────────────────────────────────────────
# PUBLIC LOOKUP API
# ─────────────────────────────────────────────────────────────────────────

def get_voicing(
    chord_name: str,
    source_voicings: Optional[Dict[str, Voicing]] = None,
) -> Optional[Voicing]:
    """
    Returns the best voicing for a chord name, using the three-tier lookup.

    Tier 1: source_voicings (from the tab's notes section)
    Tier 2: curated database
    Tier 3: algorithmic search

    Returns None if no voicing can be found (e.g. unrecognised chord name).
    """
    if source_voicings and chord_name in source_voicings:
        return source_voicings[chord_name]

    if chord_name in _VOICING_DB:
        return _VOICING_DB[chord_name]

    # Try enharmonic equivalent
    enharmonic = _ENHARMONIC.get(chord_name)
    if enharmonic and enharmonic in _VOICING_DB:
        return _VOICING_DB[enharmonic]

    # For slash chords: try the chord-only version first
    if '/' in chord_name:
        base = chord_name.split('/')[0]
        if base in _VOICING_DB:
            # Use the base voicing as a fallback (correct harmony, different bass)
            return _VOICING_DB[base]

    # Algorithmic fallback
    return _find_voicing_algorithmic(chord_name)


# ─────────────────────────────────────────────────────────────────────────
# ASCII RENDERER
# ─────────────────────────────────────────────────────────────────────────

_FRETS_TO_SHOW = 4   # how many fret rows to draw
_DIAGRAM_COL   = 17  # width of each diagram column (name + grid + gap)


def _render_single_diagram(chord_name: str, voicing: Voicing) -> List[str]:
    """
    Renders one chord diagram as a list of exactly (_FRETS_TO_SHOW * 2 + 4)
    lines, all left-padded to _DIAGRAM_COL characters so they can be
    concatenated side-by-side.

    Line structure (10 lines for 4-fret diagram):
        [0]  chord name
        [1]  mute/open indicators  (x or o above each string)
        [2]  top border (nut or 'Nfr' marker)
        [3..3+N*2-1]  fret rows and dividers
        [-1] bottom border
    """
    W = _DIAGRAM_COL
    lines: List[str] = []

    # ── Chord name ────────────────────────────────────────────────────────
    lines.append(chord_name[:W - 1])

    # ── Mute / open indicators ────────────────────────────────────────────
    # Only emit this row when there is something to show (muted or open strings).
    # Pure barre chords where all strings are fretted produce an all-spaces row
    # after rstrip — emitting it as a blank line makes diagrams look ragged.
    # We still append an empty string to preserve fixed diagram height.
    has_indicators = any(f is None or f == 0 for f in voicing)
    if has_indicators:
        ind = ''
        for f in voicing:
            ind += 'x ' if f is None else ('o ' if f == 0 else '  ')
        lines.append(ind.rstrip())
    else:
        lines.append('')   # fixed-height placeholder

    # ── Fret window ───────────────────────────────────────────────────────
    # Rules:
    #   1. If all fretted notes fit in frets 1-4, start at fret 1 (open/low position).
    #   2. If the highest fret exceeds the 4-fret window from fret 1, OR if there
    #      are no open strings, shift the window up to start at min(played).
    #      Show a 'Nfr' label so the player knows where on the neck to position.
    played     = [f for f in voicing if f is not None and f > 0]
    has_open   = any(f == 0 for f in voicing if f is not None)
    start_fret = 1

    if played:
        min_fret = min(played)
        max_fret = max(played)
        # Need to shift if: frets don't fit from fret1, or it's a pure barre chord
        needs_shift = (max_fret > _FRETS_TO_SHOW) or (not has_open and min_fret > 1)
        if needs_shift:
            start_fret = min_fret

    end_fret = start_fret + _FRETS_TO_SHOW - 1

    # ── Top border ────────────────────────────────────────────────────────
    border = '╔═╦═╦═╦═╦═╦═╗'
    if start_fret > 1:
        lines.append(f'{border} {start_fret}fr')
    else:
        lines.append(border)

    # ── Fret rows ─────────────────────────────────────────────────────────
    for fret in range(start_fret, end_fret + 1):
        row = '║'
        for s in range(6):
            v = voicing[s]
            row += '●║' if (v is not None and v == fret) else ' ║'
        lines.append(row)
        if fret < end_fret:
            lines.append('╠═╬═╬═╬═╬═╬═╣')

    lines.append('╚═╩═╩═╩═╩═╩═╝')

    # Pad each line to _DIAGRAM_COL for clean side-by-side tiling
    return [line.ljust(W) for line in lines]


def _render_placeholder(chord_name: str) -> List[str]:
    """Renders a blank diagram for chords with no known voicing."""
    W = _DIAGRAM_COL
    n_rows = _FRETS_TO_SHOW * 2 + 4  # same height as a real diagram
    lines = [chord_name[:W - 1].ljust(W)]
    lines.append('(no diagram)'.ljust(W))
    for _ in range(n_rows - 2):
        lines.append(''.ljust(W))
    return lines


# ─────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────

CHORDS_PER_ROW = 3
LINE_WIDTH      = 72
_LEFT_INDENT    = '  '


def format_chord_section(
    chord_names: List[str],
    source_voicings: Optional[Dict[str, Voicing]] = None,
) -> str:
    """
    Renders all chords in `chord_names` as a section of ASCII diagrams,
    laid out in rows of CHORDS_PER_ROW (3), fitting within LINE_WIDTH.

    Returns the full multi-row string, ready to insert into the formatter.
    """
    if not chord_names:
        return ''

    # Remove duplicates while preserving order
    seen: set = set()
    unique_chords = [c for c in chord_names if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    output_rows: List[str] = []
    col_gap = '   '  # gap between diagram columns

    for batch_start in range(0, len(unique_chords), CHORDS_PER_ROW):
        batch = unique_chords[batch_start : batch_start + CHORDS_PER_ROW]

        # Render each diagram in the batch
        diagrams: List[List[str]] = []
        for chord in batch:
            voicing = get_voicing(chord, source_voicings)
            if voicing is not None:
                diagrams.append(_render_single_diagram(chord, voicing))
            else:
                diagrams.append(_render_placeholder(chord))

        # All diagrams have the same height; zip them side-by-side
        height = max(len(d) for d in diagrams)
        for d in diagrams:
            while len(d) < height:
                d.append(''.ljust(_DIAGRAM_COL))

        for line_idx in range(height):
            combined = _LEFT_INDENT + col_gap.join(
                d[line_idx].rstrip() for d in diagrams
            )
            output_rows.append(combined.rstrip())

        output_rows.append('')  # blank line between rows of diagrams

    return '\n'.join(output_rows)
