"""
parser/tunings.py
─────────────────
Tuning registry and semitone offset calculator.

This module is the shared foundation for two future features:
  1. Chord name transposition between tunings (current phase)
  2. ASCII tablature fret-number transposition between tunings (tab phase)

DESIGN NOTES
────────────
A tuning is defined as 6 absolute MIDI-style semitone values, one per string,
ordered LOW to HIGH (string 6 → string 1, i.e. E → e in standard).

We use absolute values anchored to middle C (C4 = 60) so that comparisons
across tunings are unambiguous regardless of octave.

Standard EADGBE:
  E2=40  A2=45  D3=50  G3=55  B3=59  E4=64

CGCGCD (your primary alternate tuning):
  C2=36  G2=43  C3=48  G3=55  C4=60  D4=62

Per-string offset (standard → CGCGCD):
  String 6: 36-40 = -4   (E→C, down a major third)
  String 5: 43-45 = -2   (A→G, down a whole step)
  String 4: 48-50 = -2   (D→C, down a whole step)
  String 3: 55-55 =  0   (G→G, unchanged)
  String 2: 60-59 = +1   (B→C, up a half step)
  String 1: 62-64 = -2   (e→D, down a whole step)
"""

from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# NOTE → SEMITONE MAPPING
# ─────────────────────────────────────────────────────────────────

# Semitone values within an octave (C=0)
NOTE_TO_SEMITONE: Dict[str, int] = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11,
}

# Reverse map: semitone → preferred note name
# We maintain two variants — sharp and flat — for context-sensitive use
SEMITONE_TO_NOTE_SHARP: Dict[int, str] = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E',
    5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A',
    10: 'A#', 11: 'B',
}
SEMITONE_TO_NOTE_FLAT: Dict[int, str] = {
    0: 'C', 1: 'Db', 2: 'D', 3: 'Eb', 4: 'E',
    5: 'F', 6: 'Gb', 7: 'G', 8: 'Ab', 9: 'A',
    10: 'Bb', 11: 'B',
}


def note_to_semitone(note: str) -> int:
    """Convert a note name (e.g. 'C#', 'Bb') to its semitone value 0–11."""
    if note not in NOTE_TO_SEMITONE:
        raise ValueError(f"Unknown note name: {repr(note)}")
    return NOTE_TO_SEMITONE[note]


def semitone_to_note(semitone: int, use_flats: bool = False) -> str:
    """Convert a semitone value 0–11 to a note name."""
    semitone = semitone % 12
    if use_flats:
        return SEMITONE_TO_NOTE_FLAT[semitone]
    return SEMITONE_TO_NOTE_SHARP[semitone]


# ─────────────────────────────────────────────────────────────────
# TUNING DEFINITION
# ─────────────────────────────────────────────────────────────────

class Tuning:
    """
    Represents a 6-string guitar tuning as absolute MIDI semitone values,
    ordered string 6 (lowest) → string 1 (highest).

    Attributes:
        name        Human-readable name, e.g. "Standard", "CGCGCD"
        notes       List of note names, low→high, e.g. ['E','A','D','G','B','E']
        semitones   Absolute MIDI semitone values for each string
        category    e.g. 'standard', 'drop', 'open', 'modal', 'custom'
        description Optional notes about the tuning's character/use
    """

    def __init__(
        self,
        name: str,
        notes: List[str],       # note names only, e.g. ['E','A','D','G','B','E']
        octaves: List[int],     # octave numbers, e.g. [2,2,3,3,3,4]
        category: str = 'custom',
        description: str = '',
    ):
        assert len(notes) == 6, "Guitar tuning must have exactly 6 strings"
        assert len(octaves) == 6, "Must provide one octave per string"

        self.name        = name
        self.notes       = notes
        self.octaves     = octaves
        self.category    = category
        self.description = description

        # Absolute semitone = (octave * 12) + note_semitone
        # Anchored to C0 = 0, so C4 = 48, A4 = 57, E4 = 64, etc.
        self.semitones: List[int] = [
            (octave * 12) + NOTE_TO_SEMITONE[note]
            for note, octave in zip(notes, octaves)
        ]

    @property
    def notation(self) -> str:
        """Returns the compact tuning notation, e.g. 'CGCGCD'."""
        return ''.join(self.notes)

    def offset_from(self, other: 'Tuning') -> List[int]:
        """
        Returns per-string semitone offsets: self - other.

        A positive offset means this tuning's string is HIGHER than other's.
        A negative offset means this tuning's string is LOWER than other's.

        Used to transpose tab fret numbers: add the offset to each fret.
        If the offset is negative, the resulting fret would be negative
        (impossible), which means the note can't be played on that string
        in the target tuning — the tab transposer must handle this case.
        """
        return [s - o for s, o in zip(self.semitones, other.semitones)]

    def __repr__(self) -> str:
        return f"Tuning({self.name!r}, {self.notation!r})"


# ─────────────────────────────────────────────────────────────────
# TUNING REGISTRY
# ─────────────────────────────────────────────────────────────────

REGISTRY: Dict[str, Tuning] = {}


def register(tuning: Tuning) -> Tuning:
    """Add a tuning to the registry under its name and notation."""
    REGISTRY[tuning.name.lower()] = tuning
    REGISTRY[tuning.notation.upper()] = tuning
    return tuning


# ── Standard ────────────────────────────────────────────────────
STANDARD = register(Tuning(
    name        = 'Standard',
    notes       = ['E', 'A', 'D', 'G', 'B', 'E'],
    octaves     = [2,   2,   3,   3,   3,   4  ],
    category    = 'standard',
    description = 'Standard EADGBE tuning.',
))

# ── Drop tunings ────────────────────────────────────────────────
DROP_D = register(Tuning(
    name        = 'Drop D',
    notes       = ['D', 'A', 'D', 'G', 'B', 'E'],
    octaves     = [2,   2,   3,   3,   3,   4  ],
    category    = 'drop',
    description = 'Low E dropped to D. Power chords on 3 strings.',
))

DROP_C = register(Tuning(
    name        = 'Drop C',
    notes       = ['C', 'G', 'C', 'F', 'A', 'D'],
    octaves     = [2,   2,   3,   3,   3,   4  ],
    category    = 'drop',
    description = 'All strings down a whole step from Drop D.',
))

# ── Full-step-down tunings ───────────────────────────────────────
HALF_STEP_DOWN = register(Tuning(
    name        = 'Eb Standard',
    notes       = ['Eb', 'Ab', 'Db', 'Gb', 'Bb', 'Eb'],
    octaves     = [2,    2,    3,    3,    3,    4   ],
    category    = 'standard',
    description = 'All strings down a half step. Common in rock/metal.',
))

WHOLE_STEP_DOWN = register(Tuning(
    name        = 'D Standard',
    notes       = ['D', 'G', 'C', 'F', 'A', 'D'],
    octaves     = [2,   2,   3,   3,   3,   4  ],
    category    = 'standard',
    description = 'All strings down a whole step.',
))

# ── Open tunings ─────────────────────────────────────────────────
OPEN_G = register(Tuning(
    name        = 'Open G',
    notes       = ['D', 'G', 'D', 'G', 'B', 'D'],
    octaves     = [2,   2,   3,   3,   3,   4  ],
    category    = 'open',
    description = 'Open G chord. Used by Keith Richards, Robert Johnson.',
))

OPEN_D = register(Tuning(
    name        = 'Open D',
    notes       = ['D', 'A', 'D', 'F#', 'A', 'D'],
    octaves     = [2,   2,   3,   3,    3,   4  ],
    category    = 'open',
    description = 'Open D chord. Common in blues and slide guitar.',
))

OPEN_E = register(Tuning(
    name        = 'Open E',
    notes       = ['E', 'B', 'E', 'G#', 'B', 'E'],
    octaves     = [2,   2,   3,   3,    3,   4  ],
    category    = 'open',
    description = 'Open E chord. Same voicing as Open D, up a whole step.',
))

OPEN_A = register(Tuning(
    name        = 'Open A',
    notes       = ['E', 'A', 'E', 'A', 'C#', 'E'],
    octaves     = [2,   2,   3,   3,   3,    4  ],
    category    = 'open',
    description = 'Open A chord. Common in blues.',
))

# ── Modal / DADGAD-family ────────────────────────────────────────
DADGAD = register(Tuning(
    name        = 'DADGAD',
    notes       = ['D', 'A', 'D', 'G', 'A', 'D'],
    octaves     = [2,   2,   3,   3,   3,   4  ],
    category    = 'modal',
    description = 'Celtic/modal tuning. Dsus4 open chord. Pierre Bensusan.',
))

# ── YOUR PRIMARY TUNING ──────────────────────────────────────────
# CGCGCD — a Gsus2 / open C variant
# Per-string offsets from standard EADGBE:
#   String 6: E→C  = -4 semitones (down a major third)
#   String 5: A→G  = -2 semitones (down a whole step)
#   String 4: D→C  = -2 semitones (down a whole step)
#   String 3: G→G  =  0 semitones (unchanged)
#   String 2: B→C  = +1 semitone  (up a half step)
#   String 1: e→D  = -2 semitones (down a whole step)
CGCGCD = register(Tuning(
    name        = 'CGCGCD',
    notes       = ['C', 'G', 'C', 'G', 'C', 'D'],
    octaves     = [2,   2,   3,   3,   4,   4  ],
    category    = 'open',
    description = (
        'Open Csus2/Gsus2 voicing. Strings 3 and 4 remain at G and C '
        'as in standard. String 2 raised a half step B→C. '
        'Creates rich, ambiguous open voicings. '
        'Offsets from standard: -4, -2, -2, 0, +1, -2.'
    ),
))


# ─────────────────────────────────────────────────────────────────
# LOOKUP HELPERS
# ─────────────────────────────────────────────────────────────────

def get_tuning(name_or_notation: str) -> Optional[Tuning]:
    """
    Look up a tuning by name or notation string (case-insensitive).

    Examples:
        get_tuning('Standard')   -> STANDARD
        get_tuning('CGCGCD')     -> CGCGCD
        get_tuning('dadgad')     -> DADGAD
        get_tuning('Open G')     -> OPEN_G
        get_tuning('EADGBE')     -> STANDARD
    """
    key = name_or_notation.strip()
    # Try exact key (uppercase notation)
    if key.upper() in REGISTRY:
        return REGISTRY[key.upper()]
    # Try lowercase name
    if key.lower() in REGISTRY:
        return REGISTRY[key.lower()]
    # Try case-insensitive scan
    for k, v in REGISTRY.items():
        if k.lower() == key.lower():
            return v
    return None


def list_tunings(category: Optional[str] = None) -> List[Tuning]:
    """Return all registered tunings, optionally filtered by category."""
    seen = set()
    result = []
    for t in REGISTRY.values():
        if id(t) not in seen:
            seen.add(id(t))
            if category is None or t.category == category:
                result.append(t)
    return result


def parse_tuning_string(notation: str) -> Optional[Tuning]:
    """
    Parse a tuning notation string like 'D-G-C-F-A-d' or 'EADGBE'
    into a Tuning object, first checking the registry, then constructing
    a custom Tuning if not found.

    Handles both hyphenated ('D-G-C-F-A-d') and compact ('DADGAD') forms.
    Lowercase note letters are treated as the same note name (not octave markers).

    Returns None if the string doesn't look like a valid 6-string tuning.
    """
    # Normalise: remove hyphens, uppercase
    cleaned = notation.replace('-', '').upper()

    # Check registry first
    found = get_tuning(cleaned)
    if found:
        return found

    # Try to parse as a sequence of note names
    # Note names can be 1-2 chars: A-G optionally followed by # or b
    import re
    tokens = re.findall(r'[A-G][#b]?', cleaned, re.IGNORECASE)
    if len(tokens) != 6:
        return None

    # Assign octaves heuristically: start at octave 2 for string 6,
    # increment when the note is lower than the previous note
    # (i.e. we've crossed an octave boundary going low→high)
    notes  = [t.upper() for t in tokens]
    octave = 2
    octaves = []
    prev_semi = -1
    for note in notes:
        semi = NOTE_TO_SEMITONE.get(note, 0)
        if semi <= prev_semi:
            octave += 1
        octaves.append(octave)
        prev_semi = semi

    return Tuning(
        name     = cleaned,
        notes    = notes,
        octaves  = octaves,
        category = 'custom',
    )


# ─────────────────────────────────────────────────────────────────
# OFFSET CALCULATOR
# ─────────────────────────────────────────────────────────────────

def string_offsets(from_tuning: Tuning, to_tuning: Tuning) -> List[int]:
    """
    Returns the per-string semitone offset needed to go from one tuning
    to another.

    offset[i] = to_tuning.semitones[i] - from_tuning.semitones[i]

    A positive value means the target string is higher pitched.
    A negative value means the target string is lower pitched.

    In the tab transposition phase, these offsets are applied to raw
    fret numbers: new_fret = old_fret - offset[string_index]
    (subtract because a lower-pitched open string requires a higher
    fret number to reach the same pitch).
    """
    return to_tuning.offset_from(from_tuning)


def capo_adjusted_tuning(tuning: Tuning, capo_fret: int) -> Tuning:
    """
    Returns the effective tuning after applying a capo at the given fret.
    Each string's semitone value is raised by capo_fret.
    """
    adjusted_semitones_raw = [s + capo_fret for s in tuning.semitones]
    notes  = [semitone_to_note(s % 12) for s in adjusted_semitones_raw]
    octaves = [s // 12 for s in adjusted_semitones_raw]
    return Tuning(
        name     = f"{tuning.name} + capo {capo_fret}",
        notes    = notes,
        octaves  = octaves,
        category = tuning.category,
        description = f"Effective pitch with capo at fret {capo_fret}.",
    )


# ─────────────────────────────────────────────────────────────────
# CHORD TRANSPOSITION INTERFACE (current phase)
# ─────────────────────────────────────────────────────────────────

def sounding_key_offset(from_tuning: Tuning, to_tuning: Tuning) -> int:
    """
    Returns the number of semitones by which to transpose chord NAMES
    so that a song sounds the same when played in to_tuning as it did
    in from_tuning.

    This is the average pitch difference weighted toward the bass strings
    (strings 6, 5, 4) since those carry the root notes for most chord shapes.

    For simple all-strings-down tunings (Drop D, Eb Standard, D Standard),
    this equals the uniform offset. For complex tunings like CGCGCD where
    strings move by different amounts, we use the bass-string average.

    Example: Standard → D Standard (all strings -2 semitones)
        sounding_key_offset = -2
        A song in G standard becomes a song in F in D standard

    Example: Standard → CGCGCD
        Bass strings (6,5,4): -4, -2, -2 → average = -2.67 → rounds to -3
        Practical meaning: shapes that fingered a D chord in standard
        will finger approximately a B chord in CGCGCD.
        (The exact relationship depends on the specific chord shape.)
    """
    offsets = string_offsets(from_tuning, to_tuning)
    # Weight bass strings more heavily: strings 6, 5, 4 (indices 0, 1, 2)
    bass_offsets   = offsets[:3]
    treble_offsets = offsets[3:]
    weighted = (sum(bass_offsets) * 2 + sum(treble_offsets)) / (len(bass_offsets) * 2 + len(treble_offsets))
    return round(weighted)


# ─────────────────────────────────────────────────────────────────
# TAB TRANSPOSITION  (Phase 2 implementation)
# ─────────────────────────────────────────────────────────────────

# String label prefixes found in real tabs, ordered string 6→1 (low→high).
# Both 'E|' and 'e|' appear (low E vs high E); we normalise before matching.
_STRING_LABELS: List[str] = ['E', 'A', 'D', 'G', 'B', 'E']

# Regex to match a full tab string line, capturing:
#   group 1 — the label prefix, e.g. 'e|' or 'G|--'
#   group 2 — everything after the first '|'
import re as _re
_TAB_LINE_RE = _re.compile(r'^([eEAaDdGgBb]\|)(.*)')

# Characters that are part of a technique marker but NOT fret digits
_TECHNIQUE_CHARS = set('hpbr/\\~^<>()[]x ')

# A "token" on a tab line is one of:
#   FRET  — one or two digit number, e.g. '0', '7', '12'
#   TECH  — technique marker character: h, p, b, r, /, \, ~, ^
#   MUTE  — 'x'  (muted string)
#   BAR   — '|'  (bar line separator)
#   DASH  — '-'  (spacer, any run of dashes)
#   OTHER — anything else (should be rare)
_TOKEN_RE = _re.compile(r'(\d{1,2}|[hpbr/\\~^<>\[\]()\s]|x|\||-+|.)')


def transpose_fret(
    fret: int,
    string_index: int,
    from_tuning: Tuning,
    to_tuning: Tuning,
) -> Tuple[Optional[int], bool]:
    """
    Transposes a single fret number on a single string from one tuning
    to another, preserving the same sounding pitch.

    string_index: 0 = string 6 (lowest/thickest), 5 = string 1 (highest)

    Returns:
        (new_fret, playable)
        new_fret:  The fret number in the target tuning, or None if unplayable.
        playable:  False if the note cannot be reached on this string
                   in the target tuning (would require a negative fret).

    A positive per-string offset means the target string is higher-pitched,
    so we need a LOWER fret to reach the same note (new_fret = fret - offset).
    A negative offset means the target string is lower-pitched, requiring
    a higher fret.

    Examples (Standard → D Standard, all strings -2 semitones):
        fret 0 on string 6 → new_fret = 0 - (-2) = 2   (same E note, fret 2)
        fret 2 on string 6 → new_fret = 2 - (-2) = 4   (same F# note, fret 4)

    Examples (Standard → Open G, string 6 offset = -2):
        fret 0 on string 6 (E) → new_fret = 0 - (-2) = 2  (E at fret 2 in OpenG)
        fret 3 on string 6 (G) → new_fret = 3 - (-2) = 5  (G at fret 5 in OpenG)
    """
    offsets = string_offsets(from_tuning, to_tuning)
    offset  = offsets[string_index]
    new_fret = fret - offset
    playable = new_fret >= 0
    return (new_fret if playable else None, playable)


def _parse_tab_line(line: str) -> Tuple[Optional[int], Optional[str], str]:
    """
    Splits a tab string line into (string_index, label_prefix, body).

    string_index: 0–5 (string 6→1, low→high), or None if not a tab line.
    label_prefix: the matched prefix e.g. 'e|', or None.
    body:         everything after the first '|'.

    The string index is determined by scanning the label sequence
    [E, A, D, G, B, E] in order; 'e' (lowercase) maps to the high E (index 5).
    """
    m = _TAB_LINE_RE.match(line.strip())
    if not m:
        return (None, None, line)

    label = m.group(1)[0].upper()   # 'e' → 'E'
    body  = m.group(2)

    # Map label to string index.
    # 'E' is ambiguous (low E = index 0, high e = index 5).
    # We use lowercase 'e' for string 1 (high) and uppercase 'E' for string 6 (low).
    original_label = m.group(1)[0]
    if original_label == 'e':       # lowercase → high E, string 1
        string_index = 5
    elif label == 'A':
        string_index = 4
    elif label == 'D':
        string_index = 3
    elif label == 'G':
        string_index = 2
    elif label == 'B':
        string_index = 1
    else:                           # uppercase 'E' → low E, string 6
        string_index = 0

    return (string_index, m.group(1), body)


def _transpose_body(
    body: str,
    string_index: int,
    from_tuning: Tuning,
    to_tuning: Tuning,
) -> Tuple[str, List[str]]:
    """
    Transposes all fret numbers in a single tab body string.

    Returns:
        (transposed_body, warnings)
        transposed_body: body string with fret numbers replaced.
        warnings:        list of human-readable warnings for unplayable notes.

    Column alignment is preserved:
      - If a fret number grows in digit count (e.g. 9 → 12), the extra digit
        replaces the leading '-' before it if one exists; otherwise a '-' is
        appended to maintain minimum spacing.
      - If a fret number shrinks (e.g. 10 → 7), a '-' is inserted before it
        to fill the vacated column.
    """
    offsets  = string_offsets(from_tuning, to_tuning)
    offset   = offsets[string_index]
    tokens   = _TOKEN_RE.findall(body)
    warnings = []
    result   = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.isdigit() or (len(tok) == 2 and tok.isdigit()):
            # It's a fret number
            old_fret = int(tok)
            new_fret_val = old_fret - offset
            playable = new_fret_val >= 0

            if not playable:
                # Can't play this note on the same string in the target tuning.
                # Replace with '?' and record a warning.
                new_tok = '?' * len(tok)
                # Human-readable: string 6 = lowest (index 0), string 1 = highest (index 5)
                human_string = 6 - string_index
                warnings.append(
                    f"String {human_string}: fret {old_fret} → "
                    f"fret {new_fret_val} (unplayable in {to_tuning.name})"
                )
            else:
                new_tok = str(new_fret_val)

            old_width = len(tok)
            new_width = len(new_tok)

            if new_width > old_width:
                # Grew wider — try to eat a preceding '-' to compensate
                if result and result[-1].startswith('-'):
                    prev = result[-1]
                    if len(prev) > 1:
                        result[-1] = prev[1:]   # trim one dash
                    else:
                        result[-1] = ''          # consumed entirely
                # else: just let it grow (alignment shifts slightly)
            elif new_width < old_width:
                # Shrank — pad with a leading '-'
                new_tok = '-' * (old_width - new_width) + new_tok

            result.append(new_tok)

        elif _re.match(r'-+', tok):
            result.append(tok)
        else:
            # Technique char, bar line, mute, whitespace — pass through unchanged
            result.append(tok)

        i += 1

    return (''.join(result), warnings)


def transpose_tab_notation(
    tab_lines: List[str],
    from_tuning: Tuning,
    to_tuning: Tuning,
    capo: int = 0,
) -> List[str]:
    """
    Transposes a block of ASCII tab notation from one tuning to another,
    preserving the same sounding pitches.

    tab_lines: the raw lines of a 6-string ASCII tab block, e.g.:
        ['e|--0--2--3--|', 'B|--1--3--5--|', 'G|--0--2--4--|',
         'D|--2--4--5--|', 'A|--3--5--7--|', 'E|--x--x--x--|']

    Non-tab lines (annotations, blank lines, bar repeat markers) are passed
    through unchanged.

    capo: if the source tab uses a capo, supply the fret number so the
    effective pitch offset is computed correctly.

    Returns a list of lines with the same structure, frets transposed.

    Warnings about unplayable notes (frets that would go negative) are
    embedded as comment lines starting with '# ' immediately after the
    affected string line.
    """
    if from_tuning == to_tuning and capo == 0:
        return list(tab_lines)

    # If capo is in play, the effective from_tuning is capo-adjusted
    effective_from = capo_adjusted_tuning(from_tuning, capo) if capo else from_tuning

    output = []
    all_warnings = []

    for line in tab_lines:
        string_index, label, body = _parse_tab_line(line)

        if string_index is None or label is None:
            # Not a tab string line — pass through unchanged
            output.append(line)
            continue

        new_body, warnings = _transpose_body(body, string_index, effective_from, to_tuning)
        output.append(label + new_body)

        if warnings:
            all_warnings.extend(warnings)

    # Append a summary comment block if there were unplayable notes
    if all_warnings:
        output.append('')
        output.append('# Tab transposition warnings:')
        for w in all_warnings:
            output.append(f'#   {w}')

    return output
