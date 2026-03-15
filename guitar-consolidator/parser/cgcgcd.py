"""
parser/cgcgcd.py
────────────────
CGCGCD tuning voicing engine.

String numbering: 1=lowest(C), 2=G, 3=C, 4=G, 5=C, 6=D(highest)
Open strings:     C  G  C  G  C  D

Natural capo positions:
  Capo 0 → C, G, Am natural
  Capo 2 → D, A, Em, Bm natural
  Capo 4 → E, B, F#m natural
  Capo 5 → F, Dm natural

Voicing format: [s1,s2,s3,s4,s5,s6], 0=open, -1=mute, N=fret N from capo
Chromatic passing chords (dim, aug, b5) are omitted — hold previous chord.
"""

from typing import List, Dict, Tuple, Optional
from parser.tunings import CGCGCD, NOTE_TO_SEMITONE, SEMITONE_TO_NOTE_SHARP as ST_NOTE
import re as _re

_INTERVALS: Dict[str, List[int]] = {
    'maj':  [0, 4, 7],  'min':  [0, 3, 7],
    'dom7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10], 'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],  'dim':  [0, 3, 6],
    'aug':  [0, 4, 8],  'add9': [0, 2, 4, 7],
    '5':    [0, 7],
}

_QUALITY_MAP = {
    'm':'min','min':'min','minor':'min',
    'm7':'min7','min7':'min7',
    'maj7':'maj7','Δ':'maj7','M7':'maj7',
    '7':'dom7','sus2':'sus2','sus4':'sus4','sus':'sus4',
    'dim':'dim','dim7':'dim','o':'dim',
    'aug':'aug','+':'aug','5':'5',
}

# Chromatic passing qualities — omit from voicing output
_PASSING_QUALITIES = {'dim', 'aug'}


def _parse(chord: str) -> Tuple[str, str]:
    chord = _re.sub(r'/[A-G][#b]?$', '', chord)
    m = _re.match(r'^([A-G][#b]?)(.*)', chord)
    if not m:
        return (chord, 'maj')
    root, suf = m.group(1), m.group(2).strip()
    suf_l = suf.lower()
    if 'add9' in suf_l or 'add2' in suf_l: return (root, 'add9')
    if suf_l in _QUALITY_MAP:               return (root, _QUALITY_MAP[suf_l])
    if suf_l.startswith('m') and not suf_l.startswith('maj'): return (root, 'min')
    return (root, 'maj')


def _is_passing(chord: str) -> bool:
    """True if chord is a chromatic passing chord that should be omitted."""
    _, quality = _parse(chord)
    return quality in _PASSING_QUALITIES


# (root_semitone, quality, capo) → ([s1..s6], verified)
_V: Dict[Tuple[int,str,int], Tuple[List[int],bool]] = {

    # ── Verified ──────────────────────────────────────────────────
    (9,  'maj',  2): ([-1, 0, 2, 4, 2, 5], True),
    (2,  'maj',  2): ([-1,-1, 0, 0, 4, 5], True),
    (0,  'min',  0): ([ 0, 0, 3, 0, 0, 1], True),

    # ── C root capo 0 ─────────────────────────────────────────────
    (0,  'maj',  0): ([ 0, 0, 4, 0, 0, 0], False),
    (0,  'dom7', 0): ([ 0, 3, 4, 0, 0, 0], False),
    (0,  'maj7', 0): ([ 0, 4, 4, 0, 0, 0], False),
    (0,  'sus2', 0): ([ 0, 0, 0, 0, 0, 0], False),
    (0,  'add9', 0): ([ 0, 0, 0, 0, 0, 0], False),

    # ── G root capo 0 ─────────────────────────────────────────────
    (7,  'maj',  0): ([-1, 4, 0, 4, 0, 0], False),
    (7,  'min',  0): ([-1, 3, 0, 3, 0, 0], False),
    (7,  'dom7', 0): ([-1, 4, 3, 4, 0, 0], False),
    (7,  'maj7', 0): ([ 2, 4, 0, 4, 0, 0], False),
    (7,  'sus2', 0): ([-1, 0, 0, 0, 0, 0], False),

    # ── A root capo 0 ─────────────────────────────────────────────
    (9,  'min',  0): ([-1, 2, 4, 0, 0, 0], False),
    (9,  'dom7', 0): ([-1, 2, 2, 2, 0, 0], False),
    (9,  'min7', 0): ([-1, 2, 3, 2, 0, 0], False),

    # ── D root capo 0 ─────────────────────────────────────────────
    (2,  'maj',  0): ([-1,-1, 2, 2, 0, 4], False),
    (2,  'min',  0): ([-1,-1, 2, 2, 0, 3], False),
    (2,  'dom7', 0): ([-1,-1, 2, 0, 0, 4], False),
    (2,  'sus4', 0): ([-1,-1, 2, 2, 2, 0], False),

    # ── E root capo 0 ─────────────────────────────────────────────
    (4,  'maj',  0): ([ 4, 1, 4, 0, 0, 2], False),
    (4,  'min',  0): ([ 4, 0, 4, 0, 0, 2], False),
    (4,  'dom7', 0): ([ 4, 0, 4, 2, 0, 2], False),
    (4,  'min7', 0): ([ 4, 0, 3, 2, 0, 2], False),

    # ── F root capo 0 ─────────────────────────────────────────────
    (5,  'maj',  0): ([ 5, 2, 0, 0, 0, 3], False),
    (5,  'min',  0): ([ 5, 3, 0, 0, 0, 3], False),
    (5,  'dom7', 0): ([ 5, 2, 3, 0, 0, 3], False),

    # ── B root capo 0 ─────────────────────────────────────────────
    (11, 'maj',  0): ([-1, 4, 4, 4, 4, 0], False),
    (11, 'min',  0): ([-1, 4, 3, 4, 3, 0], False),
    (11, 'dom7', 0): ([-1, 4, 2, 4, 2, 0], False),
    (11, 'min7', 0): ([-1, 4, 2, 4, 3, 0], False),

    # ── Capo 2 ────────────────────────────────────────────────────
    (2,  'min',  2): ([-1,-1, 0, 0, 0, 1], False),
    (9,  'min',  2): ([-1, 0, 0, 3, 0, 0], False),
    (9,  'dom7', 2): ([-1, 0, 0, 2, 0, 0], False),
    (9,  'min7', 2): ([-1, 0, 0, 3, 0, 2], False),
    (11, 'min',  2): ([-1,-1, 0, 4, 3, 2], False),
    (11, 'dom7', 2): ([ 0, 0, 0, 2, 0, 2], False),
    (11, 'min7', 2): ([ 0, 0, 0, 2, 0, 2], False),
    (4,  'min',  2): ([ 0, 0, 0, 2, 0, 0], False),
    (4,  'dom7', 2): ([ 0, 0, 0, 2, 3, 0], False),
    (4,  'min7', 2): ([ 0, 0, 0, 2, 3, 2], False),
    (7,  'maj',  2): ([ 0, 0, 0, 0, 0, 3], False),
    (7,  'sus2', 2): ([ 0, 0, 0, 0, 0, 0], False),
    (6,  'min',  2): ([ 4, 0, 0, 4, 0, 0], False),
    (5,  'maj',  2): ([ 3, 0, 0, 3, 0, 0], False),
    (0,  'maj',  2): ([ 0, 3, 0, 3, 0, 2], False),

    # ── Capo 4 ────────────────────────────────────────────────────
    (4,  'maj',  4): ([ 0, 0, 0, 0, 0, 0], False),
    (4,  'min',  4): ([ 0, 0, 0, 0, 0, 3], False),
    (4,  'dom7', 4): ([ 0, 0, 0, 2, 0, 0], False),
    (11, 'maj',  4): ([ 0, 0, 4, 0, 4, 0], False),
    (11, 'min',  4): ([ 0, 0, 3, 0, 3, 0], False),
    (6,  'min',  4): ([ 0, 0, 2, 0, 2, 0], False),
    (9,  'maj',  4): ([ 0, 0, 0, 0, 0, 3], False),
    (2,  'maj',  4): ([ 0, 0, 0, 3, 0, 0], False),
    (7,  'maj',  4): ([ 3, 0, 0, 3, 0, 0], False),
}


def voice_chord(chord_name: str, capo: int) -> Optional[Dict]:
    """Returns best voicing for chord_name in CGCGCD at given capo.
    Returns None for passing chords (dim, aug) — hold previous chord."""
    root, quality = _parse(chord_name)
    if root not in NOTE_TO_SEMITONE:
        return None
    if quality in _PASSING_QUALITIES:
        return None

    root_st = NOTE_TO_SEMITONE[root]
    open_st = [(s + capo) % 12 for s in CGCGCD.semitones]

    fallbacks = [quality]
    if quality in ('dom7', 'maj7', 'add9', 'sus2', 'sus4'): fallbacks.append('maj')
    if quality in ('min7',): fallbacks.append('min')

    for q in fallbacks:
        key = (root_st, q, capo)
        if key in _V:
            frets, verified = _V[key]
            notes = [
                'x' if f == -1 else ST_NOTE[(open_st[s] + f) % 12]
                for s, f in enumerate(frets)
            ]
            fingered = [f for f in frets if f > 0]
            return {
                'frets':      frets,
                'notes':      notes,
                'score':      20 if verified else 10,
                'span':       (max(fingered)-min(fingered)) if len(fingered)>1 else 0,
                'open_count': sum(1 for f in frets if f == 0),
                'verified':   verified,
            }

    return None


def select_capo(chord_names: List[str], max_capo: int = 7) -> List[Dict]:
    """Returns top 2 capo positions for chord_names in CGCGCD."""
    unique = list(dict.fromkeys(
        _re.sub(r'/[A-G][#b]?$', '', c) for c in chord_names
        if not _is_passing(c)
    ))

    results = []
    for capo in range(0, max_capo + 1):
        voicings, total, unvoiced = {}, 0.0, []
        for chord in unique:
            v = voice_chord(chord, capo)
            if v and v['score'] > 0:
                voicings[chord] = v
                total += v['score'] * (3 if v['verified'] else 1)
            else:
                unvoiced.append(chord)

        penalty = capo * 0.4 + (2.0 if capo > 6 else 0.0) + len(unvoiced) * 3.0
        results.append({'capo': capo, 'score': total - penalty,
                        'voicings': voicings, 'unvoiced': unvoiced})

    results.sort(key=lambda r: r['score'], reverse=True)
    return results[:2]


def format_capo_suggestions(chord_names: List[str]) -> str:
    """Returns formatted CGCGCD voicing suggestions."""
    if not chord_names:
        return ""
    options = select_capo(chord_names)
    if not options:
        return ""

    lines = ["  [CGCGCD tuning]", ""]
    for i, opt in enumerate(options):
        capo  = opt['capo']
        label = f"Capo {capo}" if capo > 0 else "No capo"
        tag   = "(recommended)" if i == 0 else "(alternative)"
        lines.append(f"  ── {label} {tag}")
        lines.append("")
        for chord in sorted(opt['voicings']):
            v = opt['voicings'][chord]
            fret_str = ' '.join(str(f) if f >= 0 else 'x' for f in v['frets'])
            flag = "✓" if v['verified'] else "~"
            lines.append(f"  {flag} {chord:<10} {fret_str}")
        if opt['unvoiced']:
            lines.append(f"\n  (no voicing: {', '.join(opt['unvoiced'])})")
        lines.append("")
    lines.append("  ✓ verified  ~ unverified")
    return "\n".join(lines)
