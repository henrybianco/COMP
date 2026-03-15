"""
renderer.py
───────────
Renders a single parsed tab dict to plain text with CGCGCD voicings.
"""

from parser.cgcgcd import format_capo_suggestions

LINE_WIDTH = 72
DIVIDER    = '─' * LINE_WIDTH
HEAVY      = '═' * LINE_WIDTH


def render(tab: dict, song: str, artist: str) -> str:
    lines = []

    def out(s=""):
        lines.append(s)

    # ── Header ───────────────────────────────────────────────────
    out(HEAVY)
    out(f"  {song.upper()}")
    out(f"  {artist}")
    out(DIVIDER)

    sections  = tab.get("sections", [])
    all_chords = []

    # Determine if all sections are unlabeled
    all_unlabeled = all(
        s.get("name", "").startswith("unlabeled")
        for s in sections if s.get("blocks")
    )

    for section in sections:
        name   = section.get("name", "")
        blocks = section.get("blocks", [])
        if not blocks:
            continue

        # Only show section header if it has a meaningful name
        if not all_unlabeled and not name.startswith("unlabeled"):
            display = name.replace("_", " ").title()
            out()
            out(f"  [{display}]")

        out()

        for block in blocks:
            chords = block.get("chords", [])
            lyric  = block.get("lyric", "").rstrip()
            all_chords.extend(chords)

            if not chords and not lyric:
                continue

            if chords and lyric:
                out(f"    {_place_chords(chords, lyric)}")
                out(f"    {lyric}")
            elif chords:
                out(f"    {' '.join(chords)}")
            else:
                out(f"    {lyric}")

    out()
    out(DIVIDER)

    # ── CGCGCD voicings ──────────────────────────────────────────
    unique_chords = list(dict.fromkeys(all_chords))
    if unique_chords:
        cgcgcd = format_capo_suggestions(unique_chords)
        if cgcgcd:
            for line in cgcgcd.splitlines():
                out(line)
            out(DIVIDER)

    return "\n".join(lines)


def _place_chords(chords: list, lyric: str) -> str:
    """Space chords evenly across the lyric line width."""
    if not chords:
        return ""
    if len(chords) == 1:
        return chords[0]

    lyric_len = max(len(lyric), len(chords) * 6)
    step      = lyric_len // len(chords)
    result    = list(" " * lyric_len)

    for i, chord in enumerate(chords):
        pos = i * step
        for j, ch in enumerate(chord):
            if pos + j < len(result):
                result[pos + j] = ch

    return "".join(result).rstrip()
