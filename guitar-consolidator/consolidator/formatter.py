"""
consolidator/formatter.py
─────────────────────────
Renders a consolidated song dict to a clean, readable text string.

The consolidated dict now contains:
  - 'structure': the full ordered sequence of sections (with repeats)
  - 'sections':  a dict of templates keyed by section type

The formatter walks 'structure' to emit sections in song order,
using the pre-computed template for each section type each time
that section appears.
"""

LINE_WIDTH = 72


def _chord_line(chords):
    """Formats a chord list into a flat spaced chord line (no position data)."""
    if not chords:
        return ''
    return '  '.join(chords)


def _render_positioned(chords_positioned, lyric, indent='    ', marker=''):
    """
    Renders a chord+lyric pair in the traditional positional format where
    each chord sits directly above its onset point in the lyric.

    Parameters:
        chords_positioned  List of {'chord': str, 'col': int} dicts
        lyric              The lyric string (stripped, no leading indent)
        indent             Left margin applied to both output lines
        marker             Optional confidence marker character (e.g. '⚠', '~')

    Returns:
        (chord_line, lyric_line) as strings including indent,
        or None if no position data is available.

    COLLISION HANDLING
    ──────────────────
    If two chords are so close together that the first would overlap
    the second, we push the second rightward by the minimum needed.
    The lyric line is NOT modified — we only adjust the chord line.

    TRAILING CHORDS
    ───────────────
    Chords that fall past the lyric's end are appended after the lyric
    text in the chord line with a single space gap, and a matching
    number of spaces is appended to the lyric line to keep them flush.

    LYRIC INDENT COMPENSATION
    ─────────────────────────
    Tab source chord lines typically start at col 0 while the lyric has
    a 2-space leading indent (e.g. "  Old man..."). The col values from
    the parser reflect the chord line's coordinate system. We subtract
    the lyric's leading whitespace count before rendering so that chord
    positions land on the correct word, not 2 chars to its left.
    """
    if not chords_positioned:
        return None
    if not lyric and not chords_positioned:
        return None

    # Measure lyric leading whitespace (the indent the transcriber put before
    # the first word). Chord columns were measured from col 0 on the chord line
    # which typically has no leading whitespace — so we compensate.
    lyric_stripped = lyric.lstrip()
    lyric_indent   = len(lyric) - len(lyric_stripped)
    # Apply compensation: chord col 0 should align with lyric col `lyric_indent`
    # by subtracting lyric_indent from each chord's col.
    # Clamp to 0 so we never go negative.
    adjusted = [
        {'chord': cp['chord'], 'col': max(0, cp['col'] - lyric_indent)}
        for cp in chords_positioned
    ]

    # Build chord line character array
    # Start with a buffer as long as the lyric (we may extend it for trailing chords)
    buf_len    = len(lyric_stripped)
    chord_buf  = [' '] * max(buf_len, 1)

    cursor = 0  # tracks rightmost written position in chord_buf
    for cp in adjusted:
        chord = cp['chord']
        col   = cp['col']

        # Collision: if this chord would overlap the previous, push it right
        col = max(col, cursor)

        # Extend buffer if chord falls past current end
        needed = col + len(chord)
        if needed > len(chord_buf):
            chord_buf.extend([' '] * (needed - len(chord_buf)))

        for i, ch in enumerate(chord):
            chord_buf[col + i] = ch

        cursor = col + len(chord) + 1  # +1 minimum gap before next chord

    chord_str = ''.join(chord_buf).rstrip()

    # If the chord line extends past the lyric, pad the lyric with spaces
    lyric_out = lyric_stripped
    if len(chord_str) > len(lyric_out):
        lyric_out = lyric_out + ' ' * (len(chord_str) - len(lyric_out))

    # Apply indent and optional confidence marker to chord line
    if marker and marker.strip():
        chord_line_out = f"{indent[:-2]}{marker} {chord_str}"
    else:
        chord_line_out = f"{indent}{chord_str}"

    lyric_line_out = f"{indent}{lyric_out}"

    return chord_line_out, lyric_line_out


def _confidence_marker(confidence, disputed):
    """
    Returns a one-character indicator of chord confidence.
      ' ' = high confidence (≥ 0.85) — no distraction needed
      '~' = moderate (0.60–0.84)
      '⚠' = disputed (< 0.60)
    """
    if disputed:
        return '⚠'
    return ' '  # moderate confidence lines need no annotation


def format_consolidated(consolidated, show_confidence=True, show_alternatives=True):
    """
    Renders a consolidated song dict to a formatted text string.

    Walks 'structure' to preserve the full song order (Verse, Chorus,
    Verse, Chorus, Outro) rather than just showing each section once.
    """
    import re
    lines_out = []

    def out(text=''):
        lines_out.append(text)

    # ── Header ────────────────────────────────────────────────────────────
    out('═' * LINE_WIDTH)
    out(f"  {consolidated.get('title', 'Unknown Title').upper()}")

    artist = consolidated.get('artist', '')
    if artist:
        out(f"  {artist}")

    meta_parts = []
    key    = consolidated.get('key', '')
    capo   = consolidated.get('capo', 0)
    n_srcs = consolidated.get('source_count', 0)
    tuning = consolidated.get('tuning', 'Standard')

    if key:
        meta_parts.append(f"Key of {key}")
    if capo:
        meta_parts.append(f"Capo {capo}")
    if n_srcs:
        meta_parts.append(f"Consolidated from {n_srcs} source{'s' if n_srcs != 1 else ''}")
    if meta_parts:
        out(f"  {' · '.join(meta_parts)}")
    if tuning and tuning.lower() != 'standard':
        out(f"  Tuning: {tuning}")

    out('─' * LINE_WIDTH)

    # ── Author notes ──────────────────────────────────────────────────────
    # Strip bare [SectionHeader] lines that some tabs embed in their notes
    # block (e.g. '[Chords]', '[Chord list]') — raw-source navigation markers
    # that add nothing for the reader of a consolidated tab.
    # Also strip fret-voicing annotation lines (e.g. "G  3xx00x", "Am7 x0201x")
    # and "Chord forms" header prose — these are useful for chord diagrams but
    # not readable as human notes.
    _raw_notes = consolidated.get('notes', [])
    _voicing_line = re.compile(
        r'^(?:[A-G][b#]?\S*\s+)?[x0-9][x0-9][x0-9][x0-9]',  # fret voicing pattern
        re.IGNORECASE
    )
    _chord_forms_header = re.compile(r'^chord\s+forms?\b', re.IGNORECASE)
    notes = [
        n for n in _raw_notes
        if not re.match(r'^\s*\[[^\]]+\]\s*$', n)    # bare [Section] header
        and not _voicing_line.match(n)                 # fret voicing definition
        and not _chord_forms_header.match(n)           # "Chord forms..." header
    ]
    if notes:
        out()
        out('  [Notes from source]')
        for note in notes[:6]:
            out(f"  {note}")
        out()
        out('─' * LINE_WIDTH)

    # ── Chord diagrams ────────────────────────────────────────────────────
    # Pictorial ASCII fretboard grid for every chord in the song.
    # Three-tier voicing lookup: source annotations → curated DB → algorithmic.
    glossary = consolidated.get('chord_glossary', [])
    if False and glossary:
        from consolidator.chord_diagrams import (
            format_chord_section, parse_source_voicings
        )
        source_voicings = parse_source_voicings(consolidated.get('notes', []))
        out()
        out('  [Chord diagrams]')
        out()
        diagram_block = format_chord_section(glossary, source_voicings=source_voicings)
        for diagram_line in diagram_block.splitlines():
            out(diagram_line)
        out()
        out('─' * LINE_WIDTH)

    # ── Song body — walk 'structure' to preserve repeats ──────────────────
    # 'structure' is a list of (normalized_name, original_name) tuples
    # e.g. [('verse','verse_1'), ('chorus','chorus'), ('verse','verse_2'), ...]
    # We track the previous section name to avoid printing a redundant header
    # when two consecutive entries are the same type (rare, but possible).

    structure = consolidated.get('structure', [])
    section_templates = consolidated.get('sections', {})
    disputed_summary = []
    last_emitted = None

    for norm_name, orig_name in structure:
        # Try orig_name first (e.g. 'verse_1'), fall back to norm_name ('verse')
        template = section_templates.get(orig_name) or section_templates.get(norm_name, [])
        if not template:
            continue

        # Section label — use the display-friendly version of the original name
        display = orig_name.replace('_', ' ').title()

        # If this section type changed from the last one, print the header.
        # We always print it — even for the same type — so repeated verses
        # are clearly labelled each time they appear.
        out()
        out(f"  [{display}]")
        out()
        last_emitted = norm_name

        for song_line in template:
            chords            = song_line.get('chords', [])
            chords_positioned = song_line.get('chords_positioned', [])
            lyric             = song_line.get('lyric', '')
            confidence        = song_line.get('confidence', 1.0)
            disputed          = song_line.get('disputed', False)
            alts              = song_line.get('alternatives', [])

            marker = _confidence_marker(confidence, disputed) if show_confidence else ' '

            # Performance direction (e.g. "(x3, very short)")
            direction = song_line.get('direction', '')
            if direction:
                out(f"    {direction}")

            if chords:
                if lyric and chords_positioned:
                    # ── Positional rendering ─────────────────────────────────
                    # Chord sits directly above its onset syllable in the lyric.
                    rendered = _render_positioned(
                        chords_positioned, lyric,
                        indent='    ', marker=marker if show_confidence else ''
                    )
                    if rendered:
                        chord_line_out, lyric_line_out = rendered
                        out(chord_line_out)
                        out(lyric_line_out)
                        if disputed and show_alternatives and alts:
                            pct       = int(alts[0]['vote_share'] * 100)
                            alt_str   = '  '.join(alts[0]['chords'])
                            out(f"      ↳ alt: {alt_str}  ({pct}% of sources)")
                            disputed_summary.append((display, lyric[:45], alts))
                        out()
                        continue
                    # Fall through to flat rendering if positioned failed

                # ── Flat rendering (chord-only lines or no position data) ───
                chord_str = _chord_line(chords)
                prefix    = f"  {marker} " if show_confidence and marker.strip() else "    "
                out(f"{prefix}{chord_str}")
                if lyric:
                    out(f"    {lyric}")
                if disputed and show_alternatives and alts:
                    pct     = int(alts[0]['vote_share'] * 100)
                    alt_str = '  '.join(alts[0]['chords'])
                    out(f"      ↳ alt: {alt_str}  ({pct}% of sources)")
                    disputed_summary.append((display, lyric[:45], alts))

            elif lyric:
                out(f"    {lyric}")

            out()

    # ── Disputed lines summary ─────────────────────────────────────────────
    if disputed_summary and show_alternatives:
        out('─' * LINE_WIDTH)
        out()
        out('  ⚠  DISPUTED LINES — tabs disagreed on these chords')
        out()
        # Deduplicate by lyric (same line appears in every verse repeat)
        seen_lyrics = set()
        for sect, lyr, alts in disputed_summary:
            key = lyr.strip().lower()[:40]
            if key in seen_lyrics:
                continue
            seen_lyrics.add(key)
            out(f"  \"{lyr}...\"")
            for alt in alts[:3]:  # show up to 3 alternatives
                pct = int(alt['vote_share'] * 100)
                out(f"    alt: {' '.join(alt['chords'])}  ({pct}% of sources)")
            out()

    # ── CGCGCD tuning suggestions ─────────────────────────────────────────
    glossary = consolidated.get('chord_glossary', [])
    if glossary:
        from parser.cgcgcd import format_capo_suggestions
        out()
        out('─' * LINE_WIDTH)
        cgcgcd_block = format_capo_suggestions(glossary)
        for line in cgcgcd_block.splitlines():
            out(line)

    # ── Processing notes ──────────────────────────────────────────────────
    # Suppress implementation-detail warnings the reader doesn't need:
    #   'Found N raw ASCII tab lines — skipped'
    #   'Found N fret diagram(s) — skipped'
    #   'Removed N sign-off line(s)...'
    # Keep anything actionable, e.g. transposition notices.
    _suppress = ('Found ', 'Removed ')
    warnings = [w for w in consolidated.get('warnings', [])
                if not any(w.startswith(p) for p in _suppress)]
    if warnings:
        out('─' * LINE_WIDTH)
        out()
        out('  Processing notes:')
        for w in warnings:
            out(f"  • {w}")
        out()

    out('─' * LINE_WIDTH)
    out('  Guitar Tab Consolidator')

    # Collapse runs of 2+ blank lines to a single blank line
    collapsed = []
    prev_blank = False
    for line in lines_out:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = is_blank
    return '\n'.join(collapsed)


def save_text(consolidated, filepath, **kwargs):
    """Formats and writes the consolidated output to a .txt file."""
    text = format_consolidated(consolidated, **kwargs)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    return filepath


# ─────────────────────────────────────────────────────────────────────────
# COMPARE MODE  — side-by-side source vs consolidation report
# ─────────────────────────────────────────────────────────────────────────

def format_compare(consolidated: dict, ranked_tabs: list) -> str:
    """
    Produces a side-by-side comparison report showing what each source tab
    contributed to the consolidation vs what the final output chose.

    Layout per section:
        [Chorus]
        ────────────────────────────────────────────
        Line: "When Josie comes home ... so good"
          src1  (score 15.1 ★ 4.8):  C/F  F#7#9  B7#5#9  Em7   ← WINNER  ⚠ disputed
          src2  (score 13.8 ★ 4.6):  C7   C      D       D7

        Line: "She's the raw flame, the live wire"
          src1 + src2 agree:  Am7  D7  Gmaj7  Cmaj7  ✓

    Structure section:
        Summary at top — which sections each source had/lacked,
        which source was used as the structure template and why.

    Parameters
    ----------
    consolidated : dict   Output from consolidate()
    ranked_tabs  : list   The ranked tab list passed into consolidate()
    """
    from consolidator.engine import group_sections, align_blocks, vote_on_chords

    lines_out = []
    W = LINE_WIDTH

    def out(s=''):
        lines_out.append(s)

    # ── Header ────────────────────────────────────────────────────────────
    title  = consolidated.get('title', '')
    artist = consolidated.get('artist', '')
    key    = consolidated.get('key', '')

    out('═' * W)
    out(f"  {title.upper()}" + (f" — {artist}" if artist else ''))
    out(f"  Source comparison report  ·  Key of {key}")
    out('═' * W)
    out()

    # ── Source index ──────────────────────────────────────────────────────
    out('  Sources')
    out('  ───────')
    source_labels = []
    for i, tab in enumerate(ranked_tabs, 1):
        meta    = tab.get('metadata', {})
        source  = meta.get('source', 'unknown')
        rating  = meta.get('rating')
        votes   = meta.get('vote_count')
        score   = tab.get('score', 0)
        n_sects = len(tab.get('sections', []))
        label   = f"src{i}"
        source_labels.append({'label': label, 'score': score, 'tab': tab})
        rating_str = f"★{rating}" if rating is not None and rating != 3.0 else "—"
        votes_str  = str(votes)  if votes  is not None and votes  != 1   else "—"
        out(f"  {label}  score={score:.2f}  {rating_str}  ({votes_str} votes)  "
            f"{source}  {n_sects} sections")
    out()

    # ── Structure comparison ───────────────────────────────────────────────
    out('  Structure')
    out('  ─────────')
    # Collect section names per source
    for entry in source_labels:
        tab = entry['tab']
        sect_names = [s['name'] for s in tab.get('sections', [])]
        out(f"  {entry['label']}  [{', '.join(sect_names)}]")
    # Consolidated structure — show normalized names with repeat counts
    struct = consolidated.get('structure', [])
    struct_names = [norm for norm, orig in struct]
    # Build a compact display: if a section type appears multiple times,
    # show it as "verse x2" instead of "verse, verse"
    from collections import Counter
    type_counts = Counter(struct_names)
    seen_types  = []
    compact     = []
    for name in struct_names:
        if name not in seen_types:
            seen_types.append(name)
            n = type_counts[name]
            compact.append(f"{name} x{n}" if n > 1 else name)
    out(f"  out  [{', '.join(compact)}]  ({len(struct)} sections total)")
    out()

    # ── Per-section line-by-line comparison ───────────────────────────────
    # Re-run alignment to get raw per-source data
    groups = group_sections(ranked_tabs)

    # Determine which section names appear in consolidated structure
    seen_sections = set()
    for norm_name, orig_name in struct:
        if norm_name in seen_sections:
            continue
        seen_sections.add(norm_name)

        group_key = norm_name
        if group_key not in groups:
            # Section exists in structure but not in groups (e.g. came from
            # the template tab only) — show it as source-only
            out(f'  [{norm_name.upper()}]')
            out(f'  {"─" * (W - 2)}')
            out(f'    (only in one source — no cross-source comparison available)')
            out()
            continue

        aligned_rows = align_blocks(groups[group_key])
        if not aligned_rows:
            continue

        out(f'  [{norm_name.upper()}]')
        out(f'  {"─" * (W - 2)}')

        for row in aligned_rows:
            if not row:
                continue

            # Determine the winning chords for this row
            vote    = vote_on_chords(row)
            if not vote:
                continue
            winner_chords = vote['chords']
            is_disputed   = vote['disputed']
            confidence    = vote['confidence']
            lyric         = vote['lyric'].strip()

            # Group entries by their chord sequence (to detect agreement)
            chord_groups: dict = {}  # chord_tuple → list of (label, score)
            # Map score → label, disambiguating ties by position index.
            # Using score alone fails when two tabs have identical scores (e.g.
            # both sourced without metadata). Instead, build an ordered list
            # of (score, label) pairs and match by closest score, breaking ties
            # by assigning distinct labels in source_labels order.
            scored_labels = [(round(sl['score'], 4), sl['label'])
                             for sl in source_labels]

            def _label_for_score(score_val):
                s = round(score_val, 4)
                # Find all labels at this score
                matches = [lbl for sc, lbl in scored_labels if sc == s]
                if len(matches) == 1:
                    return matches[0]
                # Tie: return comma-separated distinct labels
                return '+'.join(dict.fromkeys(matches))  # preserves order, dedupes

            for entry in row:
                chords_key    = tuple(entry.get('chords', []))
                score_val     = entry.get('score', 0)
                matched_label = _label_for_score(score_val)
                chord_groups.setdefault(chords_key, []).append(
                    (matched_label, score_val, entry.get('orphan', False))
                )

            # Skip rows with no displayable content (empty beat markers etc.)
            winner_direction = vote.get('direction', '')
            if not winner_chords and not lyric and not winner_direction:
                continue

            # Line header
            if lyric:
                display_lyric = lyric[:60] + ('…' if len(lyric) > 60 else '')
                out(f'    Line: "{display_lyric}"')
            elif winner_chords:
                chord_preview = '  '.join(winner_chords[:6])
                out(f'    Chords: {chord_preview}')
            elif winner_direction:
                out(f'    Direction: {winner_direction}')
                out(f'      all sources agree  ✓')
                out()
                continue

            # Unanimous?
            if len(chord_groups) == 1:
                chords_str = '  '.join(winner_chords)
                if chords_str:
                    out(f'      all sources agree:  {chords_str}  ✓')
            else:
                winner_key = tuple(winner_chords)
                for chord_tuple, contributors in chord_groups.items():
                    chords_str  = '  '.join(chord_tuple) if chord_tuple else '(no chords)'
                    labels_str  = ', '.join(
                        f"{lbl}{'*' if orphan else ''}"
                        for lbl, _, orphan in contributors
                    )
                    is_winner   = chord_tuple == winner_key
                    winner_tag  = '  ← WINNER' if is_winner else ''
                    dispute_tag = '  ⚠ disputed' if (is_winner and is_disputed) else ''
                    conf_tag    = f'  ({confidence:.0%})' if is_winner and is_disputed else ''
                    out(f'      {labels_str:12s}  {chords_str}{winner_tag}{conf_tag}{dispute_tag}')

            out()

    # ── Summary ────────────────────────────────────────────────────────────
    out('═' * W)
    out()

    # Count agreements and disputes across all sections
    total_rows  = 0
    agreed_rows = 0
    disputed_rows = 0

    for norm_name, _ in struct:
        group_key = norm_name
        if group_key not in groups:
            continue
        aligned_rows = align_blocks(groups[group_key])
        for row in aligned_rows:
            if not row:
                continue
            vote = vote_on_chords(row)
            if not vote:
                continue
            total_rows += 1
            chord_seqs = set(tuple(e.get('chords', [])) for e in row)
            if len(chord_seqs) == 1:
                agreed_rows += 1
            elif vote['disputed']:
                disputed_rows += 1

    if total_rows:
        agreement_pct = agreed_rows / total_rows * 100
        out(f'  Rows total: {total_rows}')
        out(f'  Unanimous:  {agreed_rows} ({agreement_pct:.0f}%)')
        if disputed_rows:
            out(f'  Disputed:   {disputed_rows}  '
                f'(flagged where no source had ≥60% weighted vote share)')
        out()

    out('  Generated by Guitar Tab Consolidator — compare mode')
    out('═' * W)

    return '\n'.join(lines_out)


def save_compare(consolidated: dict, ranked_tabs: list, filepath: str) -> str:
    """Formats and writes the compare report to a .txt file."""
    text = format_compare(consolidated, ranked_tabs)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    return filepath
