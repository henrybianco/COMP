import re

# ─────────────────────────────────────────────
# CHORD DETECTION
# ─────────────────────────────────────────────
# This regex matches the vast majority of chord names you'll find in tabs.
# Breaking it down:
#   [A-G]         → root note (A through G)
#   [#b]?         → optional sharp (#) or flat (b)
#   (?:maj|min|m|M|sus|aug|dim|add)?  → optional quality
#   [0-9]?        → optional number (7, 9, 11, etc.)
#   (?:/[A-G][#b]?)? → optional slash chord bass note (e.g. G/B)

CHORD_PATTERN = re.compile(
    r'(?<![A-Za-z])'           # not preceded by a letter (avoids 'Em' in 'theme')
    r'([A-G][#b]?'             # root: note + optional sharp/flat
    r'(?:'
        r'maj|min|m(?!aj)|M'   # quality (m only if not followed by 'aj')
        r'|sus|s(?=\d)'        # sus or shorthand s (only before digit, e.g. s4)
        r'|aug|dim'
    r')?'
    r'(?:[0-9]{1,2})?'         # base number (7, 9, 11, 13)
    r'(?:sus[0-9]?|s[0-9])?'   # sus/s AFTER number (e.g. 7sus4, 7s4)
    r'(?:add[0-9]{1,2})?'      # optional add before tensions
    r'(?:[#b][0-9]{1,2})*'     # zero or more stacked tensions: #5, b9, #9, #5#9
    r'(?:add[0-9]{1,2})?'      # optional add after tensions (e.g. maj7add13)
    r'(?:/[A-G][#b]?)?)'       # optional slash bass note
    r'(?![#b\w])'              # not followed by #, b, or word char
)

# ─────────────────────────────────────────────
# PASSING CHORD / BASS MOVEMENT DETECTION
# ─────────────────────────────────────────────
# Some transcribers notate descending bass lines as bare slash tokens:
#   F  /F  /E  Em   means "F chord, walk bass down F→E, then Em"
#   The /F and /E are NOT chords — they're the same chord held while
#   the bass note moves. We need to strip them before chord extraction
#   so they don't get misread as standalone note names like "E".
#
# Pattern: a forward slash followed directly by a note name (no chord quality).
# This is distinct from a real slash chord like Dm/C which has a full chord
# name before the slash.

def normalize_chord(chord):
    """
    Normalizes non-standard chord shorthand to canonical form.
      D7s4  -> D7sus4,  A7s4 -> A7sus4,  Dsus4 -> Dsus4 (unchanged)
      FM7   -> Fmaj7,   CM7  -> Cmaj7  (capital-M major shorthand)
    """
    import re as _re
    # 7s4 -> 7sus4
    chord = _re.sub(r'([0-9])s([0-9])', lambda m: m.group(1) + 'sus' + m.group(2), chord)
    # s4 at start (no preceding digit) -> sus4
    chord = _re.sub(r'(?<![A-Za-z0-9])s([0-9])', lambda m: 'sus' + m.group(1), chord)
    # FM7 -> Fmaj7, CM9 -> Cmaj9, etc. (capital-M major shorthand)
    chord = _re.sub(r'^([A-G][#b]?)M([0-9])', lambda m: m.group(1) + 'maj' + m.group(2), chord)
    return chord

def extract_chords_with_positions(chord_line, lyric_line=''):
    """
    Extracts chords from a chord line with their column positions,
    adjusted relative to the lyric line's indentation.

    Returns a list of {'chord': str, 'col': int} dicts.

    The col value is the character offset into the lyric where the
    chord change occurs. Negative values are clamped to 0 (chord
    falls before the lyric starts — place at the beginning).

    Example:
        chord_line = 'D         F                C'
        lyric_line = '  Old man look at my life, ...'
        -> [{'chord': 'D', 'col': 0},   # D over 'Old'
            {'chord': 'F', 'col': 8},   # F over 'look'  (10-2 indent)
            {'chord': 'C', 'col': 25}]  # C over 'life'
    """
    # How far is the lyric indented relative to chord line col 0?
    lyric_indent = len(lyric_line) - len(lyric_line.lstrip())

    result = []
    for m in CHORD_PATTERN.finditer(chord_line):
        col = max(0, m.start() - lyric_indent)
        result.append({'chord': m.group(), 'col': col})
    return result


PASSING_NOTE_PATTERN = re.compile(
    r'(?<!\w)/([A-G][#b]?)(?![a-zA-Z0-9])'
)


def strip_passing_notes(line):
    """
    Removes bare /X passing-note tokens from a chord line, returning
    a cleaned line and a list of the passing notes that were found.

    Example:
        'F  /F  /E  Em  A7'  →  cleaned: 'F  Em  A7',  passing: ['F', 'E']

    The passing notes are preserved separately so we can store them
    in the block for reference — they carry musical information even
    if they aren't standalone chords.
    """
    passing = PASSING_NOTE_PATTERN.findall(line)
    cleaned = PASSING_NOTE_PATTERN.sub('', line)
    return cleaned, passing

# ─────────────────────────────────────────────
# SECTION DETECTION
# ─────────────────────────────────────────────

SECTION_PATTERN = re.compile(
    r'^\[?(?P<label>verse|chorus|bridge|intro|outro|pre-chorus|prechorus|'
    r'interlude|solo|hook|refrain|tag|coda|instrumental|jam|breakdown|'
    r'transition|vamp|outro\s*solo|link|pre-verse|turnaround|fill)'
    r'[\ \s\]]*(?P<number>\d+)?\]?',
    re.IGNORECASE
)

# Section labels that look like headers but are actually chord glossaries or
# metadata blocks — treat their content as notes[], not song sections.
# Strict glossary header: a full-line label like [Chords], [Voicings], etc.
# Entering this mode suspends section building until a real [Section] header.
GLOSSARY_SECTION_PATTERN = re.compile(
    r'^\[?(?:chords?|chord\s*list|chord\s*glossary|voicings?|fingerings?'
    r'|tuning|capo|key|notes?|about|intro\s*notes?'
    r'|chords?\s+used.*'           # "Chords used in this tab:"
    r'|chords?\s+in\s+this.*'      # "Chords in this song:"
    r')[\s\]:]*$',
    re.IGNORECASE
)

# Inline glossary header: prose sentence introducing voicing annotations.
# Appears before song content in tabs with no section headers.
# Goes to notes; absorbs following voicing lines until next paragraph break.
# Does NOT suppress section creation.
GLOSSARY_INLINE_PATTERN = re.compile(
    r'^chord\s+forms?\b',   # "Chord forms which are actually fingerpicked:"
    re.IGNORECASE
)

# ─────────────────────────────────────────────
# TAB LINE DETECTION
# ─────────────────────────────────────────────

TAB_LINE_PATTERN = re.compile(r'^[eEBGDAd]?\|[-\d|hpbr~/\\sxo\s]+\|?$')

# Strumming / rhythm notation lines: "v   v   v   v" or "^ v ^ v"
# These appear above or between ASCII tab blocks as beat markers.
STRUM_NOTATION_PATTERN = re.compile(r'^\s*[v\^\s|]+$')

# Triplet/annotation lines below tab blocks: "~~~3~~~ ~~~3~~~"
TAB_ANNOTATION_PATTERN = re.compile(r'^[\s~\^\(\)0-9a-zA-Z]+$')

# ─────────────────────────────────────────────
# FRET DIAGRAM DETECTION
# ─────────────────────────────────────────────
# Chord voicing diagrams like xx321x or x02210 — exactly 6 chars, digits or x

FRET_DIAGRAM_PATTERN = re.compile(r'^[xX0-9]{6}$')




# ─────────────────────────────────────────────
# PERFORMANCE DIRECTION DETECTION
# ─────────────────────────────────────────────
# Lines like "(x3, very short)", "(play loud)", "(softly)" are performance
# annotations, not lyrics. They appear fully enclosed in parentheses or
# brackets. We detect and store them separately so they don't pollute lyrics.

DIRECTION_PATTERN = re.compile(
    r'^\s*[\(\[]\s*(?:x\d+|repeat|play\s+\w+|softly|loudly|slow|fast|'
    r'very\s+\w+|whisper|accel|rit|ad\s*lib|tacet|[0-9]+x|times|'
    r'till\s+end|fade|optional|'
    r'guitar\s+solo|bass\s+solo|drum\s+solo|solo|'
    r'[0-9]+:[0-9]+\s*[-–]\s*[0-9]+:[0-9]+|'   # timestamps: (2:15 - 2:31)
    r'[0-9]+:[0-9]+)\b.*[\)\]]\s*$',
    re.IGNORECASE
)

# Separate pattern for instruction lines that aren't parenthesised:
# "Play x3 (guitar is doubled by bass):" — imperative + optional parenthetical
INSTRUCTION_PATTERN = re.compile(
    r'^\s*(?:play\s+x?\d+|repeat\s+x?\d+|play\s+\d+\s+times?'
    r'|w/\s*\w+|with\s+\w+|guitar\s+solo|bass\s+solo|drum\s+fill'
    r'|n\.c\.|n\.c|tacet)\b.*[:\.]?\s*$',
    re.IGNORECASE
)


def is_performance_direction(line):
    """
    Returns True if the line is a performance annotation, not a lyric.
    Examples:
        (x3, very short)        — parenthesised direction
        (2:15 - 2:31)           — timestamp marker
        Play x3 (guitar solo)   — bare imperative instruction
        N.C.                    — no chord notation
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Parenthesised / bracketed directions
    starts = stripped[0] in ('(', '[')
    ends   = stripped[-1] in (')', ']')
    if starts and ends and DIRECTION_PATTERN.match(stripped):
        return True
    # Bare instruction lines (not parenthesised)
    if INSTRUCTION_PATTERN.match(stripped):
        return True
    return False

# ─────────────────────────────────────────────
# SIGN-OFF / FOOTER DETECTION
# ─────────────────────────────────────────────
# Some transcribers end their tabs with social sign-offs, usernames,
# or closing remarks like "Cheers", "Nowhere Man", "Any questions email me",
# "Hope this helps!", etc. These are not lyrics or chords.
#
# We detect them as lines that:
#   1. Have no chords
#   2. Are very short (1-3 words)
#   3. Match common sign-off patterns OR appear only at the end of the tab
#
# This runs as a post-processing step after section parsing.

SIGN_OFF_PATTERNS = re.compile(
    r'^(cheers|thanks|enjoy|hope\s+this|any\s+questions|corrections|'
    r'email|feel\s+free|please\s+rate|rate\s+this|comment|feedback|'
    r'tabbed\s+by|transcribed|arranged|written\s+by|'
    r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})\b',
    re.IGNORECASE
)

# Common username-style patterns: "Nowhere Man", "GuitarHero99", single words
USERNAME_PATTERN = re.compile(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)?$|^[A-Za-z0-9_]+[0-9]+$')


# Tab legend / key block patterns
# e.g. "************************************", "| h  Hammer-on", "| p  Pull-off"
LEGEND_LINE_PATTERN = re.compile(r'^[*\-=_]{4,}$|^[|]\s*[a-zA-Z]\s+\w')

# Directions that reference the ASCII tab notation itself — meaningless in a chord chart
TAB_REF_DIRECTION = re.compile(
    r'repeat\s+above|see\s+tab|play\s+tab|as\s+tab|refer\s+to\s+tab',
    re.IGNORECASE
)


def is_sign_off(line, has_chords=False):
    """
    Returns True if a line looks like a social sign-off, username footer,
    or legend/key block (e.g. the h=hammer-on / p=pull-off notation key).
    Only flags lines that have no chords.
    """
    if has_chords:
        return False
    stripped = line.strip()
    if not stripped:
        return False
    # Legend/key lines: rows of stars/dashes or "| x  description" notation
    if LEGEND_LINE_PATTERN.match(stripped):
        return True
    # Very short lines with no chords are suspicious
    words = stripped.split()
    if len(words) > 5:
        return False
    if SIGN_OFF_PATTERNS.match(stripped):
        return True
    # Single capitalised words or simple "FirstName LastName"
    if len(words) <= 2 and USERNAME_PATTERN.match(stripped):
        return True
    return False



def merge_continuation_blocks(parsed_tab):
    """
    Post-processing step: merges lyric lines that were split across two
    text lines in the tab source.

    Example — tab source:
        Dm9
          Old man look at my life, I'm a lot like
        D        F  C  G
        you were.

    Parser produces two blocks:
        {chords: [Dm9],         lyric: "Old man look at my life, I'm a lot like"}
        {chords: [D, F, C, G],  lyric: "you were."}

    We merge these when:
      1. The current block's lyric is short (<= 5 words)
      2. The previous block's lyric does not end in sentence-ending punctuation
      3. The previous block has a lyric (not chord-only)

    Merged result:
        {chords: [Dm9, D, F, C, G], lyric: "Old man look at my life, I'm a lot like you were."}
    """
    for section in parsed_tab.get('sections', []):
        blocks = section['blocks']
        merged = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            lyric = block.get('lyric', '').strip()
            word_count = len(lyric.split()) if lyric else 0

            # Check if this block looks like a continuation of the previous one
            # Detect if lyric starts with an inline section label that got absorbed
            # e.g. "[Instrumental] Fmaj7" — this should never merge with the prior line
            import re as _re
            _has_label = bool(_re.match(r'^\[(?:verse|chorus|bridge|intro|outro|instrumental|solo|interlude)', lyric, _re.IGNORECASE))

            if (merged
                    and word_count >= 1
                    and word_count <= 4
                    and lyric
                    and not _has_label                   # don't merge if lyric is actually a section label
                    and not block.get('chords')          # don't merge if block has own chords
                    and not merged[-1].get('direction')
                    and merged[-1].get('lyric', '').strip()
                    and not merged[-1]['lyric'].strip()[-1] in '.!?'):

                # Merge: combine chords (deduplicated) and join lyrics.
                # Offset the continuation block's chord positions by the
                # length of the already-joined lyric so positions stay
                # accurate in the merged block.
                prev = merged[-1]
                prev_lyric  = prev['lyric'].rstrip()
                join_offset = len(prev_lyric) + 1  # +1 for the space we add

                combined_chords = prev.get('chords', []) + [
                    c for c in block.get('chords', [])
                    if c not in prev.get('chords', [])
                ]
                # Offset continuation positions; keep prev positions as-is
                prev_pos  = prev.get('chords_positioned', [])
                block_pos = [
                    {'chord': cp['chord'], 'col': cp['col'] + join_offset}
                    for cp in block.get('chords_positioned', [])
                    if cp['chord'] not in prev.get('chords', [])
                ]
                merged[-1] = {
                    'chords':            combined_chords,
                    'chords_positioned': prev_pos + block_pos,
                    'passing_notes':     prev.get('passing_notes', []) + block.get('passing_notes', []),
                    'lyric':             prev_lyric + ' ' + lyric,
                    'direction':         prev.get('direction', ''),
                }
            else:
                merged.append(block)
            i += 1

        section['blocks'] = merged

    return parsed_tab

def clean_sign_offs(parsed_tab):
    """
    Post-processing step: removes sign-off lines from the last section
    of a parsed tab. We only clean the final section because sign-offs
    always appear at the end of a tab, never mid-song.

    Modifies the parsed_tab dict in place and returns it.
    """
    sections = parsed_tab.get('sections', [])
    if not sections:
        return parsed_tab

    last_section = sections[-1]
    cleaned_blocks = []
    removed = []

    for block in last_section.get('blocks', []):
        lyric      = block.get('lyric', '').strip()
        has_chords = bool(block.get('chords'))
        if lyric and is_sign_off(lyric, has_chords):
            removed.append(lyric)
        else:
            cleaned_blocks.append(block)

    if removed:
        last_section['blocks'] = cleaned_blocks
        parsed_tab['warnings'].append(
            f"Removed {len(removed)} sign-off line(s) from final section: "
            + ", ".join(f'"{r}"' for r in removed)
        )

    # Remove the section entirely if it's now empty
    parsed_tab['sections'] = [s for s in sections if s['blocks']]
    return parsed_tab

# ─────────────────────────────────────────────
# INLINE TUNING DETECTION
# ─────────────────────────────────────────────
# Catches tunings written as note sequences like D-G-C-F-A-d
# Six notes (A-G, case insensitive, optional sharp/flat) separated by hyphens

INLINE_TUNING_PATTERN = re.compile(
    r'\b([A-Ga-g][#b]?-[A-Ga-g][#b]?-[A-Ga-g][#b]?-[A-Ga-g][#b]?-[A-Ga-g][#b]?-[A-Ga-g][#b]?)\b'
)


def is_tab_line(line):
    """
    Returns True if the line is part of an ASCII tab block:
      - Standard string lines:   e|--0--2--|
      - Strumming/beat markers:  v   v   v   v
      - Triplet/annotation:      ~~~3~~~ ~~~3~~~   (1/4-note triplets)
      - Caret annotation:        ^^^ 2nd time only
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Standard tab line: e|---0--- or B|---1---
    if TAB_LINE_PATTERN.match(stripped):
        return True
    # Strum/beat markers: entirely composed of v, ^, space, pipe
    # Require at least 2 v or ^ characters to avoid false-positives on short words
    if (re.match(r'^[v\^\s|]+$', stripped)
            and stripped.count('v') + stripped.count('^') >= 2):
        return True
    # Tab annotation lines: tilde runs, triplet markers, caret runs
    # e.g. "~~~3~~~ ~~~3~~~  (1/4-note triplets)"  or  "^^^ 2nd time only"
    if re.search(r'~~~|\^\^\^|1/[0-9]-note|triplet', stripped):
        return True
    return False


def is_fret_diagram(line):
    """Returns True if the line is a fret voicing diagram like xx321x."""
    return bool(FRET_DIAGRAM_PATTERN.match(line.strip()))


def is_chord_line(line):
    """
    Returns True if the line looks like a chord line rather than lyrics.
    Finds all chord-pattern matches, then checks whether they account
    for most of the non-whitespace characters on the line.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if is_tab_line(stripped) or is_fret_diagram(stripped):
        return False

    matches = CHORD_PATTERN.findall(stripped)
    if not matches:
        return False

    chord_chars = sum(len(m) for m in matches)
    total_chars = len(stripped.replace(' ', ''))

    return (chord_chars / total_chars) > 0.6


def detect_section(line):
    """
    If the line is a section header, return a normalized label string.
    '[Chorus]' → 'chorus',  '** Verse 2 **' → 'verse_2'
    Returns None if not a section header.
    """
    cleaned = line.strip().strip('*').strip('-').strip('[').strip(']').strip(':').strip()
    match = SECTION_PATTERN.match(cleaned)
    if match:
        label = match.group('label').lower()
        number = match.group('number')
        return f"{label}_{number}" if number else label
    return None


# ─────────────────────────────────────────────
# MAIN PARSE FUNCTION
# ─────────────────────────────────────────────


def _split_unlabeled_sections(result: dict) -> dict:
    """
    When a tab has no section headers, everything lands in one 'unlabeled'
    section. This post-processing step splits it into numbered parts
    (unlabeled_1, unlabeled_2, ...) at natural paragraph boundaries —
    runs of empty blocks that separate distinct musical passages.

    Only applied when:
      - The single unlabeled section has ≥ 8 blocks (worth splitting), AND
      - There is no other named section in the tab (i.e. the whole song
        is unlabeled — we don't fragment songs that are mostly labelled).
    """
    sections = result.get('sections', [])

    # Only act when ALL sections are unlabeled
    named = [s for s in sections if not s['name'].startswith('unlabeled')]
    if named:
        return result

    unlabeled = [s for s in sections if s['name'].startswith('unlabeled')]
    if not unlabeled:
        return result

    # Merge all unlabeled blocks into one flat list
    all_blocks = []
    for s in unlabeled:
        all_blocks.extend(s['blocks'])

    if len(all_blocks) < 4:
        return result  # too short to be worth splitting

    # Split at paragraph-break sentinels (explicit) or empty blocks (implicit)
    groups = []
    current_group = []
    for block in all_blocks:
        # Paragraph-break sentinels are explicit split markers
        if block.get('paragraph_break'):
            if current_group:
                groups.append(current_group)
                current_group = []
            continue
        has_content = bool(block.get('chords')) or bool(block.get('lyric', '').strip()) or bool(block.get('direction', '').strip())
        if has_content:
            current_group.append(block)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []
    if current_group:
        groups.append(current_group)

    if len(groups) <= 1:
        return result  # no natural breaks found

    # Replace the unlabeled section(s) with numbered parts
    new_sections = []
    for i, group in enumerate(groups, 1):
        new_sections.append({'name': f'unlabeled_{i}', 'blocks': group})

    result['sections'] = new_sections
    return result

def parse_tab(raw_text, metadata=None):
    """
    Parses a raw tab string into a structured Python dictionary.

    {
        'metadata': { 'source': '...', 'rating': 4.8, ... },
        'tuning': 'Standard',
        'capo': 0,
        'notes': ['Author commentary preserved here...'],
        'sections': [
            {
                'name': 'verse_1',
                'blocks': [
                    { 'chords': ['F', 'Em7', 'A7'], 'lyric': 'Yesterday...' },
                    ...
                ]
            },
            ...
        ],
        'warnings': [...]
    }
    """

    lines = raw_text.splitlines()

    result = {
        'metadata': metadata or {},
        'tuning': 'Standard',
        'capo': 0,
        'notes': [],      # prose from the header block (author commentary, key info)
        'sections': [],
        'warnings': []
    }

    # ── FIX 1: Separate header block from tab body ──────────────────────────
    # Everything before the first section label is "header" content.
    # We extract structured fields from it (tuning, capo) and save the
    # rest as notes rather than letting it bleed into lyric blocks.

    first_section_idx = 0
    for i, line in enumerate(lines):
        if detect_section(line):
            first_section_idx = i
            break

    header_lines = lines[:first_section_idx]
    body_lines   = lines[first_section_idx:]

    for line in header_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Standard "Tuning: Drop D" label
        tuning_match = re.search(r'tuning[:\s]+(.+)', stripped, re.IGNORECASE)
        if tuning_match:
            result['tuning'] = tuning_match.group(1).strip()

        # FIX 3: Inline six-note tuning like D-G-C-F-A-d
        inline_tuning = INLINE_TUNING_PATTERN.search(stripped)
        if inline_tuning and result['tuning'] == 'Standard':
            result['tuning'] = inline_tuning.group(1)

        # Capo
        capo_match = re.search(r'capo[:\s]+(\d+)', stripped, re.IGNORECASE)
        if capo_match:
            result['capo'] = int(capo_match.group(1))

        # Source key declared in header (e.g. "Key: G", "key of G", "played in G")
        # This overrides any key_hint injected from the CLI target key.
        import re as _re
        key_decl = _re.search(
            r'\bkey\s+of\s+([A-G][b#]?)\b'
            r'|\bplayed\s+in\s+(?:the\s+key\s+of\s+)?([A-G][b#]?)\b'
            r'|\bkey[:\s]+([A-G][b#]?)\b',
            stripped, _re.IGNORECASE)
        if key_decl:
            declared = next(g for g in key_decl.groups() if g)
            result['metadata']['key'] = declared

        # Save the line as a human-readable note regardless
        result['notes'].append(stripped)

    # ── Walk through the body and build sections ────────────────────────────
    current_section  = None
    pending_chord_line = None
    tab_line_count   = 0
    fret_diagram_count = 0
    in_glossary        = False
    in_inline_glossary = False  # "Chord forms..." prose — absorbs voicing lines until para break
    glossary_blank_count = 0    # track blanks inside hard glossary to detect end of block
    consecutive_blanks = 0      # track blanks outside glossary to emit paragraph-break sentinels

    for line in body_lines:

        # In glossary mode: sweep lines into notes[] and skip processing.
        # Exit conditions:
        #   (a) A real [Section] header always exits glossary mode.
        #   (b) A double blank line signals end of the chord-list block.
        #   (c) A chord line with no voicing notation (no fret patterns) signals
        #       the song body has started — exit and re-process this line.
        if in_glossary:
            if not line.strip():
                glossary_blank_count += 1
                if glossary_blank_count >= 2:
                    in_glossary = False
                    glossary_blank_count = 0
                continue
            else:
                glossary_blank_count = 0
            # (a) Real section header
            section_label = detect_section(line)
            if section_label:
                in_glossary = False
                current_section = {'name': section_label, 'blocks': []}
                result['sections'].append(current_section)
                pending_chord_line = None
                consecutive_blanks = 0
                continue
            # (c) Chord line with no colon or fret-voicing pattern → song body started
            stripped = line.strip()
            has_voicing = bool(re.search(r'[x0-9][x0-9][x0-9][x0-9]', stripped))
            has_colon   = ':' in stripped
            if is_chord_line(stripped) and not has_voicing and not has_colon:
                in_glossary = False
                # Fall through — let this line be processed normally below
            else:
                result['notes'].append(stripped)
                continue
        # Skip blank lines; flush any orphaned chord line
        if not line.strip():
            # A blank line ends inline glossary mode (voicing list is over)
            in_inline_glossary = False
            if pending_chord_line and current_section:
                cleaned_line, passing = strip_passing_notes(pending_chord_line)
                chord_matches = [
                    {'chord': normalize_chord(m.group()), 'col': m.start()}
                    for m in CHORD_PATTERN.finditer(cleaned_line)
                ]
                current_section['blocks'].append({
                    'chords':            [cm['chord'] for cm in chord_matches],
                    'chords_positioned': chord_matches,
                    'passing_notes':     passing,
                    'lyric':             ''
                })
            pending_chord_line = None
            consecutive_blanks += 1
            # Emit a paragraph-break sentinel on the 2nd consecutive blank line.
            # The section splitter uses these to divide unlabeled content into
            # logical sections (verse 1, bridge, chorus, etc.)
            if consecutive_blanks == 2 and current_section is not None:
                current_section['blocks'].append({'paragraph_break': True, 'chords': [], 'lyric': ''})
            continue

        # Non-blank line — reset consecutive blank counter
        consecutive_blanks = 0

        # FIX 2: Skip fret diagrams (xx321x) — don't treat as lyrics
        if is_fret_diagram(line):
            fret_diagram_count += 1
            pending_chord_line = None
            continue

        # Skip raw ASCII tab lines
        if is_tab_line(line):
            tab_line_count += 1
            pending_chord_line = None
            continue

        # Section headers
        # Check for strict glossary header (suspends section building)
        if GLOSSARY_SECTION_PATTERN.match(line.strip()):
            current_section = None   # suspend section building
            in_glossary = True
            in_inline_glossary = False
            pending_chord_line = None
            result['notes'].append(line.strip())
            continue

        # Check for inline glossary header (goes to notes, absorbs voicing lines
        # until next paragraph break, does NOT suspend section creation)
        if GLOSSARY_INLINE_PATTERN.match(line.strip()):
            in_inline_glossary = True
            pending_chord_line = None
            result['notes'].append(line.strip())
            continue

        # Inline glossary mode: absorb voicing-annotation lines into notes
        if in_inline_glossary:
            result['notes'].append(line.strip())
            continue

        section_label = detect_section(line)
        if section_label:
            in_glossary = False
            in_inline_glossary = False
            current_section = {'name': section_label, 'blocks': []}
            result['sections'].append(current_section)
            pending_chord_line = None
            continue

        # Default section if tab starts with no header
        if current_section is None:
            current_section = {'name': 'unlabeled', 'blocks': []}
            result['sections'].append(current_section)

        # Performance directions — store as a direction block, don't treat as a lyric.
        if is_performance_direction(line):
            if current_section:
                current_section['blocks'].append({
                    'chords':        [],
                    'passing_notes': [],
                    'lyric':         '',
                    'direction':     line.strip()
                })
            pending_chord_line = None
            continue

        # Chord lines — park and wait for lyric below
        if is_chord_line(line):
            pending_chord_line = line
            continue

        # Lyric lines
        if pending_chord_line:
            cleaned_line, passing = strip_passing_notes(pending_chord_line)
            # Use finditer instead of findall to capture column positions.
            # col is measured from the start of the chord line — the lyric
            # line may have a leading indent, which the renderer accounts for.
            chord_matches = [
                {'chord': normalize_chord(m.group()), 'col': m.start()}
                for m in CHORD_PATTERN.finditer(cleaned_line)
            ]
            block = {
                'chords':            [cm['chord'] for cm in chord_matches],
                'chords_positioned': chord_matches,
                'passing_notes':     passing,
                'lyric':             line.strip()
            }
            pending_chord_line = None
        else:
            block = {
                'chords':            [],
                'chords_positioned': [],
                'passing_notes':     [],
                'lyric':             line.strip()
            }

        current_section['blocks'].append(block)

    # ── Warnings ────────────────────────────────────────────────────────────
    if tab_line_count > 0:
        result['warnings'].append(
            f"Found {tab_line_count} raw ASCII tab lines — skipped (phase 2)."
        )
    if fret_diagram_count > 0:
        result['warnings'].append(
            f"Found {fret_diagram_count} fret diagram(s) — skipped."
        )

    # ── Clean up empty sections ──────────────────────────────────────────────
    result['sections'] = [s for s in result['sections'] if s['blocks']]

    # ── Strip sign-off and legend lines FIRST ──────────────────────────
    # Must run before merge so legend lines don't chain-merge with real lyrics.
    result = clean_sign_offs(result)

    # ── Split unlabeled sections at paragraph boundaries ───────────────────
    # Tabs with no [Section] headers produce one giant 'unlabeled' section.
    # Split it into numbered parts at natural blank-line gaps.
    result = _split_unlabeled_sections(result)

    # ── Merge continuation lyric lines (split across two text lines) ──────
    result = merge_continuation_blocks(result)

    # ── Strip tab-reference directions (Repeat above, see tab, etc.) ────
    # These refer to the raw ASCII tablature notation which we skip entirely,
    # so directions that reference it serve no purpose in the chord chart.
    for section in result['sections']:
        section['blocks'] = [
            b for b in section['blocks']
            if not (b.get('direction') and TAB_REF_DIRECTION.search(b.get('direction', '')))
        ]
    result['sections'] = [s for s in result['sections'] if s['blocks']]

    return result
