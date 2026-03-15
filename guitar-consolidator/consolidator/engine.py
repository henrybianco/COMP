"""
consolidator/engine.py
"""

import re
from collections import Counter


def similarity(a, b):
    """
    Word-level Jaccard similarity between two lyric lines.

    Normalizations applied before comparison:
      - Lowercase and strip punctuation
      - Collapse elongated characters: "Ruuuuuun" -> "run"
    Length ratio guard: lines with very different word counts score 0.
    """

    def collapse(w):
        # Replace any character repeated 3+ times with a single instance
        # e.g. "ruuuuuun" -> "run",  "noooo" -> "no"
        result = []
        i = 0
        while i < len(w):
            c = w[i]
            j = i
            while j < len(w) and w[j] == c:
                j += 1
            # if char repeated 3+ times, collapse to 1; else keep as-is
            if j - i >= 3:
                result.append(c)
            else:
                result.extend(w[i:j])
            i = j
        return ''.join(result)

    def words(s):
        # Strip ellipsis pause markers (…, ……, ...) before tokenising —
        # these appear in some tabs as musical timing cues, not lyric words.
        s = re.sub(r'[…\.]{2,}', ' ', s)
        punct = set('.,!?"')
        result = set()
        for w in s.split():
            cleaned = w.lower()
            for p in punct:
                cleaned = cleaned.replace(p, '')
            cleaned = cleaned.replace("'", '')
            cleaned = collapse(cleaned)
            if cleaned:
                result.add(cleaned)
        return result

    a_words = words(a)
    b_words = words(b)

    if not a_words and not b_words:
        return 1.0
    if not a_words or not b_words:
        return 0.0
    if a.strip() == b.strip():
        return 1.0

    a_len = len(a.split())
    b_len = len(b.split())
    if a_len > 0 and b_len > 0:
        ratio = min(a_len, b_len) / max(a_len, b_len)
        if ratio < 0.55:
            return 0.0

    intersection = len(a_words & b_words)
    union        = len(a_words | b_words)
    return round(intersection / union, 4) if union > 0 else 0.0


def normalize_section_name(name):
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def extract_song_structure(ranked_tabs, corpus=None):
    """
    Builds the ordered section sequence for the song.

    Selects the structurally richest tab as the template — the one with
    the most distinct normalized section types (not just highest score).
    When two tabs tie on richness, the higher-scored one wins.

    A richer structure (refrain, link, instrumental clearly labelled) is
    preferred over a sparser one even if the sparser tab scored higher on
    chord consistency, because structure determines what gets rendered.

    If a corpus is provided, attempts to relabel any 'unlabeled' sections
    using learned structure priors.
    """
    if not ranked_tabs:
        return []

    def richness(tab):
        names = {normalize_section_name(s['name'])
                 for s in tab.get('sections', [])}
        return len(names)

    # Pick the richest tab; break ties by score (higher is better)
    template_tab = max(
        ranked_tabs,
        key=lambda t: (richness(t), t.get('score', 0))
    )

    structure = []
    sections  = template_tab.get('sections', [])
    for i, section in enumerate(sections):
        original   = section['name']
        normalized = normalize_section_name(original)

        # Try to relabel unlabeled sections using structure priors
        if normalized == 'unlabeled' and corpus is not None:
            preceding     = structure[-1][0] if structure else None
            following_raw = sections[i+1]['name'] if i+1 < len(sections) else None
            following     = normalize_section_name(following_raw) if following_raw else None

            inferred = corpus.infer_section_label(
                block_count = len(section.get('blocks', [])),
                preceding   = preceding,
                following   = following,
            )
            if inferred:
                normalized = inferred
                original   = inferred

        structure.append((normalized, original))
    return structure


def _merge_duplicate_sections(tab):
    """Merge consecutive duplicate section labels within a single tab.

    Some tabs repeat the same section label (e.g. two [Intro] sections, one
    for an ASCII fingering diagram and one for the chord grid). Merging them
    ensures group_sections always sees a clean one-section-per-label structure,
    so the chord-grid block is not silently dropped.

    Paragraph-break sentinels between merged duplicates are discarded (the
    section boundary serves the same purpose).
    """
    sections = tab.get('sections', [])
    if not sections:
        return tab
    merged = []
    for section in sections:
        name = section['name']
        content_blocks = [b for b in section.get('blocks', [])
                          if not b.get('paragraph_break')]
        if merged and merged[-1]['name'] == name:
            merged[-1]['blocks'].extend(content_blocks)
        else:
            merged.append({'name': name, 'blocks': list(content_blocks)})
    tab = dict(tab)
    tab['sections'] = merged
    return tab


def group_sections(ranked_tabs):
    # Merge any duplicate section names within each tab before grouping.
    ranked_tabs = [_merge_duplicate_sections(t) for t in ranked_tabs]

    # Build one template per distinct numbered variant (verse_1, verse_2...)
    # rather than collapsing all verses into one shared template.
    #
    # POSITIONAL PAIRING: When Tab A uses verse_1/verse_2/verse_3 but Tab B
    # uses three plain [Verse] sections, we pair them by position:
    #   verse_1  ↔  Tab B's 1st [Verse]
    #   verse_2  ↔  Tab B's 2nd [Verse]
    #   verse_3  ↔  Tab B's 3rd [Verse]
    # This prevents Tab B's verse 1 from being reused for every variant.

    # Collect all distinct section keys seen across all tabs, preserving order
    all_keys = []
    seen_keys = set()
    for tab in ranked_tabs:
        for section in tab.get('sections', []):
            name = section['name']
            if name not in seen_keys:
                all_keys.append(name)
                seen_keys.add(name)

    # Pre-compute positional index: for each tab, build a per-norm-type list
    # of sections in document order.  When we need the N-th verse from a tab
    # that uses plain [Verse] labels, we can look it up by position.
    def positional_index(tab):
        """Returns {norm_type: [section, section, ...]} in document order."""
        idx = {}
        for section in tab.get('sections', []):
            norm = normalize_section_name(section['name'])
            idx.setdefault(norm, []).append(section)
        return idx

    tab_pos_idx = {id(tab): positional_index(tab) for tab in ranked_tabs}

    # For each key, determine which position (0-based) it is among keys of
    # the same normalized type — e.g. verse_1 is position 0, verse_2 is 1.
    key_positions = {}
    type_counter  = {}
    for key in all_keys:
        norm = normalize_section_name(key)
        pos  = type_counter.get(norm, 0)
        key_positions[key] = pos
        type_counter[norm] = pos + 1

    groups = {}

    # ── Special handling for unlabeled sections ──────────────────────────────
    # Positional pairing is unreliable when tabs use different paragraph breaks.
    # Instead: for each unlabeled_N in the anchor tab (highest-scored tab),
    # find the best-matching unlabeled section from other tabs by lyric overlap.
    def _section_lyric_words(section):
        words = set()
        for b in section.get('blocks', []):
            import re as _re
            lyric = b.get('lyric', '').strip().lower()
            words |= set(_re.sub(r"[^a-z0-9]", ' ', lyric).split())
        return words

    unlabeled_keys = [k for k in all_keys if k.startswith('unlabeled')]
    if unlabeled_keys and len(ranked_tabs) > 1:
        anchor_tab = ranked_tabs[0]
        anchor_unlabeled = [s for s in anchor_tab.get('sections', [])
                            if s['name'].startswith('unlabeled')]

        # Build a mapping: for each non-anchor tab, find its best match for
        # each anchor unlabeled section by lyric-word Jaccard similarity.
        # A match requires ≥ 30% word overlap OR both sections are chord-only.
        def _best_match(anchor_section, candidate_sections, used_indices):
            anchor_words = _section_lyric_words(anchor_section)
            best_score, best_idx = -1, None
            for i, cand in enumerate(candidate_sections):
                if i in used_indices:
                    continue
                cand_words = _section_lyric_words(cand)
                if anchor_words and cand_words:
                    overlap = len(anchor_words & cand_words)
                    union   = len(anchor_words | cand_words)
                    score   = overlap / union if union else 0
                    if score >= 0.30 and score > best_score:
                        best_score, best_idx = score, i
                elif not anchor_words and not cand_words:
                    # Both chord-only — match positionally as fallback
                    if best_idx is None:
                        best_score, best_idx = 0.0, i
            return best_idx

        used_indices_per_tab = {}  # separate tracking — NOT stored in groups

        for key in unlabeled_keys:
            target_pos = key_positions[key]
            if target_pos >= len(anchor_unlabeled):
                continue
            anchor_sec = anchor_unlabeled[target_pos]
            anchor_score = anchor_tab.get('score', 1.0)
            entries = [(anchor_tab, anchor_sec, anchor_score)]

            for other_tab in ranked_tabs[1:]:
                other_score = other_tab.get('score', 1.0)
                other_unlabeled = [s for s in other_tab.get('sections', [])
                                   if s['name'].startswith('unlabeled')]
                # Track already-matched sections to prevent double-use
                tab_id = id(other_tab)
                used = used_indices_per_tab.setdefault(tab_id, set())
                idx = _best_match(anchor_sec, other_unlabeled, used)
                if idx is not None:
                    used.add(idx)
                    entries.append((other_tab, other_unlabeled[idx], other_score))
                elif other_unlabeled:
                    # Fallback: positional
                    pos = min(target_pos, len(other_unlabeled) - 1)
                    entries.append((other_tab, other_unlabeled[pos], other_score))

            groups[key] = entries

    for key in all_keys:
        if key.startswith('unlabeled'):
            continue  # already handled above
        norm_key = normalize_section_name(key)
        target_pos = key_positions[key]
        entries    = []

        for tab in ranked_tabs:
            tab_score = tab.get('score', 1.0)

            # 1. Exact name match (e.g. both tabs use verse_2)
            exact = next(
                (s for s in tab.get('sections', []) if s['name'] == key),
                None
            )
            if exact:
                entries.append((tab, exact, tab_score))
                continue

            # 2. Positional match: pick the N-th section of this norm type
            #    e.g. for verse_2 (position 1), pick Tab B's 2nd [Verse]
            same_type = tab_pos_idx[id(tab)].get(norm_key, [])
            if target_pos < len(same_type):
                entries.append((tab, same_type[target_pos], tab_score))
                continue

            # 3. Last resort: the last available section of this type
            #    (handles tabs that have fewer repeats than the anchor)
            if same_type:
                entries.append((tab, same_type[-1], tab_score))

        if entries:
            groups[key] = entries

    return groups


def align_blocks(sections_with_scores):
    if not sections_with_scores:
        return []

    sorted_sections = sorted(sections_with_scores, key=lambda x: x[2], reverse=True)

    # FIX 1: Choose anchor as the longest section from the top-scored tab.
    # If the top-scored tab has a shorter section than others (e.g. combined
    # lines), using a longer section as anchor ensures all lines get coverage.
    top_score = sorted_sections[0][2]
    top_score_sections = [e for e in sorted_sections if e[2] == top_score]
    # Among top-scored tab's sections, pick longest; else pick overall longest
    best_anchor = max(sorted_sections, key=lambda e: len(e[1].get('blocks', [])))
    # But prefer a top-scored section if it's within 50% of the longest
    longest_count = len(best_anchor[1].get('blocks', []))
    top_long = max(top_score_sections, key=lambda e: len(e[1].get('blocks', [])))
    top_count = len(top_long[1].get('blocks', []))
    anchor_tab, anchor_section, anchor_score = (
        top_long if top_count >= longest_count * 0.5 else best_anchor
    )

    matched_keys = set()
    aligned_rows = []

    # Pass 1: anchor-led matching
    for anchor_block in anchor_section.get('blocks', []):
        anchor_lyric = anchor_block.get('lyric', '').strip()
        row = [{
            'chords':            anchor_block.get('chords', []),
            'chords_positioned': anchor_block.get('chords_positioned', []),
            'passing':           anchor_block.get('passing_notes', []),
            'lyric':             anchor_lyric,
            'direction':         anchor_block.get('direction', ''),
            'score':             anchor_score
        }]
        for other_tab, other_section, other_score in sorted_sections[1:]:
            best_block = None
            best_sim   = 0.0
            best_key   = None
            for blk_idx, other_block in enumerate(other_section.get('blocks', [])):
                other_lyric = other_block.get('lyric', '').strip()

                # FIX 3a: direction-only blocks (e.g. "(x3, very short)")
                # When both blocks have a direction and no chords/lyric, match them.
                anchor_dir = anchor_block.get('direction', '')
                other_dir  = other_block.get('direction', '')
                if anchor_dir and other_dir and not anchor_lyric and not other_lyric:
                    # Any two direction blocks in the same section match each other —
                    # they're both annotating the same musical moment
                    if 1.0 > best_sim:
                        best_sim   = 1.0
                        best_block = other_block
                        best_key   = (id(other_tab), id(other_section), blk_idx)
                    continue

                # FIX 3b: chord-only blocks (intros, instrumental lines)
                # When both lines have no lyrics, compare chord sets instead.
                if not anchor_lyric and not other_lyric:
                    anchor_chords = set(anchor_block.get('chords', []))
                    other_chords  = set(other_block.get('chords', []))
                    if anchor_chords and other_chords:
                        inter = len(anchor_chords & other_chords)
                        union = len(anchor_chords | other_chords)
                        chord_sim = inter / union if union > 0 else 0.0
                        if chord_sim > best_sim and chord_sim >= 0.50:
                            best_sim   = chord_sim
                            best_block = other_block
                            best_key   = (id(other_tab), id(other_section), blk_idx)
                    continue  # don't fall through to lyric matching

                if not anchor_lyric or not other_lyric:
                    continue
                sim = similarity(anchor_lyric, other_lyric)
                if sim > best_sim and sim >= 0.45:
                    best_sim   = sim
                    best_block = other_block
                    best_key   = (id(other_tab), id(other_section), blk_idx)
            if best_block and best_key:
                matched_keys.add(best_key)
                row.append({
                    'chords':            best_block.get('chords', []),
                    'chords_positioned': best_block.get('chords_positioned', []),
                    'passing':           best_block.get('passing_notes', []),
                    'lyric':             best_block.get('lyric', '').strip(),
                    'direction':         best_block.get('direction', ''),
                    'score':             other_score,
                    'similarity':        best_sim
                })
        aligned_rows.append(row)

    # Pass 2: orphan recovery
    # Only insert an orphan if its lyric is not already covered by any
    # existing row in aligned_rows. We check against all existing row lyrics
    # using the same similarity threshold. This prevents duplicate lines when
    # the anchor tab had a shorter/longer version of the same lyric.
    existing_lyrics = [
        entry.get('lyric', '')
        for row in aligned_rows
        for entry in row
        if entry.get('lyric')
    ]

    for other_tab, other_section, other_score in sorted_sections[1:]:
        other_blocks = other_section.get('blocks', [])
        section_len  = len(other_blocks)
        for blk_idx, other_block in enumerate(other_blocks):
            key = (id(other_tab), id(other_section), blk_idx)
            if key in matched_keys:
                continue
            other_lyric  = other_block.get('lyric', '').strip()
            other_chords = other_block.get('chords', [])
            if not other_lyric and not other_chords:
                continue

            # Skip if this lyric is already represented in an existing row.
            # Two checks:
            # (a) Jaccard similarity ≥ 0.50 — catches paraphrases and near-matches
            # (b) Word containment — catches split-line halves where one tab writes
            #     "Old man take a look at my life I'm a lot like you." as one block
            #     and another tab splits it into two lines. Jaccard scores 0 because
            #     the combined line has many more words, but all words of the shorter
            #     line are present in the combined line.
            def word_set(s):
                import re as _re
                return set(_re.sub(r"[^a-z0-9\s]", '', s.lower()).split())

            orphan_words = word_set(other_lyric)
            already_covered = False
            if orphan_words:
                for existing in existing_lyrics:
                    if not existing:
                        continue
                    if similarity(other_lyric, existing) >= 0.50:
                        already_covered = True
                        break
                    existing_words = word_set(existing)
                    # Containment: ≥ 85% of orphan's words found in the existing line
                    if existing_words and len(orphan_words & existing_words) / len(orphan_words) >= 0.85:
                        already_covered = True
                        break
            if already_covered:
                matched_keys.add(key)
                continue

            if aligned_rows:
                proportion   = blk_idx / max(section_len - 1, 1)
                insert_after = min(int(proportion * len(aligned_rows)), len(aligned_rows) - 1)
            else:
                insert_after = 0

            aligned_rows.insert(insert_after + 1, [{
                'chords':            other_chords,
                'chords_positioned': other_block.get('chords_positioned', []),
                'passing':           other_block.get('passing_notes', []),
                'lyric':             other_lyric,
                'direction':         other_block.get('direction', ''),
                'score':             other_score,
                'orphan':            True
            }])
            matched_keys.add(key)
            existing_lyrics.append(other_lyric)

    return aligned_rows


def _dedup_repeated_chords(chord_seq):
    # Collapse repeated-single-chord sequences before voting to avoid
    # false disputes: ['F','F'] vs ['F'] should be the same vote.
    # Only collapses when ALL elements are identical.
    if not chord_seq or len(set(chord_seq)) != 1:
        return tuple(chord_seq)
    return (chord_seq[0],)


def _chord_complexity(chord: str) -> int:
    """
    Returns a numeric complexity score for a single chord name.
    Higher = more harmonic information was transcribed.

    Used as a tiebreaker in vote_on_chords: when two chord sequences
    receive equal vote weight, the more detailed transcription wins.

    Scoring:
      base = 0
      +1 for each quality extension present (7, maj7, sus, add, etc.)
      +1 for slash/bass note (voicing detail)
      +1 for alterations (#5, b5, #9, b9, etc.)
      -2 for augmented/diminished (those are sometimes tab errors)

    Notably, minor vs major quality does NOT score here — that distinction
    is a harmonic disagreement, not a complexity difference, and should
    remain as a genuine dispute rather than being silently resolved.
    """
    if not chord:
        return 0
    # Strip root
    if len(chord) >= 2 and chord[1] in ('#', 'b'):
        quality = chord[2:]
    else:
        quality = chord[1:]

    score = 0
    if '/' in quality:
        score += 1          # slash chord = extra voicing info
        quality = quality.split('/')[0]

    EXTENSIONS = ('maj7', 'maj9', 'maj11', 'maj13',
                  '7', '9', '11', '13',
                  'sus2', 'sus4', 'add9', 'add11',
                  '6', '69')
    ALTERATIONS = ('#5', 'b5', '#9', 'b9', '#11', 'b13', 'b6')

    for ext in EXTENSIONS:
        if ext in quality:
            score += 1
    for alt in ALTERATIONS:
        if alt in quality:
            score += 1

    return score


def _sequence_complexity(chord_list: list) -> int:
    """Total complexity of a chord sequence."""
    return sum(_chord_complexity(c) for c in chord_list)


def vote_on_chords(aligned_row):
    if not aligned_row:
        return None
    vote_tally = {}
    for entry in aligned_row:
        # Normalise repeated identical chords before tallying so that
        # ['F','F'] vs ['F'] does not register as a chord dispute.
        chords = _dedup_repeated_chords(entry.get('chords', []))
        weight = entry.get('score', 1.0)
        if entry.get('orphan'):
            weight *= 0.5
        vote_tally[chords] = vote_tally.get(chords, 0.0) + weight

    total_votes = sum(vote_tally.values())
    # Primary sort: highest vote weight.
    # Tiebreaker: higher chord complexity (transcriber heard more detail).
    # We only apply the complexity tiebreaker when the vote difference is
    # negligible (< 2% of total), to avoid overriding genuine vote signals.
    winner = max(
        vote_tally,
        key=lambda k: (
            vote_tally[k],
            _sequence_complexity(list(k)),
        ),
    )

    # If the top vote-getter is the empty chord sequence () but non-empty
    # alternatives exist, elect the best non-empty candidate instead.
    # An empty winner means Tab A simply had no chord notation for that line —
    # it should not override a real chord annotation from another source.
    if winner == () and len(vote_tally) > 1:
        non_empty = {k: v for k, v in vote_tally.items() if k}
        if non_empty:
            winner = max(non_empty, key=lambda k: non_empty[k])

    confidence  = round(vote_tally[winner] / total_votes, 4) if total_votes > 0 else 0.0
    alternatives = [
        {'chords': list(seq), 'weighted_score': round(sc, 4),
         'vote_share': round(sc / total_votes, 4) if total_votes else 0}
        for seq, sc in sorted(vote_tally.items(), key=lambda x: -x[1])
        if seq != winner and seq  # exclude empty-chord alternatives
    ]
    best_entry = sorted(aligned_row, key=lambda e: e.get('score', 0), reverse=True)[0]
    best_lyric = best_entry.get('lyric', '')
    direction  = best_entry.get('direction', '')

    # Elect positions from the highest-scored entry whose chords match the winner.
    # Position and chord choice are coupled — they come from the same transcriber.
    # If no entry matches (e.g. corpus overrode the winner), fall back to best_entry.
    winner_list = list(winner)
    position_source = next(
        (e for e in sorted(aligned_row, key=lambda e: e.get('score', 0), reverse=True)
         if e.get('chords') == winner_list),
        best_entry
    )
    chords_positioned = position_source.get('chords_positioned', [])
    disputed   = confidence < 0.60

    # If disputed, ask the corpus whether it can break the tie.
    # Pass the current winner and all alternatives as candidate options.
    # The corpus returns its preferred option if it has enough data,
    # or None if it can't meaningfully distinguish.
    if disputed and alternatives:
        try:
            from knowledge.corpus import corpus as _corpus
            key = best_entry.get('key', '')  # set by consolidate() if available
            all_options = [list(winner)] + [a['chords'] for a in alternatives]
            corpus_pick = _corpus.resolve_chord_dispute(all_options, key)
            if corpus_pick is not None and corpus_pick != list(winner):
                # Corpus overrides vote winner — record as a corpus-resolved line
                winner     = tuple(corpus_pick)
                disputed   = False  # corpus resolved it
        except ImportError:
            pass

    return {
        'chords':            list(winner),
        'chords_positioned': chords_positioned,
        'lyric':             best_lyric,
        'direction':         direction,
        'confidence':        confidence,
        'disputed':          disputed,
        'alternatives':      alternatives if disputed else [],
        'corpus_resolved':   not disputed and confidence < 0.60
    }


def build_section_templates(ranked_tabs):
    groups    = group_sections(ranked_tabs)
    templates = {}
    for section_name, entries in groups.items():
        aligned_rows = align_blocks(entries)
        lines = [vote_on_chords(row) for row in aligned_rows]
        templates[section_name] = [l for l in lines if l and (l['chords'] or l['lyric'] or l.get('direction'))]
    return templates


def consolidate(ranked_tabs, song_title='', artist='', target_key=''):
    result = {
        'title': song_title, 'artist': artist, 'key': target_key,
        'tuning': '', 'capo': 0, 'source_count': len(ranked_tabs),
        'structure': [], 'sections': {}, 'chord_glossary': [],
        'notes': [], 'warnings': []
    }
    if not ranked_tabs:
        result['warnings'].append("No tabs provided.")
        return result

    # Load corpus once for this consolidation — used for structure inference
    # and chord dispute resolution. Gracefully absent if package not installed.
    try:
        from knowledge.corpus import corpus as _corpus
    except ImportError:
        _corpus = None

    top_tab             = ranked_tabs[0]
    result['tuning']    = top_tab.get('tuning', 'Standard')
    result['capo']      = top_tab.get('capo', 0)
    result['notes']     = top_tab.get('notes', [])
    result['structure'] = extract_song_structure(ranked_tabs, corpus=_corpus)
    result['sections']  = build_section_templates(ranked_tabs)

    all_chords = []
    for lines in result['sections'].values():
        for line in lines:
            for chord in line.get('chords', []):
                if chord not in all_chords:
                    all_chords.append(chord)
    result['chord_glossary'] = sorted(set(all_chords))

    for tab in ranked_tabs:
        for w in tab.get('warnings', []):
            if w not in result['warnings']:
                result['warnings'].append(w)

    # ── Cross-section lyric deduplication ───────────────────────────────────
    # When a section's lyric content duplicates another section's content
    # (e.g. Tab A folded the refrain into its intro), remove the duplicates
    # from the section where they don't belong.
    #
    # Walk sections in specificity order (most-specific first), so that
    # 'refrain' claims its lyrics before 'intro' or 'outro' can claim them.
    # This prevents Tab A's intro (which contains refrain content) from
    # winning the ownership race over Tab B's explicit refrain section.
    #
    # Specificity tier: named literary sections > structural wrappers
    def _word_set(s):
        import re as _re
        return set(_re.sub(r"[^a-z0-9\s]", '', s.lower()).split())

    def _lyric_matches(a, b):
        """True if lyrics a and b refer to the same musical line.
        Catches both near-identical strings and split/combined variants."""
        if similarity(a, b) >= 0.95:
            return True
        # Containment: one line is a subset of the other (split vs combined)
        wa, wb = _word_set(a), _word_set(b)
        if not wa or not wb:
            return False
        shorter, longer = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
        return len(shorter & longer) / len(shorter) >= 0.90

    SECTION_SPECIFICITY = {
        'refrain': 10, 'hook': 10,
        'verse': 8, 'chorus': 8, 'bridge': 8, 'pre_chorus': 8,
        'link': 6, 'interlude': 6,
        'instrumental': 4, 'solo': 4,
        'intro': 2, 'outro': 2,
        'unlabeled': 0,
    }

    def section_specificity(norm_name):
        return SECTION_SPECIFICITY.get(norm_name, 5)

    # Sort structure entries by specificity descending for the claim pass,
    # then restore original order for the actual removal pass.
    structure_by_specificity = sorted(
        result['structure'],
        key=lambda pair: section_specificity(pair[0]),
        reverse=True
    )

    all_seen = []   # list of (lyric, chord_set, section_name)

    # First pass: claim lyrics in specificity order
    for norm_name, orig_name in structure_by_specificity:
        template_key = orig_name if orig_name in result['sections'] else norm_name
        template = result['sections'].get(template_key, [])
        for line in template:
            lyric  = line.get('lyric', '').strip()
            chords = set(line.get('chords', []))
            if not lyric:
                continue
            lyric_word_count = len(lyric.split())
            already_claimed = False
            for seen_lyric, seen_chords, seen_section in all_seen:
                if seen_section == template_key:
                    continue
                if not _lyric_matches(lyric, seen_lyric):
                    continue
                # Chord context guard: if both sides have chords and they share
                # < 50% of chord names, treat as a different musical moment.
                # Exception: very short lyrics (≤ 3 words) are almost certainly
                # line fragments from a merged line — skip the chord guard for them
                # so "you were." doesn't survive as an intro artifact.
                if chords and seen_chords and lyric_word_count > 3:
                    overlap = len(chords & seen_chords) / len(chords | seen_chords)
                    if overlap < 0.50:
                        continue
                already_claimed = True
                break
            if not already_claimed:
                all_seen.append((lyric, chords, template_key))

    # Second pass: remove lines not owned by this section
    for norm_name, orig_name in result['structure']:
        template_key = orig_name if orig_name in result['sections'] else norm_name
        template = result['sections'].get(template_key, [])
        to_keep  = []
        for line in template:
            lyric  = line.get('lyric', '').strip()
            chords = set(line.get('chords', []))
            if not lyric:
                to_keep.append(line)
                continue
            lyric_word_count = len(lyric.split())
            # Keep if this section owns the lyric (or lyric is unclaimed)
            owner = None
            for seen_lyric, seen_chords, seen_section in all_seen:
                if not _lyric_matches(lyric, seen_lyric):
                    continue
                if chords and seen_chords and lyric_word_count > 3:
                    overlap = len(chords & seen_chords) / len(chords | seen_chords)
                    if overlap < 0.50:
                        continue
                owner = seen_section
                break
            if owner is None or owner == template_key:
                to_keep.append(line)
            # else: drop — owned by a different, more-specific section
        result['sections'][template_key] = to_keep


    # ── Deduplicate repeated unlabeled sections ──────────────────────────────
    # When a tab has no section headers (e.g. Blackbird), the paragraph-break
    # splitter creates unlabeled_1 … unlabeled_N. If the same verse appears
    # multiple times (as it often does in a song), multiple sections will have
    # identical or near-identical content. Collapse those duplicates by removing
    # them from the structure list while keeping the sections dict intact
    # (so the first occurrence still renders correctly).
    unlabeled_structure = [(n, o) for n, o in result['structure']
                           if n.startswith('unlabeled')]
    if unlabeled_structure:
        def _section_fingerprint(blocks):
            """Stable fingerprint of a section's content for dedup."""
            lines = []
            for b in blocks:
                lyric = b.get('lyric', '').strip().lower()
                chords = tuple(sorted(b.get('chords', [])))
                if lyric or chords:
                    lines.append((lyric[:60], chords))
            return tuple(lines)

        seen_fingerprints = set()
        keep_structure = []
        for norm_name, orig_name in result['structure']:
            if not norm_name.startswith('unlabeled'):
                keep_structure.append((norm_name, orig_name))
                continue
            key = orig_name if orig_name in result['sections'] else norm_name
            blocks = result['sections'].get(key, [])
            fp = _section_fingerprint(blocks)
            if fp and fp in seen_fingerprints:
                # Duplicate — drop from structure (section template stays for
                # any cross-reference, but it won't be rendered)
                continue
            seen_fingerprints.add(fp)
            keep_structure.append((norm_name, orig_name))
        result['structure'] = keep_structure

    # Record everything learnable from this consolidation into the corpus.
    # This is what makes the system improve with each song processed.
    if _corpus is not None:
        try:
            _corpus.record(result, ranked_tabs)
        except Exception:
            pass  # never let corpus I/O break a consolidation

    return result
