# Live Testing Playbook

Step-by-step guide for the first live run against Ultimate Guitar.

---

## Setup (once)

```bash
pip install cloudscraper beautifulsoup4
```

Verify:
```bash
python3 -c "import cloudscraper; print('OK')"
```

---

## Test sequence

Run these in order — each one adds confidence before the next.

### 1. Sanity check: mock mode still works on your machine

```bash
python scraper.py "Creep" "Radiohead" --key G --mock --verbose
```

Expected: 8 sections, 4 chords, 0 disputed lines, knowledge base updated.
If this fails, something is wrong with the install, not the network code.

---

### 2. First live fetch — simple, well-known song

```bash
python scraper.py "Blackbird" "The Beatles" --key G --n 3 --verbose --debug debug_html/
```

`--debug debug_html/` saves the raw HTML from UG to `debug_html/` so you have it if anything goes wrong.

**Expected output:**
```
  Guitar Tab Scraper
  Blackbird — The Beatles  (key: G)
  ────────────────────────────────────────────────────────────
  Searching UG: Blackbird — The Beatles
  Found N chord tab(s), fetching top 3
  [1/3] ★4.x (N votes)  https://tabs.ultimate-guitar.com/...
  ...
  ✓ N sections  ·  N chords  ·  N disputed line(s)
```

**Spot-check the output chords.** Blackbird is in G — you should see G, Am, C, D, Em family chords. If you see F#, Bb, Eb — transposition is wrong.

---

### 3. Check cache warm path works

```bash
python scraper.py "Blackbird" "The Beatles" --key G --n 3 --verbose
```

Should print `Cache hit — 3 file(s)` and skip all network calls. Same output as step 2.

---

### 4. Test a song with a non-obvious key

```bash
python scraper.py "Jolene" "Dolly Parton" --key Dm --n 3 --verbose
```

Jolene is in Dm/F. Expected chords: Dm, F, C, Am. If you see D, F#, A — something is wrong with minor key handling.

---

### 5. Test with artist whose name has spaces / punctuation

```bash
python scraper.py "Wish You Were Here" "Pink Floyd" --key G --n 3 --verbose
```

Tests URL encoding of multi-word artist + song names.

---

### 6. Cache refresh

```bash
python scraper.py "Blackbird" "The Beatles" --key G --n 3 --refresh --verbose
```

Should re-fetch from network even though cache exists. Useful after `--debug` reveals a parse issue.

---

## Failure modes and diagnosis

### "Fetch failed: HTTP 403" or "Cloudflare"

cloudscraper failed to bypass Cloudflare. Try:
1. Update cloudscraper: `pip install --upgrade cloudscraper`
2. Try a different browser profile — edit `_CLOUDSCRAPER_BROWSER` in `scraper.py`:
   ```python
   # Try Firefox instead of Chrome:
   _CLOUDSCRAPER_BROWSER = {"browser": "firefox", "platform": "windows", "mobile": False}
   ```
3. Add a delay: increase `REQUEST_DELAY` from 1.5 to 3.0

---

### "No results found on UG"

UG returned a valid page but no tab data was parsed from it.

1. Check the saved HTML: `ls debug_html/` → open `*_search.html` in a browser
2. If the page shows search results visually but we got nothing — UG changed their JSON structure
3. Diagnosis script:
   ```bash
   python3 - <<'EOF'
   from scraper import _extract_js_store, _parse_search_page
   html = open("debug_html/the_beatles_blackbird_search.html").read()
   try:
       store = _extract_js_store(html)
       print("js-store found, top-level keys:", list(store.keys()))
       results = store.get("store",{}).get("page",{}).get("data",{}).get("results",[])
       print(f"{len(results)} results in expected path")
       if results:
           print("First result keys:", list(results[0].keys()))
   except Exception as e:
       print("Error:", e)
   EOF
   ```
4. If `js-store` path changed, update `_parse_search_page()` in scraper.py.

---

### "Found N results but none are Chords tabs with ≥10 votes"

Two possible causes:
1. The `type=300` server-side filter is no longer working — remove it from `UG_SEARCH_URL` and let `ALLOWED_TYPES` do the client-side filtering instead
2. The search results have a `type` field with different casing — check with:
   ```bash
   python3 - <<'EOF'
   from scraper import _extract_js_store, _parse_search_page
   html = open("debug_html/..._search.html").read()
   store = _extract_js_store(html)
   results = store["store"]["page"]["data"]["results"]
   types = set(r.get("type") for r in results)
   print("Types seen:", types)
   EOF
   ```

---

### "Could not extract tab content"

The individual tab page was fetched but content extraction failed.

```bash
python3 - <<'EOF'
from scraper import _extract_js_store, _parse_tab_page
html = open("debug_html/..._tab_1.html").read()
try:
    store = _extract_js_store(html)
    tab_view = store["store"]["page"]["data"]["tab_view"]
    print("tab_view keys:", list(tab_view.keys()))
    wiki = tab_view.get("wiki_tab", {})
    print("wiki_tab keys:", list(wiki.keys()))
    print("content length:", len(wiki.get("content", "")))
except Exception as e:
    print("Error:", e)
EOF
```

If the content is at a different path, update `_parse_tab_page()` in scraper.py.

---

### Output chords look wrong (wrong key)

Run with `--verbose` to see the per-tab scoring and source key detection:

```bash
python scraper.py "Song" "Artist" --key X --verbose
```

Look for lines like:
```
  Scoring (3 tabs):
    1. score=15.4  rating=4.8  votes=900
```

Then inspect the cached files directly:
```bash
ls scraper_cache/artist_song/
cat scraper_cache/artist_song/song_ug_4.8_900_1.txt | head -30
```

If the raw tab content looks right but the output is wrong — it's a transposition bug. Report what key the raw tab is in vs what `--key` you passed.

---

### Knowledge base not updating

```bash
python scraper.py "Song" "Artist" --key X --verbose --no-learn
# vs
python scraper.py "Song" "Artist" --key X --verbose
```

Compare the `progressions_recorded` count. If it's 0 with learning enabled, check:
```bash
python3 -c "from knowledge.corpus import Corpus; c = Corpus(); print(c.stats())"
```

---

## What to report back

After live testing, note:
1. Which songs worked / failed
2. Exact error messages for any failures
3. Whether the `debug_html/` files were useful
4. Chord quality — did the output look musically correct?
5. Knowledge base counts before and after (run `python3 -c "from knowledge.corpus import Corpus; c=Corpus(); print(c.stats())"`)
