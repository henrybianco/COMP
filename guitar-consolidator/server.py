#!/usr/bin/env python3
"""
COMP local server
Usage: python3 server.py 
Open:  http://localhost:8080
"""
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

HTML_PATH = os.path.join(ROOT, "comp.html")


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/comp.html"):
            html = open(HTML_PATH, encoding="utf-8").read().encode()
            self._send(200, "text/html; charset=utf-8", html)
            return

        if parsed.path == "/generate":
            qs     = parse_qs(parsed.query)
            song   = unquote(qs.get("song",   [""])[0]).strip()
            artist = unquote(qs.get("artist", [""])[0]).strip()

            if not song or not artist:
                self._send(400, "text/plain", b"song and artist are required")
                return

            try:
                from scraper import fetch_tabs
                from scorer.scoring_engine import rank_tabs
                from parser.chord_parser import parse_tab
                from renderer import render

                print(f"  → {song} / {artist}")

                tab_paths, source_urls = fetch_tabs(song, artist, n=5)

                parsed_tabs = []
                for path in tab_paths:
                    raw = path.read_text(encoding="utf-8")
                    p   = parse_tab(raw)
                    if p.get("sections"):
                        parsed_tabs.append(p)

                if not parsed_tabs:
                    raise RuntimeError("No parseable tabs found.")

                ranked = rank_tabs(parsed_tabs)
                output = render(ranked[0], song, artist)

                if source_urls:
                    sources_block = "\nSources\n" + "─" * 72 + "\n"
                    for url in source_urls:
                        sources_block += f"  {url}\n"
                    output += "\n" + sources_block

                self._send(200, "text/plain; charset=utf-8", output.encode())

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send(500, "text/plain", str(e).encode())
            return

        self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"\n  COMP running →  http://localhost:{port}\n")
    HTTPServer(("", port), Handler).serve_forever()
