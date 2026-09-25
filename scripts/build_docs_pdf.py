"""Render docs/TECHNICAL_DOCUMENTATION.md to a PDF (needs: pip install markdown playwright)."""
import pathlib

import markdown
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
md = (ROOT / "docs" / "TECHNICAL_DOCUMENTATION.md").read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
css = """
@page { size: Letter; margin: 0.75in; }
body { font: 10.5pt/1.5 'Segoe UI', Calibri, Arial, sans-serif; color: #101418; }
h1 { font: 700 24pt Georgia, serif; border-bottom: 3px solid #d8001b; padding-bottom: 6px; }
h2 { font: 700 15pt Georgia, serif; margin-top: 22px; border-bottom: 1px solid #cdd3d8; padding-bottom: 3px; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; page-break-inside: auto; }
th { background: #101418; color: #fff; text-align: left; padding: 5px 7px; }
td { border-bottom: 1px solid #e1e5e8; padding: 5px 7px; vertical-align: top; }
tr { page-break-inside: avoid; }
code { font: 9pt Consolas, monospace; background: #f1f3f4; padding: 1px 4px; }
pre { background: #f6f6f3; border: 1px solid #cdd3d8; padding: 10px; font: 8.5pt/1.35 Consolas, monospace; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #d8001b; margin: 10px 0; padding: 2px 14px; color: #2a3138; background: #fafafa; }
"""
html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.set_content(html)
    pg.pdf(path=str(ROOT / "docs" / "TECHNICAL_DOCUMENTATION.pdf"), format="Letter", print_background=True,
           margin={"top": "0.75in", "bottom": "0.75in", "left": "0.75in", "right": "0.75in"})
    b.close()
print("wrote docs/TECHNICAL_DOCUMENTATION.pdf")
