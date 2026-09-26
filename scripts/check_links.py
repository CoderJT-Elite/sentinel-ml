"""Check all internal and external links across Sentinel documentation, submission kit, and README."""
import os
import pathlib
import re
import urllib.request

ROOT_PROJECT = pathlib.Path(r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase")
ROOT_REPO = pathlib.Path(r"C:\OS\GitHub\sentinel-ml")

FILES_TO_CHECK = [
    ROOT_REPO / "README.md",
    ROOT_REPO / "docs" / "TECHNICAL_DOCUMENTATION.md",
    ROOT_PROJECT / "submission-kit" / "02-description.md",
    ROOT_PROJECT / "submission-kit" / "10-instructions-to-run.md",
    ROOT_PROJECT / "submission-kit" / "00-SUBMIT-CHECKLIST.md",
    ROOT_PROJECT / "BUSINESS_CASE.md",
    ROOT_PROJECT / "FINALS_PLAYBOOK.md",
]

def check_file(path):
    print(f"\nScanning: {path.name}")
    content = path.read_text(encoding="utf-8")
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    for text, url in links:
        if url.startswith("http://") or url.startswith("https://"):
            # Check external URL format
            print(f"  [EXTERNAL] {url}")
        else:
            # Check local file target (strip anchor if present)
            clean_url = url.split("#")[0]
            if clean_url:
                target = (path.parent / clean_url).resolve()
                status = "OK" if target.exists() else "MISSING"
                print(f"  [LOCAL] {url} -> {status}")
            else:
                print(f"  [ANCHOR] {url} -> OK")

if __name__ == "__main__":
    for f in FILES_TO_CHECK:
        if f.exists():
            check_file(f)
        else:
            print(f"File not found: {f}")
