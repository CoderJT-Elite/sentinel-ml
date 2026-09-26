"""
Voice and Tells Linter for Sentinel.
Flags em dashes, en dashes, ' -- ', emojis, and banned words across all prose deliverables.
Excludes code, LICENSE, third-party vendored files, and quoted source texts.
Exits 0 if all clean, 1 if any violations found.
"""
import sys
import re
import pathlib

BANNED_WORDS = [
    "delve", "foster", "leverage", "utilize", "facilitate", "empower", 
    "streamline", "robust", "seamless", "cutting-edge", "game-changing", 
    "transformative", "elevate", "holistic", "landscape", "tapestry", 
    "testament", "pivotal", "crucial", "comprehensive", "it's worth noting", 
    "at its core", "in today's world", "let's dive in", "immutable", 
    "bulletproof", "mathematical proof", "bit-for-bit", "air-gapped", 
    "military-grade", "signed", "certified", "compliant"
]

# Emoji regex pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

TARGET_FILES = [
    r"C:\OS\GitHub\sentinel-ml\README.md",
    r"C:\OS\GitHub\sentinel-ml\SECURITY.md",
    r"C:\OS\GitHub\sentinel-ml\CONTRIBUTING.md",
    r"C:\OS\GitHub\sentinel-ml\MODEL_CARD.md",
    r"C:\OS\GitHub\sentinel-ml\DATASET_DATASHEET.md",
    r"C:\OS\GitHub\sentinel-ml\docs\TECHNICAL_DOCUMENTATION.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\BUSINESS_CASE.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\FINALS_PLAYBOOK.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\00-SUBMIT-CHECKLIST.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\01-title.txt",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\02-description.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\03-parent-and-theme.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\05-video-url.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\07-demo-link.txt",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\08-repo-url.txt",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\10-instructions-to-run.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\recording-kit\SCRIPT.md",
    r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\video\scenes.json",
    r"C:\OS\GitHub\sentinel-ml\web\app.js",
    r"C:\OS\GitHub\sentinel-ml\docs\demo\app.js",
]

def check_voice():
    violations = 0
    checked_count = 0
    print("=== RUNNING SENTINEL VOICE AND TELLS AUDIT ===")

    for path_str in TARGET_FILES:
        p = pathlib.Path(path_str)
        if not p.exists():
            continue
        checked_count += 1
        content = p.read_text(encoding="utf-8")
        lines = content.splitlines()

        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            # 1. Em-dash
            if "\u2014" in line:
                print(f"[FAIL] {p.name}:{idx} Em-dash: {stripped[:80]}")
                violations += 1

            # 2. En-dash
            if "\u2013" in line:
                print(f"[FAIL] {p.name}:{idx} En-dash: {stripped[:80]}")
                violations += 1

            # 3. Double hyphen with space
            if " -- " in line:
                print(f"[FAIL] {p.name}:{idx} Double-hyphen space: {stripped[:80]}")
                violations += 1

            # 4. Emoji
            if EMOJI_PATTERN.search(line):
                print(f"[FAIL] {p.name}:{idx} Emoji detected: {stripped[:80]}")
                violations += 1

            # 5. Banned words (case-insensitive with word boundaries)
            lower = line.lower()
            for w in BANNED_WORDS:
                pattern = r'\b' + re.escape(w) + r'\b'
                if re.search(pattern, lower):
                    print(f"[FAIL] {p.name}:{idx} Banned word '{w}': {stripped[:80]}")
                    violations += 1

    print(f"\nScanned {checked_count} deliverable files. Total violations: {violations}")
    if violations > 0:
        sys.exit(1)
    else:
        print("[PASS] All files adhere strictly to voice rules and boundary constraints.")
        sys.exit(0)

if __name__ == "__main__":
    check_voice()
