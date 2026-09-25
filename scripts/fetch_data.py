"""Download the public benchmark datasets Sentinel ships worked examples for.

    python scripts/fetch_data.py            # both
    python scripts/fetch_data.py cmapss     # NASA C-MAPSS FD001 only
    python scripts/fetch_data.py ai4i       # UCI AI4I 2020 only

Stdlib only, so it runs anywhere. Files land in ./data (git-ignored).
"""
import io
import pathlib
import sys
import urllib.request
import zipfile

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "cmapss": {
        "url": "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
        "keep": ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"],
    },
    "ai4i": {
        "url": "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        "keep": ["ai4i2020.csv"],
    },
}


def fetch(name: str) -> None:
    spec = SOURCES[name]
    DATA.mkdir(exist_ok=True)
    if all((DATA / f).exists() for f in spec["keep"]):
        print(f"[{name}] already present")
        return
    print(f"[{name}] downloading {spec['url']}")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "sentinel-ml"})
    blob = urllib.request.urlopen(req, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            base = pathlib.PurePath(member).name
            if base in spec["keep"]:
                (DATA / base).write_bytes(z.read(member))
                print(f"  wrote data/{base}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(SOURCES)):
        fetch(n)
