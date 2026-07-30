"""
UCMR 5 acquisition and reconnaissance.

Downloads the EPA Fifth Unregulated Contaminant Monitoring Rule (UCMR 5)
occurrence text files and prints a first look at the schema, so the cleaning
step can be written against the real column names instead of assumptions.

Data source:
    U.S. EPA, Occurrence Data from the Unregulated Contaminant Monitoring Rule.
    https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule

Usage:
    python src/acquire.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/ucmr5")

# EPA occurrence-data downloads: the national file plus the per-state split.
SOURCES = {
    "ucmr5_all": "https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data.zip",
    "ucmr5_by_state": "https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data-by-state.zip",
}


def download(name: str, url: str, dest_dir: Path) -> Path:
    """Stream a zip to disk. Skips the download if the file is already there."""
    dest = dest_dir / f"{name}.zip"
    if dest.exists():
        print(f"  already have {dest.name}")
        return dest

    print(f"  downloading {name} ...")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)

    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract(zip_path: Path, dest_dir: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
        return archive.namelist()


def find_master_file(dest_dir: Path) -> Path:
    """National file is the one with 'All' in its name; fall back to the largest."""
    text_files = sorted(dest_dir.glob("*.txt"))
    if not text_files:
        raise FileNotFoundError("No .txt files found after extraction.")
    for path in text_files:
        if re.search("all", path.name, re.IGNORECASE):
            return path
    return max(text_files, key=lambda p: p.stat().st_size)


def summarise(df: pd.DataFrame) -> None:
    print(f"\nrows x cols: {df.shape[0]:,} x {df.shape[1]}")

    print("\ncolumns:")
    for col in df.columns:
        print(f"  - {col}")

    # Inspect the fields the cleaning step will lean on, located by name match
    # rather than a hard-coded header we have not verified yet.
    for fragment in ("contam", "source", "sign", "mrl", "result", "state", "size"):
        for col in (c for c in df.columns if fragment in c.lower()):
            print(f"\n[{col}]  unique = {df[col].nunique()}")
            print(df[col].value_counts(dropna=False).head(15).to_string())

    print("\nfirst rows:")
    print(df.head(8).to_string())


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching UCMR 5 occurrence data")
    for name, url in SOURCES.items():
        zip_path = download(name, url, RAW_DIR)
        print(f"  extracted: {extract(zip_path, RAW_DIR)}")

    master = find_master_file(RAW_DIR)
    print(f"\nReading {master.name} (everything as text on the first pass)")

    # dtype=str on purpose. The result columns pair a '<' / '=' sign with a
    # number; letting pandas guess types would quietly mangle the censored
    # non-detects, which are exactly what our detection target is built from.
    df = pd.read_csv(
        master,
        sep="\t",
        dtype=str,
        encoding="latin-1",
        on_bad_lines="warn",
    )
    summarise(df)


if __name__ == "__main__":
    main()
