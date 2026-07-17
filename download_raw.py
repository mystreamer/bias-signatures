"""Populate data/raw/ with the third-party source datasets (ADR 0005).

Everything this script fetches is gitignored; the committed manual inputs
(MFQ item lists, author annotations) live in data/raw/manual/.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"

SC_URL = (
    "https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/"
    "social-chem-101.zip"
)
SC_TSV = RAW_DIR / "social-chem-101" / "social-chem-101.v1.0.tsv"

IB_FILES = {
    # config, split -> output file
    ("issues", "issues"): RAW_DIR / "issuebench_issues.csv",
    ("templates", "templates_sample"): RAW_DIR / "issuebench_templates.csv",
}


def download_social_chemistry() -> None:
    if SC_TSV.is_file():
        print(f"already present: {SC_TSV.name}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "social-chem-101.zip"
    print("downloading Social Chemistry 101 ...")
    urllib.request.urlretrieve(SC_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    print(f"extracted {SC_TSV.name}")


def download_issuebench() -> None:
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for (config, split), out in IB_FILES.items():
        if out.is_file():
            print(f"already present: {out.name}")
            continue
        print(f"downloading IssueBench {config}/{split} ...")
        load_dataset("Paul/IssueBench", config, split=split).to_pandas().to_csv(out, index=False)
        print(f"wrote {out.name}")


if __name__ == "__main__":
    download_social_chemistry()
    download_issuebench()
