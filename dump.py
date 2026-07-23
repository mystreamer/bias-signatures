"""Move all ephemeral pipeline outputs into a dated dump/ snapshot."""

import shutil
from datetime import date
from pathlib import Path

BASE = Path(__file__).parent
EPHEMERAL = ["data/components", "data/annotation_tasks", "data/template_sets",
             "data/responses", "data/signatures", "out"]

day = date.today().strftime("%Y%m%d")
dest = next(d for n in range(1, 100) if not (d := BASE / "dump" / f"{day}{n:02d}").exists())
dest.mkdir(parents=True)
for rel in EPHEMERAL:
    src = BASE / rel
    if src.exists():
        shutil.move(src, dest / src.name)
print(dest)
