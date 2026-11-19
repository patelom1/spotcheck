import csv

import cv2
import numpy as np
import yaml

from spotcheck.data.crosssite import convert, time_of_day

XML = """<?xml version="1.0"?><parking id="x">
<space id="1" occupied="{o}"><contour><point x="4" y="4"/><point x="20" y="4"/><point x="20" y="30"/><point x="4" y="30"/></contour></space>
<space id="2" occupied="0"><contour><point x="30" y="4"/><point x="46" y="4"/><point x="46" y="30"/><point x="30" y="30"/></contour></space>
</parking>"""


def make_fake_pklot(root, lots=("PUCPR", "UFPR04", "UFPR05"), days=8, per_day=3):
    for lot in lots:
        for d in range(days):
            date = f"2012-10-{d + 10:02d}"
            weather = ["Sunny", "Cloudy", "Rainy"][d % 3]
            folder = root / lot / weather / date
            folder.mkdir(parents=True)
            for k in range(per_day):
                stem = f"{date}_{8 + 4 * k:02d}_05_00"
                cv2.imwrite(str(folder / f"{stem}.jpg"), np.full((40, 60, 3), 100, np.uint8))
                (folder / f"{stem}.xml").write_text(XML.format(o=k % 2))


def test_cross_site_split(tmp_path):
    make_fake_pklot(tmp_path / "PKLot")
    out = tmp_path / "xsite"
    convert(tmp_path / "PKLot", out, target="PUCPR", sizes=(2, 4, 999), max_per_session=10)
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    # target lot never appears in source splits, and vice versa
    for r in rows:
        assert (r["lot"] == "PUCPR") == r["split"].startswith("target")
    # capture days never straddle two splits
    day_split = {}
    for r in rows:
        assert day_split.setdefault(r["session"], r["split"]) == r["split"]
    # time metadata parsed from file names
    assert {r["time_of_day"] for r in rows} == {"morning", "midday", "afternoon"}
    # nested subsets, and N larger than the pool is skipped
    s2 = (out / "target_train_2.txt").read_text().split()
    s4 = (out / "target_train_4.txt").read_text().split()
    assert s2 == s4[:2] and not (out / "target_999.yaml").exists()
    comb = yaml.safe_load((out / "combined_4.yaml").read_text())
    assert len(comb["train"]) == 2


def test_time_of_day_bins():
    assert [time_of_day(h) for h in (7, 12, 15, 18)] == ["morning", "midday", "afternoon", "evening"]
