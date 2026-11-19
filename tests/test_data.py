from spotcheck.common import label_path_for, xyxy_to_yolo
from spotcheck.data.campus import original_stem
from spotcheck.data.pklot import parse_pklot_xml, split_sessions

PKLOT_XML = """<?xml version="1.0"?>
<parking id="ufpr04">
  <space id="1" occupied="0">
    <rotatedRect><center x="50" y="40"/><size w="20" h="30"/><angle d="0"/></rotatedRect>
    <contour><point x="40" y="25"/><point x="60" y="25"/><point x="60" y="55"/><point x="40" y="55"/></contour>
  </space>
  <space id="2" occupied="1">
    <rotatedRect><center x="100" y="40"/><size w="20" h="30"/><angle d="0"/></rotatedRect>
    <contour><point x="90" y="25"/><point x="110" y="25"/><point x="110" y="55"/><point x="90" y="55"/></contour>
  </space>
  <space id="3">
    <rotatedRect><center x="150" y="40"/><size w="20" h="30"/><angle d="0"/></rotatedRect>
  </space>
</parking>"""


def test_parse_pklot_xml(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(PKLOT_XML)
    boxes = parse_pklot_xml(p)
    assert boxes == [(0, 40, 25, 60, 55), (1, 90, 25, 110, 55)]  # unlabeled space 3 skipped


def test_split_is_by_session_and_deterministic():
    s = [f"day{i}" for i in range(20)]
    a, b = split_sessions(s, seed=0), split_sessions(s, seed=0)
    assert a == b and set(a.values()) == {"train", "val", "test"}


def test_yolo_conversion_clips():
    cx, cy, w, h = xyxy_to_yolo(-10, 0, 50, 100, 100, 100)
    assert (cx, cy, w, h) == (0.25, 0.5, 0.5, 1.0)


def test_roboflow_names():
    assert original_stem("IMG_0412_jpg.rf.3f9a2b.jpg") == "IMG_0412"
    assert original_stem("IMG_0412.jpg") == "IMG_0412"


def test_label_path():
    assert str(label_path_for("d/images/train/x.jpg")).endswith("d/labels/train/x.txt")
