import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api import app
from spotcheck.counting import count_detections
from spotcheck.infer import draw


class FakeChecker:
    weights, imgsz = "fake.pt", 960

    def predict(self, img, conf=0.25):
        boxes = np.array([[10, 10, 50, 50], [60, 10, 100, 50], [110, 10, 150, 50]], float)
        cls, sc = np.array([0, 1, 0]), np.array([0.9, 0.8, 0.7])
        c = count_detections(boxes, cls, sc, conf=conf)
        return {"counts": c.as_dict(), "detections": [], "inference_ms": 1.0,
                "image_size": [img.shape[1], img.shape[0]], "_raw": (boxes, cls, sc)}

    def annotate(self, img, r):
        return draw(img, *r["_raw"])


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((120, 200, 3), 128, np.uint8))
    return ("lot.jpg", buf.tobytes(), "image/jpeg")


def client():
    app.state.checker = FakeChecker()
    return TestClient(app)


def test_predict_and_lots():
    with client() as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", files={"file": _jpeg()}).json()
        assert r["counts"] == {"empty": 2, "occupied": 1, "total": 3, "occupancy_rate": 0.3333}
        assert "_raw" not in r
        c.post("/lots/lot_j/snapshot", files={"file": _jpeg()})
        assert c.get("/lots").json()["total_open"] == 2
        img = c.post("/predict/annotated", files={"file": _jpeg()})
        assert img.headers["content-type"] == "image/jpeg"
        assert "SpotCheck" in c.get("/").text


def test_rejects_non_image():
    with client() as c:
        r = c.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r.status_code == 415
