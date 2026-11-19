"""SpotCheck REST API + lot availability dashboard.  Owner: Kevin

    SPOTCHECK_WEIGHTS=runs/finetune/weights/best.pt uvicorn app.api:app --reload
    open http://localhost:8000          (dashboard)   http://localhost:8000/docs  (Swagger)

Endpoints
  GET  /health                     model + version info
  POST /predict                    image -> counts + boxes (JSON)
  POST /predict/annotated          image -> annotated JPEG
  POST /lots/{lot_id}/snapshot     image from a lot camera/phone -> updates that lot's status
  GET  /lots                       latest availability for every lot (feeds the dashboard)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from spotcheck.infer import SpotChecker, public

MAX_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "checker", None) is None:  # tests inject a fake checker
        weights = os.environ.get("SPOTCHECK_WEIGHTS", "runs/finetune/weights/best.pt")
        app.state.checker = SpotChecker(weights)
    app.state.lots = {}
    yield


app = FastAPI(title="SpotCheck API", version="1.0",
              description="Parking space occupancy detection (YOLOv8).",
              lifespan=lifespan)


async def read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large (15 MB max)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "could not decode image (use JPEG or PNG)")
    return img


@app.get("/health")
def health(request: Request):
    c = request.app.state.checker
    return {"status": "ok", "weights": str(getattr(c, "weights", "")), "imgsz": getattr(c, "imgsz", None)}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), conf: float = Query(0.25, ge=0.01, le=0.99)):
    img = await read_image(file)
    return public(request.app.state.checker.predict(img, conf=conf))


@app.post("/predict/annotated", response_class=Response)
async def predict_annotated(request: Request, file: UploadFile = File(...),
                            conf: float = Query(0.25, ge=0.01, le=0.99)):
    img = await read_image(file)
    c = request.app.state.checker
    vis = c.annotate(img, c.predict(img, conf=conf))
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.post("/lots/{lot_id}/snapshot")
async def lot_snapshot(request: Request, lot_id: str, file: UploadFile = File(...)):
    img = await read_image(file)
    r = public(request.app.state.checker.predict(img))
    entry = {"lot_id": lot_id, **r["counts"], "updated": datetime.now(timezone.utc).isoformat()}
    request.app.state.lots[lot_id] = entry
    return entry


@app.get("/lots")
def lots(request: Request):
    items = sorted(request.app.state.lots.values(), key=lambda e: -e["empty"])
    return {"lots": items, "total_open": sum(e["empty"] for e in items)}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SpotCheck · parking availability</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e9ecf1;--mut:#8b93a3;--ok:#3ccf6e;--bad:#e5484d;--mid:#f5a524}
*{box-sizing:border-box}body{margin:0;font:15px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:28px 32px 8px}h1{margin:0;font-size:26px}p.sub{margin:4px 0 0;color:var(--mut)}
#total{font-size:44px;font-weight:700;color:var(--ok)}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px;padding:20px 32px}
.card{background:var(--card);border-radius:14px;padding:18px}.lot{font-weight:600;font-size:17px}
.big{font-size:34px;font-weight:700;margin:6px 0}.bar{height:8px;background:#2a2f3a;border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%}.meta{color:var(--mut);font-size:12px;margin-top:8px}
form{padding:0 32px 32px;color:var(--mut)}input,button{font:inherit;margin-right:8px}
button{background:#3b5bdb;color:#fff;border:0;border-radius:8px;padding:7px 14px;cursor:pointer}
</style></head><body>
<header><h1>SpotCheck — parking availability</h1><p class="sub">Open spaces detected by YOLOv8 from the latest lot snapshots</p>
<div><span id="total">–</span> <span class="sub">open spaces across lots</span></div></header>
<main id="grid"></main>
<form id="f">Upload a snapshot: <input id="lot" placeholder="lot id, e.g. lot_j" required>
<input type="file" id="img" accept="image/*" required><button>Update</button></form>
<script>
async function load(){const r=await fetch('/lots');const d=await r.json();
document.getElementById('total').textContent=d.total_open;
document.getElementById('grid').innerHTML=d.lots.map(l=>{const occ=l.occupancy_rate;
const col=occ>.9?'var(--bad)':occ>.7?'var(--mid)':'var(--ok)';
return `<div class="card"><div class="lot">${l.lot_id}</div><div class="big" style="color:${col}">${l.empty}
<span style="font-size:15px;color:var(--mut)">/ ${l.total} open</span></div>
<div class="bar"><i style="width:${(occ*100).toFixed(0)}%;background:${col}"></i></div>
<div class="meta">${(occ*100).toFixed(0)}% full · updated ${new Date(l.updated).toLocaleTimeString()}</div></div>`}).join('')
||'<p class="sub">No snapshots yet — upload one below.</p>'}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const fd=new FormData();
fd.append('file',document.getElementById('img').files[0]);
await fetch('/lots/'+encodeURIComponent(document.getElementById('lot').value)+'/snapshot',{method:'POST',body:fd});load()};
load();setInterval(load,5000);
</script></body></html>"""
