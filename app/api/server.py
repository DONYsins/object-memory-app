import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from datetime import datetime

from app.config.settings import settings
from app.config.init_storage import bootstrap_storage
from app.vision.detector import YoloDetector
from app.vision.embedder import ClipEmbedder
from app.memory.db import init_db, fetch_last_k, fetch_events_by_ids
from app.memory.vector_index import FaissIndex
from app.memory.event_store import EventStore
from app.models.schemas import IngestResponse, QueryRequest, QueryResponse, EventOut

app = FastAPI(title="Item Memory Assistant API")

detector = YoloDetector(settings.yolo_weights)
embedder = ClipEmbedder(settings.clip_model_name)

bootstrap_storage(settings)
conn = init_db(settings.db_path)
index = FaissIndex(settings.faiss_path, settings.embedding_dim)
store = EventStore(settings, conn, index)

@app.post("/ingest_frame", response_model=IngestResponse)
async def ingest_frame(file: UploadFile = File(...)):
    # learner note: backend receives a JPEG frame from client
    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    results = detector.detect(frame)
    boxes = results[0].boxes if results else []
    detected_count = 0
    stored_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        label = detector.model.names[cls_id]

        if settings.allowed_objects and label not in settings.allowed_objects:
            continue

        detected_count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bbox = (x1, y1, x2, y2)

        emb = embedder.get_embedding(frame, bbox)
        if emb is None:
            continue

        now_dt = datetime.now()
        if store.should_store(label, emb, now_dt):
            store.store_event(label, frame, bbox, emb)
            stored_count += 1

    return IngestResponse(stored_events=stored_count, detected=detected_count)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # learner note: simplest query = last K events (fast + stable)
    rows = fetch_last_k(conn, req.object_name, k=req.k)
    results = []
    for r in rows:
        _id, obj, time_iso, image_path, location, x1, y1, x2, y2 = r
        results.append(EventOut(
            id=_id, object=obj, time_iso=time_iso,
            image_path=image_path, location=location,
            x1=x1, y1=y1, x2=x2, y2=y2
        ))
    return QueryResponse(object_name=req.object_name, results=results)

@app.on_event("shutdown")
def shutdown():
    index.save()
    conn.close()
