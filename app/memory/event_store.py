import os
import cv2
from datetime import datetime

class EventStore:
    def __init__(self, settings, conn, index):
        self.s = settings
        self.conn = conn
        self.index = index
        os.makedirs(self.s.images_dir, exist_ok=True)

    def infer_location_simple(self, frame_shape, bbox):
        # learner note: coarse location from bbox position
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if cy > h * 0.75:
            return "on the floor"
        if cy < h * 0.35:
            return "on an elevated surface"
        if w * 0.3 < cx < w * 0.7:
            return "near the center"
        return "near the side"

    def should_store(self, obj, emb_vec, now_dt):
        # learner note: avoid saving duplicates too often
        scores, ids = self.index.search(emb_vec, k=min(20, self.index.index.ntotal))
        if not ids:
            return True

        # find nearest same-object event in results
        from .db import fetch_events_by_ids
        rows = fetch_events_by_ids(self.conn, ids)

        # align score->id (simple approach)
        score_map = dict(zip(ids, scores))

        for row in rows:
            event_id, row_obj, time_iso, *_ = row
            if row_obj != obj:
                continue
            past_dt = datetime.fromisoformat(time_iso)
            time_diff = (now_dt - past_dt).total_seconds()
            sim = score_map.get(event_id, 0.0)

            if sim >= self.s.similarity_threshold and time_diff <= self.s.time_gap_seconds:
                return False
            break

        return True

    def store_event(self, obj, frame, bbox, emb_vec):
        now_dt = datetime.now()
        now_iso = now_dt.isoformat(timespec="seconds")

        # store full frame for context
        frame_name = f"frame_{obj}_{now_dt.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        frame_path = os.path.join(self.s.images_dir, frame_name)
        cv2.imwrite(frame_path, frame)

        location = self.infer_location_simple(frame.shape, bbox)

        from .db import insert_event
        event_id = insert_event(
            self.conn, obj=obj, time_iso=now_iso, image_path=frame_path,
            location=location, bbox=bbox
        )

        self.index.add(emb_vec, event_id)
        return event_id, now_iso, frame_path, location
