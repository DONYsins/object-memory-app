from pydantic import BaseModel
from typing import Optional, List

class IngestResponse(BaseModel):
    stored_events: int
    detected: int

class QueryRequest(BaseModel):
    object_name: str
    k: int = 3

class EventOut(BaseModel):
    id: int
    object: str
    time_iso: str
    image_path: Optional[str] = None
    location: Optional[str] = None
    x1: int
    y1: int
    x2: int
    y2: int
    score: Optional[float] = None

class QueryResponse(BaseModel):
    object_name: str
    results: List[EventOut]
