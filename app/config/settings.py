from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    # Model files
    yolo_weights: str = os.getenv("YOLO_WEIGHTS", "best.pt")
    clip_model_name: str = os.getenv("CLIP_MODEL", "ViT-B/32")

    # Storage
    data_dir: str = os.getenv("DATA_DIR", "reid_store")
    images_dir: str = os.path.join(data_dir, "images")
    db_path: str = os.path.join(data_dir, "events.sqlite")
    faiss_path: str = os.path.join(data_dir, "index.faiss")

    # Memory / filtering
    embedding_dim: int = 512
    similarity_threshold: float = float(os.getenv("SIM_THRESHOLD", "0.78"))
    time_gap_seconds: int = int(os.getenv("TIME_GAP", "10"))

    # What to store
    allowed_objects: tuple = tuple(os.getenv("ALLOWED_OBJECTS", "MyWatch,MyWallet,MyBikeKeys").split(","))

settings = Settings()