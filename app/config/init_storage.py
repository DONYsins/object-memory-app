from pathlib import Path
from app.config.settings import Settings

def bootstrap_storage(settings: Settings) -> None:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.images_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_path).parent.mkdir(parents=True, exist_ok=True)
