from pathlib import Path

def bootstrap_storage(settings: Settings) -> None:
    data_dir = Path(settings.data_dir)
    images_dir = Path(settings.images_dir)
    db_path = Path(settings.db_path)
    faiss_path = Path(settings.faiss_path)

    # Directories
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Parents for file-backed storage
    db_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
