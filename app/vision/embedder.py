import torch
import clip
import cv2
from PIL import Image
import numpy as np

class ClipEmbedder:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def get_embedding(self, frame_bgr, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(img_t).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)

        # (512,) float32 for FAISS
        return emb.squeeze(0).cpu().numpy().astype("float32")
