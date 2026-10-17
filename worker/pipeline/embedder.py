"""DINOv2 ViT-B/14 embedder (global vector for retrieval).

Without a fine-tuned checkpoint it uses plain DINOv2 (works as a baseline).
After you fine-tune with ArcFace, save:
    torch.save({"backbone": model.state_dict(),
                "head": proj.state_dict() or None}, "embedder.pt")
where `proj` is nn.Linear(768, EMBED_DIM). Then bump EMBED_MODEL_VERSION and re-index.
"""
import os

import numpy as np
import torch
from torchvision import transforms

from . import config


class Embedder:
    def __init__(self, device: str):
        from transformers import AutoModel

        self.device = device
        self.backbone = AutoModel.from_pretrained(config.DINOV2_MODEL)
        self.head = None
        if os.path.exists(config.EMBEDDER_WEIGHTS):
            ckpt = torch.load(config.EMBEDDER_WEIGHTS, map_location="cpu")
            self.backbone.load_state_dict(ckpt["backbone"])
            if ckpt.get("head"):
                self.head = torch.nn.Linear(self.backbone.config.hidden_size, config.EMBED_DIM)
                self.head.load_state_dict(ckpt["head"])
                self.head = self.head.to(device).eval()
        self.backbone = self.backbone.to(device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, img) -> np.ndarray:
        x = self.tf(img).unsqueeze(0).to(self.device)
        v = self.backbone(pixel_values=x).pooler_output
        if self.head is not None:
            v = self.head(v)
        v = torch.nn.functional.normalize(v, dim=-1)[0]
        out = v.float().cpu().numpy()
        assert out.shape[0] == config.EMBED_DIM, "EMBED_DIM must match the model output"
        return out
