"""BioCLIP 2.5 zero-shot species classification over the 7 sea turtle species."""
import numpy as np
import torch

from . import config


class SpeciesClassifier:
    def __init__(self, device: str):
        import open_clip

        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(config.BIOCLIP_MODEL)
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        tokenizer = open_clip.get_tokenizer(config.BIOCLIP_MODEL)

        self.labels = list(config.SPECIES.keys())
        prompts = [f"a photo of {config.SPECIES[l]}." for l in self.labels]
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
            t = self.model.encode_text(tokenizer(prompts).to(device)).float()
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.scale = float(self.model.logit_scale.exp())

    @torch.no_grad()
    def probs(self, crops) -> np.ndarray:
        """crops: list of PIL images -> [n, 7] probabilities."""
        batch = torch.stack([self.preprocess(c) for c in crops]).to(self.device)
        with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
            f = self.model.encode_image(batch).float()
        f = f / f.norm(dim=-1, keepdim=True)
        return (self.scale * f @ self.text.T).softmax(dim=-1).cpu().numpy()

    def aggregate(self, crops):
        """Average over all usable crops -> (label, confidence, scores dict)."""
        p = self.probs(crops).mean(axis=0)
        scores = {l: round(float(s), 4) for l, s in zip(self.labels, p)}
        best = int(p.argmax())
        label, conf = self.labels[best], float(p[best])
        if conf < config.SPECIES_MIN_CONF:
            label = config.UNKNOWN
        return label, conf, scores
