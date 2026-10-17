"""Modal GPU worker: all models in one container, loaded once per cold start.

Deploy:  cd worker && modal deploy modal_app.py
"""
import os

import modal

GPU = os.environ.get("TURTLE_GPU", "T4")  # T4 is cheapest; "L4" is faster

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.6", "torchvision",
        "rfdetr", "open_clip_torch", "transformers", "ultralytics>=8.4",
        "opencv-python-headless", "numpy<2.2", "pillow",
        "psycopg[binary]", "pgvector", "boto3", "httpx",
    )
    .pip_install("git+https://github.com/cvg/LightGlue.git")
    .env({"HF_HOME": "/cache/hf", "TORCH_HOME": "/cache/torch"})
    .add_local_python_source("pipeline")
)

app = modal.App("turtle-pipeline")
models_vol = modal.Volume.from_name("turtle-taging-models", create_if_missing=True)
cache_vol = modal.Volume.from_name("turtle-hf-cache", create_if_missing=True)


@app.cls(
    gpu=GPU,
    image=image,
    volumes={"/models": models_vol, "/cache": cache_vol},
    secrets=[modal.Secret.from_name("turtle-secrets")],
    scaledown_window=120,   # keep warm 2 min after last job, then scale to zero
    timeout=900,
)
class Pipeline:
    @modal.enter()
    def load(self):
        from pipeline import config
        from pipeline.detector import Detector
        from pipeline.embedder import Embedder
        from pipeline.matcher import Matcher
        from pipeline.segmenter import Segmenter
        from pipeline.species import SpeciesClassifier

        device = "cuda"
        self.detector = Detector(config.DETECTOR_WEIGHTS)
        self.species = SpeciesClassifier(device)
        self.segmenter = Segmenter(config.SEGMENTER_WEIGHTS)
        self.embedder = Embedder(device)
        self.matcher = Matcher(device)
        cache_vol.commit()  # persist downloaded HF weights for faster next cold start

    @modal.method()
    def run(self, job_id: str, images: list[dict], uid: str | None = None) -> dict:
        from pipeline import notify, run, store

        conn = store.connect()
        store.set_job(conn, job_id, "running")
        conn.close()
        try:
            result = run.process(self, job_id, images, uid)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            conn = store.connect()
            store.set_job(conn, job_id, "failed", error=err)
            conn.close()
            notify.send({"job_id": job_id, "uid": uid, "status": "failed",
                         "result": None, "error": "processing failed"})
            raise
        notify.send({"job_id": job_id, "uid": uid, "status": "completed",
                     "result": result, "error": None})
        return result

    @modal.method()
    def reindex(self, batch_commit: int = 50) -> dict:
        """Re-segment + re-embed the gallery for the CURRENT model versions.
        Old vectors stay until you delete them; search only uses current versions."""
        from pipeline import config, storage, store

        conn = store.connect()
        rows = conn.execute(
            """
            select distinct on (e.det_crop_key) e.turtle_id, e.sighting_id, e.side, e.det_crop_key
            from embeddings e
            where not exists (
                select 1 from embeddings n
                where n.det_crop_key = e.det_crop_key
                  and n.embed_model_version = %s and n.seg_model_version = %s)
            """,
            (config.EMBED_MODEL_VERSION, config.SEG_MODEL_VERSION),
        ).fetchall()
        done = skipped = 0
        for turtle_id, sighting_id, side, det_key in rows:
            scute = self.segmenter.scute_crop(storage.load_key(det_key))
            if scute is None:
                skipped += 1
                continue
            scute_key = det_key.replace("_det.jpg", f"_scute_{config.SEG_MODEL_VERSION}.jpg")
            storage.save_jpeg(scute, scute_key)
            store.insert_embedding(conn, turtle_id, sighting_id, side, det_key, scute_key,
                                   self.embedder.embed(scute))
            done += 1
            if done % batch_commit == 0:
                conn.commit()
        conn.commit()
        conn.close()
        return {"reindexed": done, "skipped": skipped, "total": len(rows)}
