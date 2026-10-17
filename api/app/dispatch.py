"""Hand work to the Modal GPU worker (looked up by app/class name; auth via MODAL_TOKEN_* env)."""
import modal

from .config import settings

_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = modal.Cls.from_name(settings.MODAL_APP, settings.MODAL_CLASS)()
    return _pipeline


def start_identify(job_id: str, images: list[dict], uid: str) -> str:
    return pipeline().run.spawn(job_id, images, uid).object_id


def start_reindex() -> str:
    return pipeline().reindex.spawn().object_id
