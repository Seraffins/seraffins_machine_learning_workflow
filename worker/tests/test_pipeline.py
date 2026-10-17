"""Pipeline decision logic against a real Postgres, with stand-in models (no GPU needed).

    TEST_DATABASE_URL=postgresql://postgres:pg@localhost:5433/postgres pytest worker/tests
Schema is built with the API's Alembic migrations.
"""
import os
import sys
import types
import uuid

import numpy as np
import pytest
from PIL import Image

URL = os.environ.get("TEST_DATABASE_URL")
if not URL:
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)

os.environ["DATABASE_URL"] = URL
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.modules.setdefault("torch", types.SimpleNamespace(no_grad=lambda: (lambda f: f)))

from pipeline import run, storage, store  # noqa: E402

INLIERS = {frozenset(["A"]): 100, frozenset(["B"]): 100, frozenset(["A", "C"]): 40}
SAVED = {}


def _img(ident):
    rng = np.random.default_rng(abs(hash(ident)) % 2**32)
    im = Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))
    im.info["ident"] = ident
    return im


class Models:
    species_label = "green_turtle"

    class detector:
        @staticmethod
        def detect(img):
            return (None, None, 0.0) if img.size == (10, 10) else (img, (0, 0, 1, 1), 0.95)

    class species:
        @staticmethod
        def aggregate(crops):
            return Models.species_label, 0.9, {Models.species_label: 0.9}

    class segmenter:
        @staticmethod
        def scute_crop(c):
            out = c.copy()
            out.info["ident"] = c.info["ident"]
            return out

    class embedder:
        @staticmethod
        def embed(img):
            base = {"C": "A"}.get(img.info["ident"], img.info["ident"])  # C looks like A
            v = np.random.default_rng(ord(base[0])).normal(size=768)
            return v / np.linalg.norm(v)

    class matcher:
        @staticmethod
        def features(img):
            return img.info.get("ident")

        @staticmethod
        def inliers(a, b):
            return INLIERS.get(frozenset([a, b]), 5)


@pytest.fixture(scope="module", autouse=True)
def setup(monkeypatch_module):
    from alembic import command
    from alembic.config import Config
    os.environ["MIGRATION_DATABASE_URL"] = URL
    cfg = Config(os.path.join(ROOT, "..", "api", "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT, "..", "api", "migrations"))
    sys.path.insert(0, os.path.join(ROOT, "..", "api"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    monkeypatch_module.setattr(storage, "load_url",
                               lambda url: Image.new("RGB", (10, 10)) if "nodet" in url
                               else _img(url.rsplit("/", 1)[1].split("_")[0]))
    monkeypatch_module.setattr(storage, "save_jpeg", lambda img, key: SAVED.setdefault(key, img) and key)
    monkeypatch_module.setattr(storage, "load_key", lambda k: SAVED[k])


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def submit(ident, views=("whole", "left_face", "right_face")):
    c = store.connect()
    jid = str(uuid.uuid4())
    c.execute("insert into jobs (id, request, user_id) values (%s, '{}', 'u1')", (jid,))
    c.commit()
    c.close()
    imgs = [{"url": f"https://x/{ident}_{i}.jpg", "view": v, "slot": v} for i, v in enumerate(views)]
    r = run.process(Models, jid, imgs, "u1")
    return r["species"]["label"], r["identification"]


def test_full_decision_flow():
    sp, a1 = submit("A")
    assert (sp, a1["status"], a1["turtle_id"]) == ("green_turtle", "new", "GT-000001")
    assert submit("A")[1]["status"] == "matched"                       # same turtle again
    assert submit("A")[1]["turtle_id"] == "GT-000001"
    assert submit("B")[1] == {"status": "new", "turtle_id": "GT-000002",
                              "confidence": submit_conf_new(), "is_provisional": False}
    c = submit("C")[1]                                                  # ambiguous
    assert c["status"] == "uncertain" and c["is_provisional"]

    Models.species_label = "hawksbill"
    assert submit("D") == ("hawksbill", {"status": "skipped", "turtle_id": None,
                                         "confidence": None, "is_provisional": False})
    Models.species_label = "green_turtle"
    assert submit("E", views=("whole", "carapace"))[1]["status"] == "insufficient_views"
    assert submit("nodet")[0] == "data_deficient"

    conn = store.connect()
    assert conn.execute("select count(*) from review_queue").fetchone()[0] == 1
    assert conn.execute("select count(*) from jobs where status='completed'").fetchone()[0] == 8


def submit_conf_new():
    from pipeline.matcher import inliers_to_confidence
    return round(1 - inliers_to_confidence(5), 4)
