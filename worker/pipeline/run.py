"""End-to-end pipeline for one submission.

detect -> quality gate -> species -> (green only) segment -> embed -> search -> re-rank -> decide -> save
"""
from collections import defaultdict

from . import config, quality, storage, store
from .matcher import inliers_to_confidence


def process(m, job_id: str, images: list[dict], uid: str | None = None) -> dict:
    """m: loaded models (detector, species, segmenter, embedder, matcher)."""
    per_image, usable = [], []  # usable: (idx, view, det_crop)
    for idx, src in enumerate(images):
        view = src.get("view") or "other"
        rec = {"index": idx, "slot": src.get("slot"), "view": view, "usable": False, "reason": None, "detection_confidence": 0.0}
        try:
            img = storage.load_url(src["url"])
        except Exception:
            rec["reason"] = "download_failed"
            per_image.append(rec)
            continue
        crop, _, conf = m.detector.detect(img)
        ok, reason = quality.check(crop, conf)
        rec.update(usable=ok, reason=reason, detection_confidence=round(conf, 4))
        per_image.append(rec)
        if ok:
            usable.append((idx, view, crop))

    # ---- species ----
    if len(usable) < config.MIN_USABLE_IMAGES:
        species = {"label": config.DATA_DEFICIENT, "confidence": 0.0, "scores": {}}
    else:
        label, conf, scores = m.species.aggregate([c for _, _, c in usable])
        species = {"label": label, "confidence": round(conf, 4), "scores": scores}

    ident = {"status": "skipped", "turtle_id": None, "confidence": None, "is_provisional": False}

    conn = store.connect()
    try:
        if species["label"] == "green_turtle":
            ident = _reid(m, conn, job_id, usable, per_image, species)
        else:
            store.insert_sighting(conn, job_id, None, species["label"], species["confidence"],
                                  "skipped", None)
        result = {
            "job_id": job_id,
            "uid": uid,
            "species": species,
            "identification": ident,
            "images": per_image,
            "model_versions": {
                "detector": config.DETECTOR_VERSION,
                "species": config.SPECIES_MODEL_VERSION,
                "segmenter": config.SEG_MODEL_VERSION,
                "embedder": config.EMBED_MODEL_VERSION,
            },
        }
        conn.commit()
        store.set_job(conn, job_id, "completed", result=result)
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _reid(m, conn, job_id, usable, per_image, species):
    queries = []  # dicts: idx, side, det_key, scute_key, vec, feats
    for idx, view, crop in usable:
        if view not in config.REID_VIEWS:
            continue
        scute = m.segmenter.scute_crop(crop)
        if scute is None:
            per_image[idx]["reason"] = "scutes_not_found"
            continue
        base = f"{config.CROPS_PREFIX}/{job_id}/{idx}_{view}"
        queries.append({
            "idx": idx, "side": view,
            "det_key": storage.save_jpeg(crop, f"{base}_det.jpg"),
            "scute_key": storage.save_jpeg(scute, f"{base}_scute.jpg"),
            "vec": m.embedder.embed(scute),
            "feats": m.matcher.features(scute),
        })

    if not queries:
        store.insert_sighting(conn, job_id, None, species["label"], species["confidence"],
                              "insufficient_views", None)
        return {"status": "insufficient_views", "turtle_id": None, "confidence": None,
                "is_provisional": False}

    # ---- retrieve + re-rank: best inliers per (turtle, side) ----
    best = defaultdict(dict)  # turtle_id -> {side: inliers}
    feat_cache = {}
    for q in queries:
        for turtle_id, key, _sim in store.search(conn, q["vec"], q["side"], config.TOP_K):
            if key not in feat_cache:
                feat_cache[key] = m.matcher.features(storage.load_key(key))
            n = m.matcher.inliers(q["feats"], feat_cache[key])
            best[turtle_id][q["side"]] = max(best[turtle_id].get(q["side"], 0), n)

    ranked = sorted(((max(s.values()), t, s) for t, s in best.items()), reverse=True)
    top = ranked[0][0] if ranked else 0
    second = ranked[1][0] if len(ranked) > 1 else 0

    # sides pointing at different confident turtles -> conflict
    side_winners = {}
    for t, sides in best.items():
        for side, n in sides.items():
            if n >= config.MATCH_INLIERS and n > side_winners.get(side, (0, None))[0]:
                side_winners[side] = (n, t)
    conflict = len({t for _, t in side_winners.values()}) > 1

    if ranked and top >= config.MATCH_INLIERS and top - second >= config.MARGIN_INLIERS and not conflict:
        status, turtle_id, provisional = "matched", ranked[0][1], False
        conf = inliers_to_confidence(top)
    elif top < config.NEW_INLIERS:
        status, provisional = "new", False
        turtle_id = store.create_turtle(conn, "green_turtle", "confirmed")
        conf = round(1 - inliers_to_confidence(top), 4)
    else:
        status, provisional = "uncertain", True
        turtle_id = store.create_turtle(conn, "green_turtle", "provisional")
        conf = inliers_to_confidence(top)

    sighting_id = store.insert_sighting(conn, job_id, turtle_id, species["label"],
                                        species["confidence"], status, conf)
    for q in queries:
        store.insert_embedding(conn, turtle_id, sighting_id, q["side"], q["det_key"],
                               q["scute_key"], q["vec"])
    if status == "uncertain":
        candidates = [{"turtle_id": t, "inliers": n, "sides": s} for n, t, s in ranked[:5]]
        store.insert_review(conn, sighting_id, turtle_id, candidates)

    return {"status": status, "turtle_id": turtle_id, "confidence": conf,
            "is_provisional": provisional}
