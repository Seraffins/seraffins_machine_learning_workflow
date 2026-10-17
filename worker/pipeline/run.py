"""End-to-end pipeline for one submission.

detect -> quality gate -> species -> (green only) segment -> embed -> search -> re-rank -> decide -> save
"""
import json
from collections import defaultdict

from . import config, quality, storage, store


def _log(**kv):
    """One JSON line per step; shows up in `modal app logs turtle-pipeline`."""
    print(json.dumps(kv, default=str), flush=True)
from .matcher import inliers_to_confidence



def _reid(m, conn, job_id, usable, per_image, species):
    queries = []  # dicts: idx, side, det_key, scute_key, vec, feats
    for idx, view, crop in usable:
        # LOG 1: Check view against REID_VIEWS
        _log(step="reid_view_check", job_id=job_id, idx=idx, view=view,
             reid_views=config.REID_VIEWS, matches=(view in config.REID_VIEWS))
        
        if view not in config.REID_VIEWS:
            per_image[idx]["reason"] = "view_not_in_reid_views"
            continue
            
        scute = m.segmenter.scute_crop(crop)
        if scute is None:
            per_image[idx]["reason"] = "scutes_not_found"
            # LOG 2: Check scute crop failures
            _log(step="reid_scute_crop", job_id=job_id, idx=idx, view=view, found=False)
            continue
            
        base = f"{config.CROPS_PREFIX}/{job_id}/{idx}_{view}"
        queries.append({
            "idx": idx, "side": view,
            "det_key": storage.save_jpeg(crop, f"{base}_det.jpg"),
            "scute_key": storage.save_jpeg(scute, f"{base}_scute.jpg"),
            "vec": m.embedder.embed(scute),
            "feats": m.matcher.features(scute),
        })

    # LOG 3: Query list size summary
    _log(step="reid_queries_summary", job_id=job_id, total_usable=len(usable), total_queries=len(queries))

    if not queries:
        store.insert_sighting(conn, job_id, None, species["label"], species["confidence"],
                              "insufficient_views", None)
        return {"status": "insufficient_views", "turtle_id": None, "confidence": None,
                "is_provisional": False}

    # ---- retrieve + re-rank ----
    best = defaultdict(dict)
    feat_cache = {}
    for q in queries:
        search_results = store.search(conn, q["vec"], q["side"], config.TOP_K)
        # LOG 4: Vector search matches found in database
        _log(step="reid_vector_search", job_id=job_id, side=q["side"], matches_found=len(search_results))
        
        for turtle_id, key, _sim in search_results:
            if key not in feat_cache:
                feat_cache[key] = m.matcher.features(storage.load_key(key))
            n = m.matcher.inliers(q["feats"], feat_cache[key])
            best[turtle_id][q["side"]] = max(best[turtle_id].get(q["side"], 0), n)

    ranked = sorted(((max(s.values()), t, s) for t, s in best.items()), reverse=True)
    top = ranked[0][0] if ranked else 0
    second = ranked[1][0] if len(ranked) > 1 else 0

    # LOG 5: Matcher score thresholds check
    _log(step="reid_scoring", job_id=job_id, top_inliers=top, second_inliers=second,
         match_threshold=config.MATCH_INLIERS, new_threshold=config.NEW_INLIERS)

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

    # LOG 6: Final turtle registration output
    _log(step="reid_outcome", job_id=job_id, status=status, turtle_id=turtle_id, conf=conf)

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