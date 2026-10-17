# Sea Turtle ID Backend

Identifies sea turtle species from photos, and individual green turtles from their facial scutes.

```
Frontend backend ──X-API-Key──► API (FastAPI · Cloud Run) ──spawn──► Modal GPU worker
        ▲                              │ SQLAlchemy                    │ RF-DETR → quality gate
        │                              ▼                               │ → BioCLIP 2.5 species
        └──── signed webhook ◄── Supabase Postgres + pgvector ◄────────┘ → YOLO26s-seg scutes
                                 (schema managed by Alembic)             → DINOv2 → pgvector top-10
                                                                         → ALIKED+LightGlue → decide
Photos: frontend's Cloudflare R2 (downloaded by URL) · Our crops: Supabase Storage
```

| Doc | For |
|---|---|
| [DEPLOY.md](DEPLOY.md) | Setting up DB → Modal → Cloud Run → secure hand-off, step by step |
| [API.md](API.md) | The frontend team: endpoints, payloads, errors, webhook |

## Layout

```
api/                      FastAPI service (Cloud Run)
  app/models.py           SQLAlchemy models = schema source of truth
  app/routers/            public (/v1) and admin (/v1/admin) endpoints
  app/security.py         API keys, rate limiting, hardening middleware
  migrations/             Alembic (versions/ = schema history)
  tests/                  pytest, runs against real Postgres
worker/                   Modal GPU worker
  modal_app.py            container + model loading
  pipeline/               detector, quality, species, segmenter, embedder, matcher, store, run
  tests/                  pipeline decision tests (stand-in models, real Postgres)
.github/workflows/ci.yml  tests + migration drift check on every push
```

## Species labels

`green_turtle`, `hawksbill`, `loggerhead`, `olive_ridley`, `leatherback`, `kemps_ridley`, `flatback`,
`data_deficient` (fewer than 2 usable photos), `unknown` (species confidence < 0.5).
Individual re-ID runs for `green_turtle` only, from the `left`/`right` head close-ups.

## Before real users: calibrate thresholds

Defaults live in `worker/pipeline/config.py` and can be overridden through the Modal secret.
Run pairs of photos where you know the answer (same turtle / different turtle) through the matcher:

- `MATCH_INLIERS`: lowest inlier count seen for true same-turtle pairs (be conservative)
- `NEW_INLIERS`: highest count seen for different-turtle pairs
- `CONF_MID` / `CONF_SCALE`: shape the confidence curve between them
- `BLUR_MIN_VAR`, `DET_MIN_CONF`, `SPECIES_MIN_CONF`: check against real field photos

## Changing a model

| Changed | Do |
|---|---|
| Detector / BioCLIP | Upload weights, bump `DETECTOR_VERSION` / `SPECIES_MODEL_VERSION`, `modal deploy` |
| Segmenter / embedder | Upload weights, bump `SEG_MODEL_VERSION` / `EMBED_MODEL_VERSION`, `modal deploy`, then `POST /v1/admin/reindex` right away (search only uses vectors of the current versions) |

## Notes

- Segmenter: YOLO26s-seg via ultralytics (AGPL-3.0). Check licensing before commercial closed-source use.
- Two simultaneous submissions of the same *new* turtle can both create IDs; fix with the review queue / merge endpoint.
- Rate limits are per Cloud Run instance (in-memory), enough for abuse protection at this scale.
