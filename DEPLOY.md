# Deployment Guide

Four phases, in this order. Each one ends with a check; don't move on until it passes.

| Phase | What | Where |
|---|---|---|
| 1 | Database + crop storage | Supabase |
| 2 | Models (GPU worker) | Modal |
| 3 | Backend API | Google Cloud Run |
| 4 | Secure hand-off to the frontend | Cloud Run + docs |

## Phase 0: Tools on your laptop

- Python 3.11, Git
- `gcloud` CLI (Phase 3)
- The repo cloned or unzipped. All commands run from the repo root unless they say `cd`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r api/requirements.txt modal
```

---

## Phase 1: Database (Supabase + Alembic)

### 1.1 Create the project
supabase.com → **New project** → region **South Asia (Mumbai)** → save the database password.

### 1.2 Get the two connection strings
**Connect** (top bar) → copy both, replacing `[YOUR-PASSWORD]`:

| Name | Pooler | Port | Used by |
|---|---|---|---|
| `MIGRATION_DATABASE_URL` | **Session** pooler | 5432 | Alembic only |
| `DATABASE_URL` | **Transaction** pooler | 6543 | API + Modal worker |

If the password contains `@ # / : ?`, URL-encode those characters (`@` → `%40`).

### 1.3 Create the schema with Alembic
```bash
cd api
export MIGRATION_DATABASE_URL='postgresql://postgres.<ref>:<PASSWORD>@aws-0-ap-south-1.pooler.supabase.com:5432/postgres'
alembic upgrade head
alembic current          # expect: 0001 (head)
alembic check            # expect: No new upgrade operations detected.
```
✅ **Check:** Supabase → Table Editor shows `turtles`, `jobs`, `sightings`, `embeddings`, `review_queue` (+ `alembic_version`), all with RLS enabled.

pgvector, the turtle ID sequence, indexes and RLS are all created by the migration. Nothing is clicked by hand.

### 1.4 Crop storage bucket
1. **Storage → New bucket** → `turtle-crops` → Public **off**.
2. **Storage → Settings → S3 Connection**: copy **Endpoint** + **Region**.
3. Same page → **S3 Access Keys → New access key**: copy the ID and secret (shown once).

### 1.5 How schema changes work from now on
```bash
cd api
# 1. edit app/models.py
alembic revision --autogenerate -m "add turtle nickname"
# 2. open the new file in migrations/versions/ and review it
# 3. commit it; it is applied on the next deploy (Phase 3.6), or locally:
alembic upgrade head
```
Rules:
- Never edit a migration that has already run anywhere. Write a new one.
- Never change the schema in the Supabase SQL Editor. CI runs `alembic check` and fails on drift.
- Renamed a column? Also update the raw SQL in `worker/pipeline/store.py`.
- Roll back one step: `alembic downgrade -1`.

---

## Phase 2: Models on Modal

### 2.1 Log in
```bash
modal setup
```

### 2.2 Worker secret
```bash
modal secret create turtle-secrets \
  DATABASE_URL='<transaction pooler URL, port 6543>' \
  STORAGE_ENDPOINT='https://<ref>.supabase.co/storage/v1/s3' \
  STORAGE_REGION='ap-south-1' \
  STORAGE_BUCKET='turtle-crops' \
  STORAGE_ACCESS_KEY_ID='...' \
  STORAGE_SECRET_ACCESS_KEY='...'
```
Optional webhook: also add `WEBHOOK_URL=...` and `WEBHOOK_SECRET=...`. To change a value later, re-run with `--force`.

### 2.3 Upload weights
```bash
modal volume create turtle-models
modal volume put turtle-models ./weights/rfdetr.pth   /rfdetr.pth     # RF-DETR Large detector
modal volume put turtle-models ./weights/segmenter.pt /segmenter.pt   # YOLO26s-seg scute model
modal volume ls turtle-models
```
BioCLIP 2.5 and DINOv2 download automatically on first run.

### 2.4 Deploy
```bash
cd worker
modal deploy modal_app.py
```
✅ **Check:** Modal dashboard → Apps → `turtle-pipeline` is **deployed**. It costs nothing until a job arrives, and scales to zero 2 minutes after the last one.

### 2.5 Token for the backend
Modal dashboard → **Settings → API Tokens → New token** → keep `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` for Phase 3.

---

## Phase 3: Backend on Cloud Run

### 3.1 Project setup (once)
```bash
gcloud init                                    # pick/create a project, link billing
export PROJECT=$(gcloud config get-value project)
export REGION=asia-south1
gcloud config set run/region $REGION
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
gcloud artifacts repositories create turtle --repository-format=docker --location=$REGION
```
Then set a budget alert: Billing → Budgets & alerts → e.g. $5.

### 3.2 A dedicated runtime identity (least privilege)
```bash
gcloud iam service-accounts create turtle-api --display-name="Turtle API runtime"
export SA=turtle-api@$PROJECT.iam.gserviceaccount.com
```

### 3.3 Generate your API keys
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"   # -> API key for the frontend's backend
python -c "import secrets; print(secrets.token_urlsafe(48))"   # -> admin key (only you)
```

### 3.4 Store secrets in Secret Manager
```bash
add_secret () {  # name value
  printf '%s' "$2" | gcloud secrets create "$1" --data-file=- --replication-policy=automatic
  gcloud secrets add-iam-policy-binding "$1" --member="serviceAccount:$SA" \
    --role=roles/secretmanager.secretAccessor >/dev/null
}
add_secret database-url           '<transaction pooler URL, 6543>'
add_secret migration-database-url '<session pooler URL, 5432>'
add_secret api-keys               '<API key>'
add_secret admin-keys             '<admin key>'
add_secret modal-token-id         '<MODAL_TOKEN_ID>'
add_secret modal-token-secret     '<MODAL_TOKEN_SECRET>'
```
Only the `turtle-api` service account can read them.

### 3.5 Build the image
```bash
export TAG=$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/turtle/api:$TAG
gcloud builds submit . --tag $IMAGE
```
If this fails with a permissions error, grant the Cloud Build service account
`roles/artifactregistry.writer` and retry.

### 3.6 Run migrations (a Cloud Run Job, same image)
```bash
gcloud run jobs deploy turtle-migrate --image $IMAGE \
  --command alembic --args upgrade,head \
  --service-account $SA \
  --set-secrets MIGRATION_DATABASE_URL=migration-database-url:latest \
  --max-retries 0 --task-timeout 300
gcloud run jobs execute turtle-migrate --wait
```
Running this when there's nothing new is harmless: Alembic only applies missing migrations.

### 3.7 Deploy the API
```bash
gcloud run deploy turtle-api --image $IMAGE \
  --service-account $SA \
  --allow-unauthenticated \
  --min-instances 0 --max-instances 3 --concurrency 40 \
  --cpu 1 --memory 512Mi --timeout 30 \
  --set-env-vars "^;^ENV=production;ENABLE_DOCS=false;ALLOWED_IMAGE_HOSTS=pub-xxxx.r2.dev;CORS_ORIGINS=" \
  --set-secrets "DATABASE_URL=database-url:latest,API_KEYS=api-keys:latest,ADMIN_KEYS=admin-keys:latest,MODAL_TOKEN_ID=modal-token-id:latest,MODAL_TOKEN_SECRET=modal-token-secret:latest"
```
`ENV=production` makes the app **refuse to start** if keys are missing or shorter than 32 characters,
no image hosts are set, or CORS is `*`. The deploy fails loudly instead of running insecurely.

`--allow-unauthenticated` only means Google doesn't check callers itself; every `/v1` route still
requires your key.

✅ **Check:**
```bash
export API=$(gcloud run services describe turtle-api --format='value(status.url)')
curl $API/health                    # {"ok":true}
curl $API/ready                     # {"ok":true}  (database reachable)
curl -s -o /dev/null -w "%{http_code}\n" $API/v1/species                        # 401
curl -H "X-API-Key: <API key>" $API/v1/species                                  # labels
curl -X POST $API/v1/identify -H "X-API-Key: <API key>" -H "Content-Type: application/json" \
  -d '{"uid":"test-user","images":{"left":"https://pub-xxxx.r2.dev/l.jpg","right":"https://pub-xxxx.r2.dev/r.jpg","whole":"https://pub-xxxx.r2.dev/w.jpg"}}'
curl -H "X-API-Key: <API key>" $API/v1/jobs/<job_id>
```
Watch the run with `modal app logs turtle-pipeline`. The very first job downloads BioCLIP and DINOv2, so it takes several minutes.

### 3.8 Every later release
```bash
export TAG=$(git rev-parse --short HEAD); export IMAGE=$REGION-docker.pkg.dev/$PROJECT/turtle/api:$TAG
gcloud builds submit . --tag $IMAGE
gcloud run jobs update turtle-migrate --image $IMAGE && gcloud run jobs execute turtle-migrate --wait
gcloud run deploy turtle-api --image $IMAGE          # keeps the env vars and secrets set above
```
Order matters: **migrate, then deploy**, so new code never runs against an old schema.
Rollback: `gcloud run services update-traffic turtle-api --to-revisions=<previous-revision>=100`.

Worker changes: `cd worker && modal deploy modal_app.py`.

---

## Phase 4: Secure hand-off to the frontend

### 4.1 Who calls the API
The API key is a **server secret**. It must live in the frontend team's **backend**, never in
browser JavaScript or a mobile app bundle, where anyone can extract it.

```
User's app ─► Frontend backend (holds X-API-Key) ─► Turtle API (Cloud Run) ─► Modal GPU
                         ▲                                         │
                         └──────────── webhook (signed) ◄──────────┘
```
`CORS_ORIGINS` is empty, so browsers cannot call the API directly. That's intentional.
If the frontend has **no backend** (calls straight from the app), stop and tell me which auth they
use (Firebase, Supabase Auth, …). The API should then verify the user's login token instead of a shared key.

### 4.2 What to give the frontend team
1. Base URL: `https://turtle-api-xxxx.a.run.app`
2. The API key, sent through a password manager or a one-time secret link. Never by chat or email in plain text.
3. `API.md` (request/response contract, error codes, webhook verification)
4. Webhook secret, if you enabled the webhook

### 4.3 Built-in protections
| Protection | How |
|---|---|
| Authentication | `X-API-Key` (constant-time compare), separate `X-Admin-Key` |
| Input validation | https only, allow-listed image hosts, known slots only, size limits |
| Abuse limits | 10 identify/min per uid, 300/min per key → `429` + `Retry-After` |
| Duplicate submits | `Idempotency-Key` header returns the original job |
| Request size | Bodies over 32 KB rejected (`413`) |
| Headers | HSTS, nosniff, frame-deny, no-referrer, no-store |
| Errors | Generic message + `request_id`; no stack traces leave the server |
| Secrets | Secret Manager, readable only by the runtime service account |
| Database | RLS blocks Supabase's public REST API; only the backend's postgres role has access |
| Docs | `/docs` disabled in production |

### 4.4 Rotating the API key (no downtime)
```bash
gcloud secrets versions add api-keys --data-file=- <<< 'OLD_KEY,NEW_KEY'
gcloud run deploy turtle-api --image $IMAGE        # picks up the new secret version
# frontend switches to NEW_KEY, then:
gcloud secrets versions add api-keys --data-file=- <<< 'NEW_KEY'
gcloud run deploy turtle-api --image $IMAGE
```

### 4.5 Monitoring
- Logs: Cloud Run → turtle-api → Logs. One JSON line per request with `request_id`, status, latency.
- Alert: Monitoring → Alerting → Cloud Run 5xx count > 0 over 5 min → email.
- Worker: Modal dashboard → turtle-pipeline → logs and failed calls.
- Review queue: `GET /v1/admin/review` with your admin key.

---

## Local development

```bash
docker run -d --name turtle-pg -p 5433:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16
cd api && cp ../.env.example .env      # set DATABASE_URL/MIGRATION_DATABASE_URL to the local DB
alembic upgrade head
uvicorn app.main:app --reload           # http://localhost:8000/docs

# tests (same as CI)
export TEST_DATABASE_URL=postgresql://postgres:pg@localhost:5433/postgres
pytest -q                                # API + migrations
cd .. && pip install -r worker/requirements-test.txt && pytest -q worker/tests
```
