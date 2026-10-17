"""Endpoints for the frontend's backend (header X-API-Key)."""
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import dispatch
from ..config import settings
from ..db import get_db
from ..models import Job, Sighting, Turtle
from ..schemas import AcceptedOut, IdentifyIn, JobOut
from ..security import limiter, require_api_key

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
log = logging.getLogger("turtle.api")

SPECIES_LABELS = ["green_turtle", "hawksbill", "loggerhead", "olive_ridley", "leatherback",
                  "kemps_ridley", "flatback", "data_deficient", "unknown"]


def job_out(j: Job) -> dict:
    return {"job_id": str(j.id), "uid": j.user_id, "status": j.status, "result": j.result,
            "error": j.error, "created_at": j.created_at, "updated_at": j.updated_at}


def accepted(j: Job) -> dict:
    return {"job_id": str(j.id), "status": j.status, "poll": f"/v1/jobs/{j.id}"}


@router.get("/species")
def species():
    return {"labels": SPECIES_LABELS}


@router.post("/identify", status_code=202, response_model=AcceptedOut)
def identify(
    body: IdentifyIn,
    db: Session = Depends(get_db),
    key: str = Depends(require_api_key),
    idempotency_key: str | None = Header(default=None, max_length=128),
):
    limiter.check(f"uid:{body.uid}", settings.RATE_LIMIT_PER_UID)
    limiter.check(f"key:{key}", settings.RATE_LIMIT_PER_KEY)

    # Same Idempotency-Key from the same uid -> return the original job, never run twice.
    if idempotency_key:
        existing = db.scalar(select(Job).where(Job.user_id == body.uid,
                                               Job.idempotency_key == idempotency_key))
        if existing:
            return accepted(existing)

    job = Job(request=body.model_dump(mode="json"), user_id=body.uid,
              idempotency_key=idempotency_key)
    db.add(job)
    try:
        db.commit()
    except IntegrityError:  # concurrent retry with the same key won the race
        db.rollback()
        return accepted(db.scalar(select(Job).where(Job.user_id == body.uid,
                                                    Job.idempotency_key == idempotency_key)))

    try:
        job.modal_call_id = dispatch.start_identify(str(job.id), body.to_pipeline(), body.uid)
    except Exception:
        log.exception("dispatch to Modal failed for job %s", job.id)
        job.status, job.error = "failed", "dispatch failed"
        db.commit()
        raise HTTPException(503, "model service unavailable, try again")
    db.commit()
    return accepted(job)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job_out(job)


@router.get("/users/{uid}/jobs")
def user_jobs(uid: str, limit: int = 20, before: datetime | None = None,
              db: Session = Depends(get_db)):
    """A user's submissions, newest first. Pass `before` (created_at of the last item) to page."""
    q = select(Job).where(Job.user_id == uid)
    if before:
        q = q.where(Job.created_at < before)
    rows = db.scalars(q.order_by(Job.created_at.desc()).limit(max(1, min(limit, 100)))).all()
    return {"items": [job_out(j) for j in rows]}


@router.get("/turtles/{turtle_id}")
def get_turtle(turtle_id: str, db: Session = Depends(get_db)):
    t = db.get(Turtle, turtle_id)
    if not t:
        raise HTTPException(404, "turtle not found")
    if t.merged_into:
        return {"id": t.id, "merged_into": t.merged_into}
    sightings = db.scalars(select(Sighting).where(Sighting.turtle_id == turtle_id)
                           .order_by(Sighting.created_at.desc())).all()
    return {
        "id": t.id, "species": t.species, "status": t.status, "created_at": t.created_at,
        "sightings_count": len(sightings),
        "sightings": [{"id": str(s.id), "job_id": str(s.job_id), "species": s.species,
                       "species_confidence": s.species_confidence,
                       "match_status": s.match_status, "match_confidence": s.match_confidence,
                       "created_at": s.created_at} for s in sightings],
    }
