"""Postgres + pgvector access used by the worker."""
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from . import config


def _plain_url(url: str) -> str:
    # psycopg wants a libpq URL; drop SQLAlchemy driver suffixes like "+psycopg2"
    scheme, sep, rest = url.partition("://")
    return f"postgresql://{rest}" if sep and scheme.startswith("postgres") else url


def connect():
    # prepare_threshold=None keeps it working behind Supabase/Neon poolers
    conn = psycopg.connect(_plain_url(config.DATABASE_URL), prepare_threshold=None)
    register_vector(conn)
    return conn


def set_job(conn, job_id, status, result=None, error=None):
    conn.execute(
        "update jobs set status=%s, result=coalesce(%s, result), error=%s, updated_at=now() where id=%s",
        (status, Jsonb(result) if result is not None else None, error, job_id),
    )
    conn.commit()


def search(conn, vec, side, k):
    conn.execute("set local hnsw.ef_search = 100")
    return conn.execute(
        """
        select e.turtle_id, e.scute_crop_key, 1 - (e.embedding <=> %(v)s) as sim
        from embeddings e join turtles t on t.id = e.turtle_id
        where e.side = %(side)s
          and e.embed_model_version = %(ev)s and e.seg_model_version = %(sv)s
          and t.merged_into is null
        order by e.embedding <=> %(v)s
        limit %(k)s
        """,
        {"v": vec, "side": side, "ev": config.EMBED_MODEL_VERSION,
         "sv": config.SEG_MODEL_VERSION, "k": k},
    ).fetchall()


def create_turtle(conn, species, status):
    return conn.execute(
        "insert into turtles (species, status) values (%s, %s) returning id", (species, status)
    ).fetchone()[0]


def insert_sighting(conn, job_id, turtle_id, species, species_conf, match_status, match_conf):
    return conn.execute(
        """insert into sightings (job_id, turtle_id, species, species_confidence, match_status, match_confidence)
           values (%s,%s,%s,%s,%s,%s) returning id""",
        (job_id, turtle_id, species, species_conf, match_status, match_conf),
    ).fetchone()[0]


def insert_embedding(conn, turtle_id, sighting_id, side, det_key, scute_key, vec):
    conn.execute(
        """insert into embeddings (turtle_id, sighting_id, side, det_crop_key, scute_crop_key,
                                   embedding, embed_model_version, seg_model_version)
           values (%s,%s,%s,%s,%s,%s,%s,%s)""",
        (turtle_id, sighting_id, side, det_key, scute_key, vec,
         config.EMBED_MODEL_VERSION, config.SEG_MODEL_VERSION),
    )


def insert_review(conn, sighting_id, provisional_id, candidates):
    conn.execute(
        "insert into review_queue (sighting_id, provisional_turtle_id, candidates) values (%s,%s,%s)",
        (sighting_id, provisional_id, Jsonb(candidates)),
    )
