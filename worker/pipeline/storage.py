"""Image I/O.

- User photos: downloaded from the frontend's R2 URLs (we have no R2 credentials).
- Our crops: private S3-compatible bucket (Supabase Storage).
"""
import io
import os

import boto3
import httpx
from botocore.config import Config
from PIL import Image, ImageOps

from . import config

_s3 = None


def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            endpoint_url=config.STORAGE_ENDPOINT,
            aws_access_key_id=os.environ["STORAGE_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["STORAGE_SECRET_ACCESS_KEY"],
            region_name=config.STORAGE_REGION,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          retries={"max_attempts": 3}),
        )
    return _s3


def _open(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # phone photos come rotated
    return img.convert("RGB")


def load_url(url: str) -> Image.Image:
    """Download a user photo from the frontend's R2 link."""
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return _open(resp.content)


def load_key(key: str) -> Image.Image:
    """Load one of our stored crops."""
    obj = s3().get_object(Bucket=config.STORAGE_BUCKET, Key=key)
    return _open(obj["Body"].read())


def save_jpeg(img: Image.Image, key: str, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    s3().put_object(Bucket=config.STORAGE_BUCKET, Key=key, Body=buf.getvalue(),
                    ContentType="image/jpeg")
    return key
