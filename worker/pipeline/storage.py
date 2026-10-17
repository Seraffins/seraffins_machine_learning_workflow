"""Image I/O.

- User photos: downloaded from the frontend's R2 URLs (we have no R2 credentials).
- Our crops: private S3-compatible bucket (Supabase Storage).
"""
import io
import os
from urllib.parse import urlparse

import boto3
import httpx
from botocore.config import Config
from PIL import Image, ImageOps

from . import config

_s3 = None
_r2 = None


def s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            endpoint_url=config.STORAGE_ENDPOINT,
            aws_access_key_id=os.environ.get("STORAGE_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("STORAGE_SECRET_ACCESS_KEY", ""),
            region_name=config.STORAGE_REGION,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          retries={"max_attempts": 3}),
        )
    return _s3


def r2():
    global _r2
    if _r2 is None:
        _r2 = boto3.client(
            "s3",
            endpoint_url=config.R2_ENDPOINT,
            aws_access_key_id=config.R2_ACCESS_KEY_ID,
            aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          retries={"max_attempts": 3}),
        )
    return _r2


def _open(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # phone photos come rotated
    return img.convert("RGB")


def _is_r2_url(url: str) -> bool:
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()
    return (
        "r2.cloudflarestorage.com" in netloc
        or "r2.dev" in netloc
        or (config.R2_ACCOUNT_ID and config.R2_ACCOUNT_ID in netloc)
    )


def _extract_r2_key(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    # If the URL path begins with the bucket name (e.g. "totrace/avatars/file.jpg")
    if config.R2_BUCKET_NAME and path.startswith(f"{config.R2_BUCKET_NAME}/"):
        path = path[len(config.R2_BUCKET_NAME) + 1:]
    return path


def load_url(url: str) -> Image.Image:
    """Download a user photo from the frontend's R2 link or external host."""
    # If it is a Cloudflare R2 bucket URL and we have R2 credentials, read directly via S3 client
    if _is_r2_url(url) and config.R2_ACCESS_KEY_ID and config.R2_SECRET_ACCESS_KEY:
        try:
            key = _extract_r2_key(url)
            obj = r2().get_object(Bucket=config.R2_BUCKET_NAME, Key=key)
            return _open(obj["Body"].read())
        except Exception as e:
            # Fall back to standard HTTP GET (in case it is signed or public)
            print(f"[storage.load_url] R2 get_object error for {url}: {e}")

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
