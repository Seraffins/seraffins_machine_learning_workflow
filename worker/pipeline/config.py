"""All tunables in one place. Override any of these with env vars (Modal secret)."""
import os


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

# Model files inside the Modal volume
DETECTOR_WEIGHTS = os.path.join(MODELS_DIR, os.environ.get("DETECTOR_FILE", "rfdetr.pth"))
SEGMENTER_WEIGHTS = os.path.join(MODELS_DIR, os.environ.get("SEGMENTER_FILE", "segmenter.pt"))
EMBEDDER_WEIGHTS = os.path.join(MODELS_DIR, os.environ.get("EMBEDDER_FILE", "embedder.pt"))  # optional
BIOCLIP_MODEL = os.environ.get("BIOCLIP_MODEL", "hf-hub:imageomics/bioclip-2.5-vith14")
DINOV2_MODEL = os.environ.get("DINOV2_MODEL", "facebook/dinov2-base")

# Bump these whenever you retrain / swap the model. Vectors are searched per version.
SEG_MODEL_VERSION = os.environ.get("SEG_MODEL_VERSION", "seg-v1")
EMBED_MODEL_VERSION = os.environ.get("EMBED_MODEL_VERSION", "embed-v1")
DETECTOR_VERSION = os.environ.get("DETECTOR_VERSION", "rfdetr-v1")
SPECIES_MODEL_VERSION = os.environ.get("SPECIES_MODEL_VERSION", "bioclip-2.5")

EMBED_DIM = _i("EMBED_DIM", 768)  # must match vector(768) in schema.sql

# Quality gate
DET_MIN_CONF = _f("DET_MIN_CONF", 0.5)
MIN_CROP_SIDE = _i("MIN_CROP_SIDE", 96)
BLUR_MIN_VAR = _f("BLUR_MIN_VAR", 60.0)       # Laplacian variance; tune on your data
MIN_USABLE_IMAGES = _i("MIN_USABLE_IMAGES", 2)  # fewer -> data_deficient
MIN_MASK_AREA_FRAC = _f("MIN_MASK_AREA_FRAC", 0.02)

# Species
SPECIES_MIN_CONF = _f("SPECIES_MIN_CONF", 0.5)  # below -> unknown

# Re-ID
TOP_K = _i("TOP_K", 10)
MATCH_INLIERS = _i("MATCH_INLIERS", 60)   # >= this: confident match
NEW_INLIERS = _i("NEW_INLIERS", 20)       # < this: confident new
MARGIN_INLIERS = _i("MARGIN_INLIERS", 15) # best must beat 2nd-best turtle by this
CONF_MID = _f("CONF_MID", 40.0)           # inlier count mapping to 0.5 confidence
CONF_SCALE = _f("CONF_SCALE", 10.0)

# Views
REID_VIEWS = ("left_face", "right_face")
ALL_VIEWS = ("whole", "left_face", "right_face", "front", "carapace", "rear")

SPECIES = {
    # label: prompt terms (scientific + common name)
    "green_turtle": "Chelonia mydas, green sea turtle",
    "hawksbill": "Eretmochelys imbricata, hawksbill sea turtle",
    "loggerhead": "Caretta caretta, loggerhead sea turtle",
    "olive_ridley": "Lepidochelys olivacea, olive ridley sea turtle",
    "leatherback": "Dermochelys coriacea, leatherback sea turtle",
    "kemps_ridley": "Lepidochelys kempii, Kemp's ridley sea turtle",
    "flatback": "Natator depressus, flatback sea turtle",
}
DATA_DEFICIENT = "data_deficient"
UNKNOWN = "unknown"

# Crop storage: our own private bucket (Supabase Storage via its S3 API).
# User photos are downloaded from the frontend's R2 URLs or fetched via R2 S3 API.
STORAGE_ENDPOINT = os.environ.get("STORAGE_ENDPOINT", "")  # https://<ref>.supabase.co/storage/v1/s3
STORAGE_REGION = os.environ.get("STORAGE_REGION", "ap-south-1")
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "turtle-crops")
CROPS_PREFIX = os.environ.get("CROPS_PREFIX", "crops")

# Cloudflare R2 bucket credentials for direct image access
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID").strip()
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY").strip()
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME").strip()
R2_ENDPOINT = (
    os.environ.get("R2_ENDPOINT", "").strip()
    or (f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else "")
)
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "").strip()

# Optional: POST the finished result to the frontend backend
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

