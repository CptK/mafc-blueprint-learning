"""Shared configuration across all project code.

Loads configuration from environment variables. If `config/.env` exists,
its key=value pairs are loaded into the environment (without failing when
missing). Prefer environment variables; `.env` is a local convenience.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Directories
working_dir = Path.cwd()  # working_dir should be project root
data_root_dir = Path("data/")  # Where the datasets are stored
result_base_dir = working_dir / "out/"  # Where outputs are to be saved
temp_dir = working_dir / "temp/"  # Where to store temporary files (e.g. search cache)
os.makedirs(temp_dir, exist_ok=True)


# Try to load local .env (optional). Do not overwrite existing env vars.
env_path = working_dir / "config/.env"
load_dotenv(dotenv_path=env_path, override=False)
google_service_account_key_path = Path("config/google_service_account_key.json")


# Common endpoints/configs
firecrawl_url = os.environ.get("firecrawl_url", "http://localhost:3002")
selfhosted_url = os.environ.get("selfhosted_url", None)
geolocator_url = os.environ.get("geolocator_url", "http://0.0.0.0:5555")
data_path = os.environ.get("data_path", "data/")

# TruFor manipulation detection
# Read-only stores of precomputed scores, e.g. "data/veritas_2026_q1/trufor:data/other/trufor"
trufor_stores = [Path(p) for p in os.environ.get("trufor_stores", "").split(":") if p]
trufor_cache_dir = temp_dir / "trufor"  # where on-the-fly scores get cached

# Sightengine AI-generated / deepfake / ai_speech detection
# Read-only stores of precomputed scores, e.g. "data/veritas_2026_q1/sightengine:data/other/sightengine"
sightengine_stores = [Path(p) for p in os.environ.get("sightengine_stores", "").split(":") if p]
sightengine_cache_dir = temp_dir / "sightengine"  # where on-the-fly scores get cached

# Oracle manipulation detector — ceiling experiments ONLY, reads the answer key.
# e.g. "data/veritas_2026_q1/media_integrity_labels.json"
_oracle_labels = os.environ.get("oracle_labels_path", "")
oracle_labels_path = Path(_oracle_labels) if _oracle_labels else None

# GenD face-deepfake detection
# Read-only stores of precomputed scores, e.g. "data/veritas_2026_q1/gend:data/other/gend"
gend_stores = [Path(p) for p in os.environ.get("gend_stores", "").split(":") if p]
gend_cache_dir = temp_dir / "gend"  # where on-the-fly scores get cached

# Geolocator defaults
default_countries_path = Path(__file__).resolve().with_name("default_countries_list.txt")
with default_countries_path.open("r", encoding="utf-8") as f:
    geolocator_default_countries = [line.strip() for line in f if line.strip()]


# Random seed for reproducibility
random_seed = 42
