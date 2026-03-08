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


# Try to load local .env (optional). Do not overwrite existing env vars.
env_path = working_dir / "config/.env"
load_dotenv(dotenv_path=env_path, override=False)


# Common endpoints/configs
firecrawl_url = os.environ.get("firecrawl_url", "http://localhost:3002")
selfhosted_url = os.environ.get("selfhosted_url", None)
geolocator_url = os.environ.get("geolocator_url", "http://0.0.0.0:5555")
data_path = os.environ.get("data_path", "data/")

# Geolocator defaults
default_countries_path = Path(__file__).resolve().with_name("default_countries_list.txt")
with default_countries_path.open("r", encoding="utf-8") as f:
    geolocator_default_countries = [line.strip() for line in f if line.strip()]


# Random seed for reproducibility
random_seed = 42
