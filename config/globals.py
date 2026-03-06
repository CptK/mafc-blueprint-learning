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
