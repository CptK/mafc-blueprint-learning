from abc import ABC
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import os
import pickle
import sqlite3
import threading
from pathlib import Path

from config.globals import temp_dir
from mafc.common.logger import logger
from mafc.utils.console import yellow
from mafc.tools.web_search.common import SearchResults, Query


class SearchPlatform(ABC):
    """Abstract base class for all local and remote search platforms."""

    name: str
    is_local: bool
    description: str

    def __init__(self):
        self.n_searches = 0
        assert self.name is not None

    def _before_search(self, query: Query):
        self.n_searches += 1
        logger.log(yellow(f"Searching {self.name} with query: {query}"))

    def search(self, query: Query | str) -> SearchResults | None:
        """Runs the API by submitting the query and obtaining a list of search results."""
        if isinstance(query, str):
            query = Query(text=query)
        self._before_search(query)
        return self._call_api(query)

    def _call_api(self, query: Query) -> SearchResults | None:
        raise NotImplementedError()

    def reset(self):
        """Resets the search API to its initial state (if applicable) and sets all stats to zero."""
        self.n_searches = 0

    @property
    def stats(self) -> dict:
        return {"Searches (API Calls)": self.n_searches}


class LocalSearchPlatform(SearchPlatform):
    is_local = True


class RemoteSearchPlatform(SearchPlatform):
    """Any search engine that leverages an external/non-local API. Employs a caching
    mechanism to improve efficiency."""

    is_local = False

    def __init__(self, activate_cache: bool = True, max_search_results: int = 10, **kwargs):
        super().__init__()
        self.max_search_results = max_search_results
        self.conn: sqlite3.Connection | None = None
        self.cur: sqlite3.Cursor | None = None
        self._cache_lock = threading.Lock()

        self.search_cached_first = activate_cache
        self.cache_file_name = f"{self.name}_cache.db"
        self.path_to_cache = Path(temp_dir) / self.cache_file_name
        self.n_cache_hits = 0
        self.n_cache_write_errors = 0

        if self.search_cached_first:
            if is_new := not self.path_to_cache.exists():
                os.makedirs(os.path.dirname(self.path_to_cache), exist_ok=True)
            self.conn = sqlite3.connect(self.path_to_cache, timeout=10, check_same_thread=False)
            try:
                # Enable Write-Ahead Logging (WAL) for concurrent access
                self.conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass  # WAL unsupported on this filesystem (e.g. NFS/VAST); use default journal mode
            self.cur = self.conn.cursor()
            if is_new:
                self._init_db()

    @property
    def _cache_ready(self) -> bool:
        return self.search_cached_first and self.conn is not None and self.cur is not None

    def close(self):
        """Closes cache DB connection if open."""
        if self.conn is None:
            return
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
        finally:
            self.conn = None
            self.cur = None

    def __del__(self):
        self.close()

    def _init_db(self):
        """Initializes a clean, new DB."""
        assert self.cur is not None
        assert self.conn is not None
        stmt = """
            CREATE TABLE Query(hash TEXT PRIMARY KEY, results BLOB);
        """
        with self._cache_lock:
            self.cur.execute(stmt)
            self.conn.commit()

    def _add_to_cache(self, query: Query, search_result: SearchResults):
        """Adds the given query-results pair to the cache."""
        if not self._cache_ready:
            return
        assert self.cur is not None
        assert self.conn is not None
        stmt = """
            INSERT INTO Query(hash, results)
            VALUES (?, ?);
        """
        try:
            with self._cache_lock:
                self.cur.execute(stmt, (self._cache_key(query), pickle.dumps(search_result)))
                self.conn.commit()
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            self.n_cache_write_errors += 1

    def _get_from_cache(self, query: Query) -> SearchResults | None:
        """Search the local in-memory data for matching results."""
        if not self._cache_ready:
            return None
        assert self.cur is not None
        stmt = """
            SELECT results FROM Query WHERE hash = ?;
        """
        with self._cache_lock:
            response = self.cur.execute(stmt, (self._cache_key(query),))
            result = response.fetchone()
        if result is not None:
            return pickle.loads(result[0])

    def _cache_key(self, query: Query) -> str:
        """Builds a deterministic key for cross-process cache reuse."""

        def _serialize(value):
            if value is None or isinstance(value, str | int | float | bool):
                return value
            if isinstance(value, date | datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, (list, tuple)):
                return [_serialize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): _serialize(val) for key, val in value.items()}
            return str(value)

        payload = {
            "text": query.text,
            "image": _serialize(query.image),
            "search_mode": _serialize(query.search_mode),
            "limit": query.limit,
            "start_date": _serialize(query.start_date),
            "end_date": _serialize(query.end_date),
        }
        query_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(query_json.encode("utf-8")).hexdigest()

    def search(self, query: Query | str) -> SearchResults | None:
        if isinstance(query, str):
            query = Query(text=query)

        # Try to load from cache
        if self._cache_ready:
            cache_results = self._get_from_cache(query)
            if cache_results:
                self.n_cache_hits += 1
                return cache_results

        # Run actual search
        search_result = super().search(query)
        if search_result:
            self._add_to_cache(query, search_result)
        return search_result

    def reset(self):
        super().reset()
        self.n_cache_hits = 0
        self.n_cache_write_errors = 0

    @property
    def stats(self) -> dict:
        stats = super().stats
        stats.update({"Cache hits": self.n_cache_hits, "Cache write errors": self.n_cache_write_errors})
        return stats
