"""
Cached Supabase fetcher — saves rows to local CSV, only fetches new rows on retrain.
First fetch: slow (30-60 min for 150k rows on free tier).
Subsequent fetches: fast (only fetches rows since last cache).
"""
import os
import json
import logging
import time
import requests
import numpy as np

log = logging.getLogger(__name__)

SUPABASE_URL = "https://kcluwyzyetmkxhvszpxi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtjbHV3eXp5ZXRta3hodnN6cHhpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA4NTY2NCwiZXhwIjoyMDg5NjYxNjY0fQ.IbxuXRW0K9_UFZKG1i951EoL9KtCsOXCaz5Z_YqsmYE"
REST_H = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

CACHE_DIR = "cache"


def _rest_fetch_cursor(table, params, limit=500):
    """Fetch using created_at cursor with retries."""
    rows = []
    retries = 0
    cursor = ""
    while True:
        p = {**params, "limit": limit}
        if cursor:
            p["created_at"] = f"gt.{cursor}"
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=REST_H, params=p, timeout=180)
            if not r.text:
                retries += 1
                if retries > 5:
                    break
                time.sleep(2)
                continue
            batch = r.json()
            if isinstance(batch, dict):
                retries += 1
                if retries > 5:
                    log.warning(f"  API error after 5 retries at {len(rows):,} rows: {batch.get('message','?')}")
                    break
                log.warning(f"  API error, retry {retries}: {batch.get('message','?')[:60]}")
                time.sleep(5)
                continue
            if not batch or not isinstance(batch, list):
                break
            rows.extend(batch)
            retries = 0
            if len(rows) % 5000 == 0:
                log.info(f"  Fetched {len(rows):,}...")
            if len(batch) < limit:
                break
            last_ts = batch[-1].get("created_at")
            if not last_ts:
                break
            cursor = last_ts
        except Exception as e:
            retries += 1
            if retries > 5:
                log.warning(f"  Fetch stopped at {len(rows):,} rows: {e}")
                break
            log.warning(f"  Retry {retries}: {e}")
            time.sleep(5)
    return rows


def cached_fetch(cache_name, cols, filters, table="market_snapshots"):
    """
    Fetch from Supabase with local JSON cache.
    First call: full fetch + save to cache/{cache_name}.json
    Subsequent calls: load cache + fetch only new rows since last cached timestamp.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.json")

    cached_rows = []
    last_ts = ""

    # Load existing cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_rows = json.load(f)
            if cached_rows:
                last_ts = cached_rows[-1].get("created_at", "")
                log.info(f"  Cache loaded: {len(cached_rows):,} rows (last: {last_ts[-20:]})")
        except Exception as e:
            log.warning(f"  Cache load failed: {e}, doing full fetch")
            cached_rows = []
            last_ts = ""

    # Fetch new rows since last cache
    params = {
        "select": cols,
        **filters,
        "order": "created_at.asc",
    }
    if last_ts:
        params["created_at"] = f"gt.{last_ts}"
        log.info(f"  Fetching new rows since {last_ts[-20:]}...")
    else:
        log.info(f"  No cache — full fetch...")

    new_rows = _rest_fetch_cursor(table, params)
    log.info(f"  New rows: {len(new_rows):,}")

    # Merge
    all_rows = cached_rows + new_rows
    log.info(f"  Total: {len(all_rows):,} rows")

    # Save cache
    try:
        with open(cache_path, "w") as f:
            json.dump(all_rows, f)
        log.info(f"  Cache saved: {cache_path}")
    except Exception as e:
        log.warning(f"  Cache save failed: {e}")

    return all_rows
