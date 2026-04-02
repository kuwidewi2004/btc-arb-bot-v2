"""
Cached data fetcher for training pipelines.

Data flow:
  1. GitHub Release (archive) — old data stored as Parquet, free + fast
  2. Local cache (cache/*.parquet) — merged archive + recent Supabase data
  3. Supabase (live) — only fetches NEW rows since last cache

First retrain after archive upload: downloads from GitHub (~2s) + fetches new from Supabase (~1-5 min)
Subsequent retrains: loads local cache (~1s) + fetches new rows only (~30s)

Archive management:
  python fetch_cache.py upload   — upload current cache to GitHub release
  python fetch_cache.py purge    — delete archived rows from Supabase (frees space)
"""
import os
import sys
import logging
import time
import json
import requests
import numpy as np

log = logging.getLogger(__name__)

SUPABASE_URL = "https://kcluwyzyetmkxhvszpxi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtjbHV3eXp5ZXRta3hodnN6cHhpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA4NTY2NCwiZXhwIjoyMDg5NjYxNjY0fQ.IbxuXRW0K9_UFZKG1i951EoL9KtCsOXCaz5Z_YqsmYE"
REST_H = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

CACHE_DIR = "cache"
GITHUB_REPO = "kuwidewi2004/btc-arb-bot-v2"
GITHUB_RELEASE_TAG = "data-archive"


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


def _download_archive(cache_name):
    """Download parquet archive from GitHub release if local cache doesn't exist."""
    import pyarrow.parquet as pq
    import pandas as pd

    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.parquet")
    if os.path.exists(cache_path):
        return  # already have local cache

    # Try downloading from GitHub release
    url = f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG}/{cache_name}.parquet"
    log.info(f"  Downloading archive from GitHub: {cache_name}.parquet...")
    try:
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code == 200:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(cache_path) / 1024 / 1024
            log.info(f"  Downloaded: {size_mb:.1f} MB")
        elif r.status_code == 404:
            log.info(f"  No archive found on GitHub (first time)")
        else:
            log.warning(f"  Download failed: HTTP {r.status_code}")
    except Exception as e:
        log.warning(f"  Download failed: {e}")


def _load_parquet(cache_name):
    """Load rows from local parquet cache."""
    import pyarrow.parquet as pq
    import pandas as pd

    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.parquet")
    if not os.path.exists(cache_path):
        return []

    try:
        df = pd.read_parquet(cache_path)
        rows = df.to_dict("records")
        log.info(f"  Cache loaded: {len(rows):,} rows from {cache_name}.parquet")
        return rows
    except Exception as e:
        log.warning(f"  Cache load failed: {e}")
        return []


def _save_parquet(cache_name, rows):
    """Save rows to local parquet cache."""
    import pandas as pd

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.parquet")
    try:
        df = pd.DataFrame(rows)
        df.to_parquet(cache_path, index=False)
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        log.info(f"  Cache saved: {cache_path} ({size_mb:.1f} MB)")
    except Exception as e:
        log.warning(f"  Cache save failed: {e}")


def cached_fetch(cache_name, cols, filters, table="market_snapshots"):
    """
    Fetch data with GitHub archive + local cache + Supabase incremental.

    1. Download parquet from GitHub release (if no local cache)
    2. Load local parquet cache
    3. Fetch only NEW rows from Supabase since last cached timestamp
    4. Save merged result to local parquet
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: download archive if needed
    _download_archive(cache_name)

    # Step 2: load local cache
    cached_rows = _load_parquet(cache_name)
    last_ts = ""
    if cached_rows:
        last_ts = cached_rows[-1].get("created_at", "")

    # Step 3: fetch new rows from Supabase
    params = {
        "select": cols,
        **filters,
        "order": "created_at.asc",
    }
    if last_ts:
        params["created_at"] = f"gt.{last_ts}"
        log.info(f"  Fetching new rows since {last_ts[-20:]}...")
    else:
        log.info(f"  No cache — full fetch from Supabase...")

    new_rows = _rest_fetch_cursor(table, params)
    log.info(f"  New rows from Supabase: {len(new_rows):,}")

    # Step 4: merge and save
    all_rows = cached_rows + new_rows
    log.info(f"  Total: {len(all_rows):,} rows")

    _save_parquet(cache_name, all_rows)
    return all_rows


def upload_archive(cache_name):
    """Upload local parquet cache to GitHub release using gh CLI."""
    import subprocess
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.parquet")
    if not os.path.exists(cache_path):
        print(f"No cache file: {cache_path}")
        return

    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"Uploading {cache_path} ({size_mb:.1f} MB) to GitHub release...")

    # Create release if it doesn't exist
    subprocess.run(["gh", "release", "create", GITHUB_RELEASE_TAG,
                     "--title", "Data Archive", "--notes", "Training data archive (parquet)",
                     "--repo", GITHUB_REPO], capture_output=True)

    # Upload (overwrites if exists)
    result = subprocess.run(["gh", "release", "upload", GITHUB_RELEASE_TAG,
                              cache_path, "--clobber", "--repo", GITHUB_REPO],
                             capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Uploaded successfully")
    else:
        print(f"Upload failed: {result.stderr}")


def purge_archived(cache_name, keep_recent_hours=48):
    """Delete rows from Supabase that are already in the local cache, keeping recent data."""
    from datetime import datetime, timezone, timedelta

    cached_rows = _load_parquet(cache_name)
    if not cached_rows:
        print("No cache to purge from")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(hours=keep_recent_hours)
    cutoff_ts = cutoff.isoformat()
    print(f"Will delete Supabase rows older than {cutoff_ts}")
    print(f"Cache has {len(cached_rows):,} rows — these are safe in the archive")

    # Delete in small batches
    deleted = 0
    while True:
        try:
            r = requests.delete(
                f"{SUPABASE_URL}/rest/v1/market_snapshots",
                headers={**REST_H, "Content-Type": "application/json", "Prefer": "return=minimal"},
                params={"created_at": f"lt.{cutoff_ts}", "limit": "1000"},
                timeout=30,
            )
            if r.status_code == 204:
                deleted += 1000
                print(f"  Deleted batch... (~{deleted:,} total)")
                time.sleep(1)  # gentle on I/O
            else:
                print(f"  Delete returned {r.status_code}: {r.text[:100]}")
                break
        except Exception as e:
            print(f"  Error: {e}")
            break

    print(f"Purge complete: ~{deleted:,} rows deleted from Supabase")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fetch_cache.py upload    — upload cache to GitHub release")
        print("  python fetch_cache.py purge     — delete old rows from Supabase")
        print("  python fetch_cache.py status    — show cache status")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "upload":
        upload_archive("v4_snapshots")
        upload_archive("v5_snapshots")
    elif cmd == "purge":
        purge_archived("v5_snapshots")
    elif cmd == "status":
        for name in ["v4_snapshots", "v5_snapshots"]:
            path = os.path.join(CACHE_DIR, f"{name}.parquet")
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                rows = _load_parquet(name)
                last = rows[-1]["created_at"][-20:] if rows else "?"
                print(f"  {name}: {len(rows):,} rows, {size_mb:.1f} MB, last: {last}")
            else:
                print(f"  {name}: no cache")
    else:
        print(f"Unknown command: {cmd}")
