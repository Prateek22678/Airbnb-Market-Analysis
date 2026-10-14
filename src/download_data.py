"""
Script to download Inside Airbnb raw data for Austin, Texas (Snapshot: 2026-09-17).
Saves archives to data/raw/ directory.
"""
import os
import urllib.request
import time

AUSTIN_BASE_URL = "https://data.insideairbnb.com/united-states/tx/austin/2026-09-17/data"
FILES_TO_DOWNLOAD = [
    "listings.csv.gz",
    "calendar.csv.gz",
    "reviews.csv.gz",
]

def download_raw_data(dest_dir: str = "data/raw") -> None:
    os.makedirs(dest_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AirbnbProjectMentor/1.0"}

    for filename in FILES_TO_DOWNLOAD:
        file_path = os.path.join(dest_dir, filename)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            print(f"[SKIP] {filename} already exists ({os.path.getsize(file_path):,} bytes).")
            continue

        url = f"{AUSTIN_BASE_URL}/{filename}"
        print(f"[DOWNLOAD] Downloading {filename} from {url}...")
        start_time = time.time()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(file_path, "wb") as out_f:
                total_bytes = 0
                while True:
                    chunk = resp.read(1024 * 512) # 512KB chunks
                    if not chunk:
                        break
                    out_f.write(chunk)
                    total_bytes += len(chunk)
            elapsed = time.time() - start_time
            print(f"[SUCCESS] {filename} downloaded: {total_bytes:,} bytes in {elapsed:.1f}s.")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

if __name__ == "__main__":
    download_raw_data()
