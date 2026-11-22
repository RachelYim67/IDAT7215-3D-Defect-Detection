"""Resume a raw-image download, verifying every file against the pinned Git tree."""
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REVISION = "52fe75dec70c1e8046493b4e3256f8f752a4d39d"
ROOT = Path(__file__).resolve().parent / "data" / f"AMdataset-{REVISION}"
LOCAL = local()


def fetch(url):
    if not hasattr(LOCAL, "client"):
        LOCAL.client = requests.Session()
        retries = Retry(total=6, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        LOCAL.client.mount("https://", HTTPAdapter(max_retries=retries))
    response = LOCAL.client.get(url, timeout=(15, 90))
    response.raise_for_status()
    return response


def valid(data, entry):
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest() == entry["sha"]


def download(entry):
    name = entry["path"]
    path = ROOT / name
    if path.exists() and valid(path.read_bytes(), entry):
        return
    url = f"https://raw.githubusercontent.com/mahsan2/AMdataset/{REVISION}/{quote(name)}"
    try:
        data = fetch(url).content
    except requests.RequestException:
        data = subprocess.check_output(["curl", "-fsSL", "--retry", "5", "--retry-all-errors", url])
    if not valid(data, entry):
        raise ValueError(f"Failed Git hash check: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part")
    temp.write_bytes(data)
    temp.replace(path)


if __name__ == "__main__":
    entries = []
    manifest = Path(__file__).with_name("originals-tree.txt")
    for line in manifest.read_text().splitlines():
        metadata, path = line.split("\t")
        entries.append({"path": path, "sha": metadata.split()[-1]})
    if len(entries) != 271:
        raise ValueError("Unexpected dataset tree")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for n, _ in enumerate(executor.map(download, entries), 1):
            if n % 25 == 0 or n == len(entries):
                print(f"Verified {n}/{len(entries)}", flush=True)
