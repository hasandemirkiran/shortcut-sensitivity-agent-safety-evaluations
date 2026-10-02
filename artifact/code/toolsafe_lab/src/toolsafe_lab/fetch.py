from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from toolsafe_lab.manifest import ASSETS, Asset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, asset: Asset) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == asset.size
        and sha256_file(path) == asset.sha256
    )


def download_asset(root: Path, asset: Asset, force: bool = False) -> dict[str, object]:
    destination = root / asset.destination
    if not force and verify(destination, asset):
        return {"path": asset.destination, "status": "verified", "size": asset.size}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        request = urllib.request.Request(
            asset.url,
            headers={"User-Agent": "ToolSafe-Lab/0.1 dataset fetcher"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if not verify(temporary, asset):
            actual_size = temporary.stat().st_size if temporary.exists() else -1
            actual_hash = sha256_file(temporary) if temporary.exists() else "missing"
            raise ValueError(
                f"Integrity failure for {asset.destination}: "
                f"size={actual_size}, sha256={actual_hash}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {"path": asset.destination, "status": "downloaded", "size": asset.size}


def fetch_all(root: Path, force: bool = False, workers: int = 4) -> list[dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_asset, root, asset, force): asset for asset in ASSETS
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['status']:>10}  {result['path']}")

    results.sort(key=lambda row: str(row["path"]))
    provenance = {
        "assets": [
            {
                "path": asset.destination,
                "source": asset.url,
                "sha256": asset.sha256,
                "size": asset.size,
                "kind": asset.kind,
            }
            for asset in ASSETS
        ]
    }
    (root / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return results

