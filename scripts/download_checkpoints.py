#!/usr/bin/env python3
"""Download ARGaze checkpoints from a Hugging Face model repository."""

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="Hugging Face repo id, e.g. org/argaze.")
    parser.add_argument("--manifest", default=str(ROOT / "checkpoints/manifest.json"))
    parser.add_argument("--output", default=str(ROOT / "checkpoints"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    repo_id = args.repo or manifest.get("hf_repo") or "<hf-org-or-user>/argaze"
    output_root = Path(args.output)

    if args.dry_run:
        for item in manifest["files"]:
            print(f"{repo_id}:{item['hf_path']} -> {output_root / item['local_path']}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to download checkpoints.") from exc

    for item in manifest["files"]:
        target = output_root / item["local_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(repo_id=repo_id, filename=item["hf_path"])
        shutil.copy2(downloaded, target)
        print(f"Downloaded {item['hf_path']} -> {target}")


if __name__ == "__main__":
    main()
