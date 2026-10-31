"""Download a sentence-transformers model through a faster mirror, verified against huggingface.co.

huggingface.co can be very slow on some networks. This fetches the official commit and file
hashes from huggingface.co (small requests), downloads the files from a mirror pinned to that
commit, and checks every file against the official hashes before it is used.

Usage (from backend/):
    python -m scripts.fetch_model                                   # BAAI/bge-small-en-v1.5
    python -m scripts.fetch_model --repo BAAI/bge-base-en-v1.5
After this, the model loads from the local cache (set HF_HUB_OFFLINE=1 to skip network checks).
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP downloads work through mirrors

import requests  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

OFFICIAL = "https://huggingface.co"
FILES = [  # what sentence-transformers needs; skips the ONNX / PyTorch-bin duplicates
    "config.json", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
    "model.safetensors", "1_Pooling/config.json",
]


def official_manifest(repo: str) -> tuple[str, dict[str, dict]]:
    commit = requests.get(f"{OFFICIAL}/api/models/{repo}", timeout=120).json()["sha"]
    tree = requests.get(f"{OFFICIAL}/api/models/{repo}/tree/{commit}?recursive=true", timeout=120).json()
    return commit, {e["path"]: e for e in tree if e["type"] == "file"}


def file_matches(path: Path, entry: dict) -> bool:
    data = path.read_bytes()
    if entry.get("lfs"):  # large files: sha256 of the content
        return hashlib.sha256(data).hexdigest() == entry["lfs"]["oid"]
    # small files: git blob id
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == entry["oid"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    args = parser.parse_args()

    commit, manifest = official_manifest(args.repo)
    wanted = [f for f in FILES if f in manifest]
    print(f"{args.repo} @ {commit[:10]}: {len(wanted)} files, "
          f"{sum(manifest[f]['size'] for f in wanted) / 1e6:.0f} MB from {args.mirror}")

    folder = Path(snapshot_download(args.repo, revision=commit, allow_patterns=wanted, endpoint=args.mirror))

    bad = [f for f in wanted if not (folder / f).exists() or not file_matches(folder / f, manifest[f])]
    if bad:
        shutil.rmtree(folder, ignore_errors=True)
        print(f"VERIFICATION FAILED for {bad}; deleted the download.")
        return 1

    # Point refs/main at the verified commit so the model loads by name.
    refs = folder.parents[1] / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(commit)
    print(f"All {len(wanted)} files match huggingface.co. Cached at {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
