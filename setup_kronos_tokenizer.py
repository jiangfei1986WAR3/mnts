from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))


REPO_URL = "https://github.com/shiyu-coder/Kronos.git"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup official Kronos repo and tokenizer-only assets.")
    parser.add_argument("--workspace", default="external")
    parser.add_argument("--repo-dir-name", default="Kronos")
    parser.add_argument("--tokenizer-dir-name", default="kronos_tokenizer_base")
    parser.add_argument("--hf-cache-dir", default=".hf_cache")
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None) -> None:
    print(">", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        print(f"Kronos repo already exists: {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])


def ensure_hf_download(tokenizer_dir: Path, hf_cache_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it first, then rerun this script."
        ) from exc

    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir.resolve())

    snapshot_download(
        repo_id=TOKENIZER_REPO,
        local_dir=str(tokenizer_dir.resolve()),
        local_dir_use_symlinks=False,
    )
    print(f"Tokenizer files saved to: {tokenizer_dir.resolve()}")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    workspace = (root / args.workspace).resolve()
    repo_dir = workspace / args.repo_dir_name
    tokenizer_dir = workspace / args.tokenizer_dir_name
    hf_cache_dir = (root / args.hf_cache_dir).resolve()

    ensure_repo(repo_dir)
    ensure_hf_download(tokenizer_dir, hf_cache_dir)

    print()
    print("Kronos tokenizer setup complete.")
    print(f"Repo dir: {repo_dir}")
    print(f"Tokenizer dir: {tokenizer_dir}")
    print()
    print("Note:")
    print("- This script downloads the official Kronos codebase.")
    print("- It only pulls the tokenizer repository, not the full Kronos predictor weights.")
    print("- Using the tokenizer still requires torch + huggingface_hub in your Python environment.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
