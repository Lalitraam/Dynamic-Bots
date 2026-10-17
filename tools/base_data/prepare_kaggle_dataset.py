"""
Package codebase tools and model definition for Kaggle environment.
Builds the base-data-tools directory structure and optional .tar.gz archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tarfile

# Source files to bundle into base-data-tools
# Maps (source relative to repo root) -> (destination relative to base-data-tools)
FILES_TO_BUNDLE = [
    # app/
    ("backend/app/__init__.py", "app/__init__.py"),
    ("backend/app/config.py", "app/config.py"),
    ("backend/app/ml/__init__.py", "app/ml/__init__.py"),
    ("backend/app/ml/encoding.py", "app/ml/encoding.py"),
    ("backend/app/ml/encoding_v3.py", "app/ml/encoding_v3.py"),
    ("backend/app/ml/model_v2.py", "app/ml/model_v2.py"),
    ("backend/app/ml/losses_v2.py", "app/ml/losses_v2.py"),
    ("backend/app/ml/tensor_cache.py", "app/ml/tensor_cache.py"),
    ("backend/app/ml/dataset_v3.py", "app/ml/dataset_v3.py"),
    ("backend/app/services/__init__.py", "app/services/__init__.py"),
    ("backend/app/services/classifier.py", "app/services/classifier.py"),
    ("backend/app/services/storage.py", "app/services/storage.py"),
    # scripts/
    ("backend/scripts/train_v2.py", "scripts/train_v2.py"),
    ("backend/scripts/baselines_v3.py", "scripts/baselines_v3.py"),
    ("backend/scripts/gates_c_a_c_b_c_c.py", "scripts/gates_c_a_c_b_c_c.py"),
    # tools/base_data/
    ("tools/base_data/__init__.py", "tools/base_data/__init__.py"),
    ("tools/base_data/stream_pgn.py", "tools/base_data/stream_pgn.py"),
    ("tools/base_data/filter.py", "tools/base_data/filter.py"),
    ("tools/base_data/build_shards.py", "tools/base_data/build_shards.py"),
    # config/
    ("config/cohorts.yaml", "config/cohorts.yaml"),
]

DISALLOWED_DIRS = {"data", "docs", "tests", "__pycache__"}


def print_tree(root_dir: Path) -> int:
    file_count = 0
    print(f"{root_dir.name}/")
    for path in sorted(root_dir.rglob("*")):
        rel = path.relative_to(root_dir)
        depth = len(rel.parts)
        indent = "  " * depth
        if path.is_dir():
            print(f"{indent}{path.name}/")
        else:
            file_count += 1
            print(f"{indent}{path.name} ({path.stat().st_size} bytes)")
    print(f"\nTotal files written: {file_count}")
    return file_count


def prepare_dataset(out_dir: Path, make_tar: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target_base = out_dir / "base-data-tools"

    # Validate that no disallowed paths are in destination
    for src_rel, dst_rel in FILES_TO_BUNDLE:
        dst_parts = set(Path(dst_rel).parts)
        if dst_parts & DISALLOWED_DIRS:
            print(f"ERROR: Disallowed directory in target path: {dst_rel}", file=sys.stderr)
            sys.exit(1)

    # Validate all source files exist
    missing: list[Path] = []
    for src_rel, _ in FILES_TO_BUNDLE:
        src_path = repo_root / src_rel
        if not src_path.is_file():
            missing.append(src_path)

    if missing:
        print("ERROR: Missing required source files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    # Clean and create target_base
    if target_base.exists():
        shutil.rmtree(target_base)
    target_base.mkdir(parents=True, exist_ok=True)

    # Copy files
    for src_rel, dst_rel in FILES_TO_BUNDLE:
        src_path = repo_root / src_rel
        dst_path = target_base / dst_rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    # Print tree and file count
    print(f"Prepared Kaggle dataset directory at: {target_base}")
    print_tree(target_base)

    # Create tar.gz if requested
    if make_tar:
        tar_path = out_dir.parent / f"{out_dir.name}.tar.gz"
        print(f"\nCreating tar archive: {tar_path}")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(target_base, arcname="base-data-tools")
        print(f"Wrote archive ({tar_path.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle dataset package.")
    parser.add_argument(
        "--out-dir",
        default="build/kaggle_dataset",
        help="Output directory path (default: build/kaggle_dataset)",
    )
    parser.add_argument(
        "--tar",
        action="store_true",
        help="Also write .tar.gz archive alongside out-dir",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    prepare_dataset(out_dir, make_tar=args.tar)


if __name__ == "__main__":
    main()
