#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


MODEL_SUFFIXES = {".onnx", ".pte", ".pt", ".pth", ".tflite"}
FILENAME_PATTERN = re.compile(r"(?m)^filename\s*:\s*['\"]?([^'\"\n#]+)")


def model_directory(output_directory: Path, model_id: str) -> Path:
    return output_directory / model_id.replace("/", "__")


def primary_model_path(directory: Path, filename: str | None = None) -> Path:
    if filename is not None:
        candidate = (directory / filename).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as exception:
            raise SystemExit(
                f"The requested model filename is outside the package directory: {filename}"
            ) from exception
        if not candidate.is_file():
            raise SystemExit(f"The requested model file was not found: {candidate}")
        return candidate

    for metadata_name in ("metadata.yaml", "metadata.yml"):
        metadata_path = directory / metadata_name
        if not metadata_path.is_file():
            continue
        match = FILENAME_PATTERN.search(
            metadata_path.read_text(encoding="utf-8", errors="replace")
        )
        if not match:
            continue
        candidate = (directory / match.group(1).strip()).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as exception:
            raise SystemExit(
                f"The model filename in {metadata_path} is outside the package directory."
            ) from exception
        if candidate.is_file():
            return candidate
        break

    candidates = [
        path.resolve()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit("The downloaded package does not identify a model binary.")

    optimized_candidates = [
        path for path in candidates if "optimized" in path.stem.lower()
    ]
    if len(optimized_candidates) == 1:
        return optimized_candidates[0]

    raise SystemExit(
        "The downloaded package contains multiple model binaries and does not identify "
        "one optimized file. Run again with --filename."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a complete model package for adapter generation."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--filename",
        help="Primary model filename when package metadata is missing or stale",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--print-model-path",
        action="store_true",
        help="Print the primary model binary instead of the package directory.",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    destination = model_directory(args.output_dir, args.repo_id)
    downloaded = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=destination,
    )
    downloaded_directory = Path(downloaded).resolve()
    if args.print_model_path:
        print(primary_model_path(downloaded_directory, args.filename))
    else:
        print(downloaded_directory)


if __name__ == "__main__":
    main()
