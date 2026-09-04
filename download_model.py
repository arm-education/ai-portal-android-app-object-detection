#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


SUPPORTED_MODELS = {
    "Arm/yolov5s-int8-xnnpack-executorch": "yolov5s_raspberry_executorch_optimized.pte",
    "Arm/yolov8s-int8-xnnpack-executorch": "yolov8s_raspberry_executorch_optimized.pte",
    "Arm/yolov9s-int8-xnnpack-executorch": "yolov9s_raspberry_executorch_optimized.pte",
}


def model_directory(output_directory: Path, model_id: str) -> Path:
    return output_directory / model_id.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an ExecuTorch model file for Scene Detector."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--filename",
        help="Model filename when the repository contains multiple .pte files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Parent directory for downloaded models (default: models)",
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print only the downloaded model path to standard output",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    destination = model_directory(args.output_dir, args.repo_id)
    output_stream = sys.stderr if args.print_path else sys.stdout
    filename = args.filename or SUPPORTED_MODELS.get(args.repo_id)
    if filename is not None:
        print(
            f"Downloading {args.repo_id}/{filename} to {destination} ...",
            file=output_stream,
        )
        downloaded_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                local_dir=destination,
            )
        )
    else:
        print(f"Downloading {args.repo_id} to {destination} ...", file=output_stream)
        snapshot_path = Path(
            snapshot_download(repo_id=args.repo_id, local_dir=destination)
        )
        candidates = sorted(
            path
            for path in snapshot_path.rglob("*.pte")
            if path.is_file()
        )
        if not candidates:
            raise SystemExit("The repository does not contain a .pte model file.")
        if len(candidates) > 1:
            choices = ", ".join(path.name for path in candidates)
            raise SystemExit(
                "The repository contains multiple .pte files. "
                f"Run the command again with --filename. Found: {choices}"
            )
        downloaded_path = candidates[0]
    if not downloaded_path.is_file():
        raise SystemExit(f"The downloaded model file was not found: {downloaded_path}")

    resolved_path = downloaded_path.resolve()
    if args.print_path:
        print(resolved_path)
    else:
        print(f"Model file: {resolved_path}")


if __name__ == "__main__":
    main()
