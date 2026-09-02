#!/usr/bin/env python3
"""Build one PathoS2VD ``blur_motion_data6.csv`` from a raw z-stack root.

Expected layout: ``ROOT/slide/z00...z18/patch.png``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PLANES = [f"z{i:02d}" for i in range(19)]


def find_stacks(root: Path) -> list[tuple[str, str]]:
    """Return patches that really exist in all 19 physical planes."""
    stacks: list[tuple[str, str]] = []
    for slide_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        folders = [slide_dir / plane for plane in PLANES]
        if not all(folder.is_dir() for folder in folders):
            continue
        patch_names = set(path.name for path in folders[0].iterdir() if path.is_file())
        for folder in folders[1:]:
            patch_names &= set(path.name for path in folder.iterdir() if path.is_file())
        stacks.extend((slide_dir.name, patch) for patch in sorted(patch_names))
    return stacks


def blur_one_stack(task: tuple[str, str, str]) -> dict[str, float | str]:
    """Calculate the historical blur score for all 19 images of one patch."""
    from pathos2vd.preprocessing.blur_estimation import estimate_blur_score

    root, slide, patch = task
    row: dict[str, float | str] = {"slide_name": slide, "patch_name": patch}
    for plane in PLANES:
        row[plane] = estimate_blur_score(str(Path(root) / slide / plane / patch))
    return row


def process_one_stack(root: Path, row: dict) -> dict | None:
    """Align to the sharpest image, measure motion, then apply tissue filtering."""
    import cv2
    from pathos2vd.preprocessing.alignment import align_to_reference
    from pathos2vd.preprocessing.motion_estimation import estimate_motion_from_frames
    from pathos2vd.preprocessing.tissue_filtering import tissue_fraction

    slide, patch, minimum = row["slide_name"], row["patch_name"], int(row["min_index"])
    frames = [cv2.imread(str(root / slide / plane / patch)) for plane in PLANES]
    if any(frame is None for frame in frames):
        return None
    reference = frames[minimum]
    aligned = [align_to_reference(reference, frame, max_iterations=1000, epsilon=1e-7)[0] for frame in frames]
    motion = estimate_motion_from_frames(reference, aligned)
    if -1.0 in motion:
        return None
    if tissue_fraction(str(root / slide / PLANES[minimum] / patch), low_s=50, high_v=200) < 0.5:
        return None
    for index, plane in enumerate(PLANES):
        relative = row[f"relative_blur_{plane}"]
        row[plane] = f"{motion[index]};{relative}"
        row[f"motion_{plane}"] = motion[index]
    row["keep"] = True
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blur_motion_data.csv from ROOT/slide/z00...z18/patch.png")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    import pandas as pd
    from tqdm import tqdm
    from pathos2vd.preprocessing.blur_filtering import calibrate_relative_blur

    root = Path(args.source_root)
    output = Path(args.output_csv) if args.output_csv else root / "blur_motion_data.csv"
    tasks = [(str(root), slide, patch) for slide, patch in find_stacks(root)]
    if not tasks:
        raise SystemExit(f"No complete 19-plane stacks found in {root}")

    print(f"Found {len(tasks)} complete stacks. Calculating blur...")
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            blur_rows = list(tqdm(pool.map(blur_one_stack, tasks), total=len(tasks), desc="Blur"))
    else:
        blur_rows = [blur_one_stack(task) for task in tqdm(tasks, desc="Blur")]

    # These constants are the active settings in the historical PathoS2VD code.
    selected = calibrate_relative_blur(
        pd.DataFrame(blur_rows), PLANES,
        threshold=0.2, min_sharp_planes=5,
        absolute_min_low=0.6, absolute_min_high=1.0, std_multiplier=2.0,
    )
    print(f"{len(selected)} stacks remain after blur filtering. Calculating motion...")
    final_rows = []
    for row in tqdm(selected.to_dict("records"), desc="Motion"):
        try:
            result = process_one_stack(root, row)
            if result is not None:
                final_rows.append(result)
        except Exception:
            # The historical pipeline skips failed patches; the count is shown below.
            pass

    final = pd.DataFrame(sorted(final_rows, key=lambda row: (row["slide_name"], row["patch_name"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False)
    print(f"Done. Kept {len(final)} / {len(tasks)} stacks.")
    print(f"Final annotation: {output}")


if __name__ == "__main__":
    main()
