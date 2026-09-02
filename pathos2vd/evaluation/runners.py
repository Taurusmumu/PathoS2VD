from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wasserstein_distance

from .io import generated_middle_index, generated_sample_dir, load_generated_stack, load_gt_stack
from .metrics import LegacyImageMetrics
from .structure_tensor import evaluate_st_pair, local_z_axis_descriptors, vz_descriptor


def _summary(rows: list[dict[str, Any]], *, skip: set[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    numeric = table.select_dtypes(include=[np.number]).columns.difference(list(skip))
    return pd.DataFrame([{"statistic": "mean", **table[numeric].mean().to_dict()}, {"statistic": "std", **table[numeric].std().to_dict()}])


def _source_records(gt_csv: str | Path) -> list[dict[str, str]]:
    table = pd.read_csv(gt_csv)
    required = {"slide_name", "patch_name"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{gt_csv} is missing required columns: {sorted(missing)}")
    return table.loc[:, ["slide_name", "patch_name"]].astype(str).to_dict("records")


def _st_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    st = config.get("structure_tensor", {})
    return {
        "sigma": float(st.get("sigma", 1.0)),
        "rho": float(st.get("rho", 2.0)),
        "spacing": tuple(float(value) for value in st.get("spacing", [0.3, 0.5, 0.5])),
        "anis_thresh": float(st.get("anis_thresh", 0.3)),
    }


def _methods(config: dict[str, Any]) -> dict[str, str]:
    if config.get("methods"):
        return {str(name): str(root) for name, root in config["methods"].items()}
    aliases = {
        "stage1": "stage1_generated_root",
        "conventional_ft": "conventional_ft_generated_root",
        "mfda": "mfda_generated_root",
    }
    methods = {name: str(config[key]) for name, key in aliases.items() if config.get(key) is not None}
    if not methods:
        raise ValueError("Provide methods or stage1_generated_root/conventional_ft_generated_root/mfda_generated_root")
    return methods


def _legacy_distribution_metrics(
    config: dict[str, Any],
    *,
    gt_frame_paths: list[str],
    generated_frame_paths: list[str],
    gt_videos: list[np.ndarray],
    generated_videos: list[np.ndarray],
) -> dict[str, float]:
    """Optional standalone FID/FVD; no legacy checkout is required."""
    options = config.get("legacy_distribution_metrics", {})
    if not options.get("fid", False) and not options.get("fvd", False):
        return {}
    from .distribution_metrics import fid, fvd
    device = config.get("device", "cuda")
    result: dict[str, float] = {}
    if options.get("fid", False):
        limit = int(options.get("fid_num_samples", 10000))
        result["fid"] = fid(gt_frame_paths[:limit], generated_frame_paths[:limit], device=device)
    if options.get("fvd", False):
        gt_tensor = torch.from_numpy(np.stack(gt_videos)).permute(0, 1, 4, 2, 3).float() / 255.0
        generated_tensor = torch.from_numpy(np.stack(generated_videos)).permute(0, 1, 4, 2, 3).float() / 255.0
        result["fvd"] = fvd(gt_tensor, generated_tensor, device=device, i3d_weights=options.get("i3d_weights"))
    return result


def evaluate_source_root(config: dict[str, Any], generated_root: str | Path, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Evaluate one model root against held-out GT; always use central generated frames by default."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gt_root = Path(config["gt_root"])
    layout = config.get("generated_frame_layout", "central_33")
    device = config.get("device", "cuda")
    image_metrics = LegacyImageMetrics(device=device, use_lpips=bool(config.get("use_lpips", True)))
    frame_rows: list[dict[str, Any]] = []
    stack_rows: list[dict[str, Any]] = []
    st_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    gt_frame_paths: list[str] = []
    generated_frame_paths: list[str] = []
    gt_videos: list[np.ndarray] = []
    generated_videos: list[np.ndarray] = []
    for record in _source_records(config["gt_csv"]):
        slide, patch = record["slide_name"], record["patch_name"]
        gt_dir = generated_sample_dir(gt_root, slide, patch)
        generated_dir = generated_sample_dir(generated_root, slide, patch)
        sample_id = f"{slide}/{Path(patch).stem}"
        try:
            gt_rgb = load_gt_stack(gt_dir)
            generated_rgb = load_generated_stack(generated_dir, generated_frame_layout=layout)
            gt_videos.append(gt_rgb)
            generated_videos.append(generated_rgb)
            gt_frame_paths.extend(str(gt_dir / f"z{index:02d}.png") for index in range(11))
            generated_start = 11 if layout == "central_33" else 0
            generated_frame_paths.extend(str(generated_dir / f"z{generated_start + index:02d}.png") for index in range(11))
            per_frame, stack = image_metrics.paired_stack(gt_rgb, generated_rgb)
            stack_rows.append({"sample_id": sample_id, "slide_name": slide, "patch_name": patch, **stack})
            for row in per_frame:
                offset = int(row.pop("frame_offset"))
                frame_rows.append(
                    {
                        "sample_id": sample_id,
                        "slide_name": slide,
                        "patch_name": patch,
                        "frame_offset": offset,
                        "gt_frame": f"z{offset:02d}",
                        "generated_frame": f"z{(11 + offset) if layout == 'central_33' else offset:02d}",
                        **row,
                    }
                )
            gt_gray = load_gt_stack(gt_dir, grayscale=True).astype(np.float32) / 255.0
            generated_gray = load_generated_stack(generated_dir, generated_frame_layout=layout, grayscale=True).astype(np.float32) / 255.0
            st_rows.append({"sample_id": sample_id, "slide_name": slide, "patch_name": patch, **evaluate_st_pair(gt_gray, generated_gray, **_st_kwargs(config))})
        except Exception as error:  # Keep incomplete generated sets auditable.
            failures.append({"sample_id": sample_id, "slide_name": slide, "patch_name": patch, "reason": type(error).__name__, "details": str(error)})
    frames = pd.DataFrame(frame_rows)
    stacks = pd.DataFrame(stack_rows)
    st_table = pd.DataFrame(st_rows)
    frames.to_csv(output / "per_frame_metrics.csv", index=False)
    stacks.to_csv(output / "per_stack_metrics.csv", index=False)
    _summary(stack_rows, skip={"sample_id"}).to_csv(output / "summary.csv", index=False)
    st_table.to_csv(output / "st_per_stack.csv", index=False)
    _summary(st_rows, skip={"sample_id"}).to_csv(output / "st_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failed_samples.csv", index=False)
    pd.DataFrame([_legacy_distribution_metrics(
        config,
        gt_frame_paths=gt_frame_paths,
        generated_frame_paths=generated_frame_paths,
        gt_videos=gt_videos,
        generated_videos=generated_videos,
    )]).to_csv(output / "legacy_distribution_metrics.csv", index=False)
    return {"per_frame": frames, "per_stack": stacks, "st_per_stack": st_table, "failures": pd.DataFrame(failures)}


def _sample_stacks(
    candidates: list[tuple[str, Path]],
    *,
    count: int,
    rng: np.random.Generator,
    loader: Any,
    st_kwargs: dict[str, Any],
) -> tuple[list[tuple[str, np.ndarray]], list[dict[str, str]]]:
    """Randomly select exactly ``count`` readable 11-plane volumes, without pairing."""
    if count <= 0:
        raise ValueError("num_stacks must be positive")
    if len(candidates) < count:
        raise ValueError(f"Requested {count} stacks, but only {len(candidates)} candidates were found")

    accepted: list[tuple[str, np.ndarray]] = []
    failures: list[dict[str, str]] = []
    for index in rng.permutation(len(candidates)):
        sample_id, directory = candidates[int(index)]
        try:
            volume = loader(directory)
            if volume.shape[0] != 11:
                raise ValueError(f"Expected exactly 11 frames, found {volume.shape[0]}")
            accepted.append((sample_id, vz_descriptor(volume.astype(np.float32) / 255.0, **st_kwargs)))
            if len(accepted) == count:
                break
        except Exception as error:
            failures.append({"sample_id": sample_id, "directory": str(directory), "reason": type(error).__name__, "details": str(error)})
    if len(accepted) != count:
        raise RuntimeError(f"Only {len(accepted)} readable stacks were available; requested {count}")
    return accepted, failures


def _descriptor_rows(samples: list[tuple[str, np.ndarray]]) -> list[dict[str, Any]]:
    return [
        {"sample_id": sample_id, "descriptor_name": "abs_v1_z", "descriptor_value": float(value)}
        for sample_id, values in samples
        for value in values
    ]


def _distribution_statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def evaluate_st_distribution(config: dict[str, Any]) -> pd.DataFrame:
    """Compare independently sampled real and generated ST descriptor distributions.

    This is deliberately unpaired: stack identity never enters the W1
    calculation.  Each distribution consists of all valid-voxel ``|v1_z|``
    values from its independently sampled 11-frame stacks.
    """
    mode = config.get("mode", "full")
    if mode == "local":
        return evaluate_local_st_distribution(config)
    if mode != "full":
        raise ValueError("mode must be 'full' or 'local'")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    count = int(config.get("num_stacks", 100))
    rng = np.random.default_rng(int(config.get("random_seed", 0)))
    st_kwargs = _st_kwargs(config)
    layout = config.get("generated_frame_layout", "central_33")

    gt_candidates = [
        (f"{record['slide_name']}/{Path(record['patch_name']).stem}", generated_sample_dir(config["gt_root"], record["slide_name"], record["patch_name"]))
        for record in _source_records(config["gt_csv"])
    ]
    generated_root = Path(config["generated_root"])
    marker = 11 if layout == "central_33" else 0
    # Discovery identifies stack directories only.  Frame ordering is always
    # explicit in load_generated_stack(), never inferred from directory lists.
    generated_candidates = sorted(
        {(path.parent.relative_to(generated_root).as_posix(), path.parent) for path in generated_root.rglob(f"z{marker:02d}.png")},
        key=lambda item: item[0],
    )

    gt_samples, gt_failures = _sample_stacks(
        gt_candidates,
        count=count,
        rng=rng,
        loader=lambda directory: load_gt_stack(directory, grayscale=True),
        st_kwargs=st_kwargs,
    )
    generated_samples, generated_failures = _sample_stacks(
        generated_candidates,
        count=count,
        rng=rng,
        loader=lambda directory: load_generated_stack(directory, generated_frame_layout=layout, grayscale=True),
        st_kwargs=st_kwargs,
    )

    gt_values = np.concatenate([values for _, values in gt_samples])
    generated_values = np.concatenate([values for _, values in generated_samples])
    # pd.DataFrame(_descriptor_rows(gt_samples)).to_csv(output / "gt_st_descriptors.csv", index=False)
    # pd.DataFrame(_descriptor_rows(generated_samples)).to_csv(output / "generated_st_descriptors.csv", index=False)
    # pd.DataFrame(gt_failures).to_csv(output / "failed_gt_stacks.csv", index=False)
    # pd.DataFrame(generated_failures).to_csv(output / "failed_generated_stacks.csv", index=False)

    summary = pd.DataFrame([{
        "num_gt_stacks": len(gt_samples),
        "num_generated_stacks": len(generated_samples),
        "descriptor": "abs_v1_z",
        "wasserstein_distance": float(wasserstein_distance(gt_values, generated_values)),
        **_distribution_statistics(gt_values, "gt"),
        **_distribution_statistics(generated_values, "generated"),
    }])
    summary.to_csv(output / "st_distribution_summary.csv", index=False)
    return summary


def _candidate_sets(config: dict[str, Any]) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], str]:
    layout = config.get("generated_frame_layout", "central_33")
    gt_candidates = [
        (f"{record['slide_name']}/{Path(record['patch_name']).stem}", generated_sample_dir(config["gt_root"], record["slide_name"], record["patch_name"]))
        for record in _source_records(config["gt_csv"])
    ]
    generated_root = Path(config["generated_root"])
    marker = 11 if layout == "central_33" else 0
    generated_candidates = sorted(
        {(path.parent.relative_to(generated_root).as_posix(), path.parent) for path in generated_root.rglob(f"z{marker:02d}.png")},
        key=lambda item: item[0],
    )
    return gt_candidates, generated_candidates, layout


def _sample_volumes(
    candidates: list[tuple[str, Path]], *, count: int, rng: np.random.Generator, loader: Any
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    if len(candidates) < count:
        raise ValueError(f"Requested {count} stacks, but only {len(candidates)} candidates were found")
    accepted: dict[str, np.ndarray] = {}
    failures: list[dict[str, str]] = []
    for index in rng.permutation(len(candidates)):
        sample_id, directory = candidates[int(index)]
        try:
            volume = loader(directory)
            if volume.shape[0] != 11:
                raise ValueError(f"Expected exactly 11 frames, found {volume.shape[0]}")
            accepted[sample_id] = volume
            if len(accepted) == count:
                break
        except Exception as error:
            failures.append({"sample_id": sample_id, "directory": str(directory), "reason": type(error).__name__, "details": str(error)})
    if len(accepted) != count:
        raise RuntimeError(f"Only {len(accepted)} readable stacks were available; requested {count}")
    return accepted, failures


def _crop_volume(volume: np.ndarray, *, x: int, y: int, crop_size: int) -> np.ndarray:
    if x < 0 or y < 0 or x + crop_size > volume.shape[2] or y + crop_size > volume.shape[1]:
        raise ValueError(f"Crop x={x}, y={y}, size={crop_size} is outside volume {tuple(volume.shape)}")
    crop = volume[:, y:y + crop_size, x:x + crop_size]
    if crop.shape != (11, crop_size, crop_size):
        raise RuntimeError(f"Unexpected local crop shape {crop.shape}")
    return crop


def _tissue_fraction(crop: np.ndarray, threshold: float) -> float:
    """Simple optional bright-background filter; disabled unless a minimum is set."""
    return float(np.mean(crop.astype(np.float32) / 255.0 < threshold))


def _new_local_manifest(
    volumes: dict[str, np.ndarray], *, dataset_type: str, local: dict[str, Any], rng: np.random.Generator
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    size, crops_per_stack = int(local.get("crop_size", 50)), int(local.get("crops_per_stack", 1))
    if local.get("sampling", "random") != "random":
        raise ValueError("local_crop.sampling currently supports only 'random'")
    minimum = local.get("min_tissue_fraction")
    minimum = None if minimum is None else float(minimum)
    threshold, maximum_attempts = float(local.get("tissue_intensity_threshold", 0.9)), int(local.get("max_attempts_per_crop", 100))
    accepted: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for stack_id, volume in volumes.items():
        if size > volume.shape[1] or size > volume.shape[2]:
            raise ValueError(f"crop_size {size} exceeds stack {stack_id!r} shape {tuple(volume.shape)}")
        for crop_id in range(crops_per_stack):
            for attempt in range(maximum_attempts):
                x, y = int(rng.integers(0, volume.shape[2] - size + 1)), int(rng.integers(0, volume.shape[1] - size + 1))
                crop = _crop_volume(volume, x=x, y=y, crop_size=size)
                fraction = _tissue_fraction(crop, threshold) if minimum is not None else np.nan
                ok = minimum is None or fraction >= minimum
                row = {"dataset_type": dataset_type, "stack_id": stack_id, "crop_id": crop_id, "x": x, "y": y, "width": size, "height": size, "accepted": ok, "tissue_fraction_if_available": fraction, "attempt": attempt}
                all_rows.append(row)
                if ok:
                    accepted.append(row)
                    break
            else:
                all_rows.append({"dataset_type": dataset_type, "stack_id": stack_id, "crop_id": crop_id, "x": np.nan, "y": np.nan, "width": size, "height": size, "accepted": False, "tissue_fraction_if_available": np.nan, "attempt": maximum_attempts, "reason": "tissue_filter_exhausted"})
    return accepted, all_rows


def _manifest_local_rows(path: str | Path) -> list[dict[str, Any]]:
    table = pd.read_csv(path)
    required = {"dataset_type", "stack_id", "crop_id", "x", "y", "width", "height", "accepted"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Sampling manifest is missing columns: {sorted(missing)}")
    return table.loc[table["accepted"].astype(bool)].to_dict("records")


def _local_descriptor_rows(
    dataset_type: str, stack_id: str, crop: dict[str, Any], volume: np.ndarray, st_kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    local_volume = _crop_volume(volume, x=int(crop["x"]), y=int(crop["y"]), crop_size=int(crop["width"]))
    descriptors = local_z_axis_descriptors(local_volume.astype(np.float32) / 255.0, **st_kwargs)
    rows: list[dict[str, Any]] = []
    for descriptor, (z_indices, values) in descriptors.items():
        rows.extend({"dataset_type": dataset_type, "stack_id": stack_id, "crop_id": int(crop["crop_id"]), "x": int(crop["x"]), "y": int(crop["y"]), "crop_size": int(crop["width"]), "descriptor": descriptor, "z_index": int(z_index), "value": float(value)} for z_index, value in zip(z_indices, values))
    return rows


def evaluate_local_st_distribution(config: dict[str, Any]) -> pd.DataFrame:
    """Unpaired local-3D ST distribution evaluation; full-stack code is untouched."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    local = dict(config.get("local_crop", {}))
    if not local.get("enabled", True):
        raise ValueError("mode: local requires local_crop.enabled: true")
    count = int(config.get("num_stacks", 100))
    rng = np.random.default_rng(int(local.get("seed", config.get("random_seed", 0))))
    gt_candidates, generated_candidates, layout = _candidate_sets(config)
    sampling_path = config.get("sampling_manifest")
    if sampling_path:
        manifest_rows = _manifest_local_rows(sampling_path)
        gt_rows = [row for row in manifest_rows if row["dataset_type"] == "gt"]
        generated_rows = [row for row in manifest_rows if row["dataset_type"] == "generated"]
        gt_ids, generated_ids = {str(row["stack_id"]) for row in gt_rows}, {str(row["stack_id"]) for row in generated_rows}
        gt_map, generated_map = dict(gt_candidates), dict(generated_candidates)
        gt_volumes, gt_failures = _sample_volumes([(key, gt_map[key]) for key in sorted(gt_ids) if key in gt_map], count=len(gt_ids), rng=rng, loader=lambda directory: load_gt_stack(directory, grayscale=True))
        generated_volumes, generated_failures = _sample_volumes([(key, generated_map[key]) for key in sorted(generated_ids) if key in generated_map], count=len(generated_ids), rng=rng, loader=lambda directory: load_generated_stack(directory, generated_frame_layout=layout, grayscale=True))
        accepted_rows, all_manifest_rows = gt_rows + generated_rows, manifest_rows
    else:
        gt_volumes, gt_failures = _sample_volumes(gt_candidates, count=count, rng=rng, loader=lambda directory: load_gt_stack(directory, grayscale=True))
        generated_volumes, generated_failures = _sample_volumes(generated_candidates, count=count, rng=rng, loader=lambda directory: load_generated_stack(directory, generated_frame_layout=layout, grayscale=True))
        gt_accepted, gt_manifest = _new_local_manifest(gt_volumes, dataset_type="gt", local=local, rng=rng)
        generated_accepted, generated_manifest = _new_local_manifest(generated_volumes, dataset_type="generated", local=local, rng=rng)
        accepted_rows, all_manifest_rows = gt_accepted + generated_accepted, gt_manifest + generated_manifest

    volume_sets = {"gt": gt_volumes, "generated": generated_volumes}
    descriptor_rows: list[dict[str, Any]] = []
    descriptor_failures: list[dict[str, Any]] = []
    for crop in accepted_rows:
        dataset_type, stack_id = str(crop["dataset_type"]), str(crop["stack_id"])
        try:
            descriptor_rows.extend(_local_descriptor_rows(dataset_type, stack_id, crop, volume_sets[dataset_type][stack_id], _st_kwargs(config)))
        except Exception as error:
            descriptor_failures.append({"dataset_type": dataset_type, "stack_id": stack_id, "crop_id": crop["crop_id"], "reason": type(error).__name__, "details": str(error)})
    descriptors = pd.DataFrame(descriptor_rows)
    crops = pd.DataFrame(all_manifest_rows)
    descriptors.to_csv(output / "local_st_descriptors.csv", index=False)
    crops.to_csv(output / "local_crop_manifest.csv", index=False)
    if config.get("save_sampling_manifest"):
        saved_manifest = Path(config["save_sampling_manifest"])
        saved_manifest.parent.mkdir(parents=True, exist_ok=True)
        crops.to_csv(saved_manifest, index=False)
    pd.DataFrame(gt_failures + generated_failures + descriptor_failures).to_csv(output / "failed_local_st_samples.csv", index=False)

    labels = {"vz": "Vz Wasserstein", "st_evolution": "ST Evolution Wasserstein", "orientation_change": "Orientation-Change Wasserstein", "anisotropy_change": "Anisotropy-Change Wasserstein", "st_evolution_roughness": "ST Evolution Roughness Wasserstein"}
    summary_rows: list[dict[str, Any]] = []
    for descriptor, metric in labels.items():
        gt_values = descriptors.loc[(descriptors["dataset_type"] == "gt") & (descriptors["descriptor"] == descriptor), "value"].to_numpy()
        generated_values = descriptors.loc[(descriptors["dataset_type"] == "generated") & (descriptors["descriptor"] == descriptor), "value"].to_numpy()
        if not len(gt_values) or not len(generated_values):
            raise RuntimeError(f"No descriptor values available for {descriptor}")
        summary_rows.append({"metric": metric, "gt_num_stacks": len(gt_volumes), "generated_num_stacks": len(generated_volumes), "gt_num_crops": int(((crops["dataset_type"] == "gt") & crops["accepted"]).sum()), "generated_num_crops": int(((crops["dataset_type"] == "generated") & crops["accepted"]).sum()), "gt_num_observations": len(gt_values), "generated_num_observations": len(generated_values), "gt_mean": float(np.mean(gt_values)), "gt_std": float(np.std(gt_values)), "generated_mean": float(np.mean(generated_values)), "generated_std": float(np.std(generated_values)), "wasserstein_distance": float(wasserstein_distance(gt_values, generated_values))})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "local_st_distribution_summary.csv", index=False)
    return summary


def _target_records(config: dict[str, Any]) -> list[dict[str, str]]:
    manifest = config["target_manifest"]
    if isinstance(manifest, (str, Path)):
        manifest = {"path": manifest}
    table = pd.read_excel(manifest["path"]) if Path(manifest["path"]).suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(manifest["path"])
    # AGGC manifests contain ``file_path`` and ``class`` only.  A separate
    # sample-id column is therefore optional; the path itself is sufficient
    # to reproduce the generated-stack directory below.
    id_column = manifest.get("sample_id_column")
    path_column = manifest.get("input_path_column", "file_path")
    if path_column not in table:
        raise ValueError(f"Target manifest needs the input-path column {path_column!r}")
    if id_column is not None and id_column not in table:
        raise ValueError(f"Configured sample-id column {id_column!r} is absent from the target manifest")
    generated_column = manifest.get("generated_dir_column")
    records: list[dict[str, str]] = []
    for _, row in table.iterrows():
        input_path = Path(str(row[path_column]))
        # ``sample_id`` is only an output identifier.  It does not affect the
        # generated-path mapping or evaluation pairing.
        sample_id = str(row[id_column]) if id_column is not None else input_path.as_posix()
        record = {"sample_id": sample_id, "input_path": str(input_path)}
        if generated_column is not None and pd.notna(row[generated_column]):
            record["generated_dir"] = str(row[generated_column])
        records.append(record)
    return records


def _target_image(path: str | Path, preprocessing: dict[str, Any]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        crop = preprocessing.get("center_crop")
        if crop is not None:
            crop = int(crop)
            left, top = max((image.width - crop) // 2, 0), max((image.height - crop) // 2, 0)
            image = image.crop((left, top, min(left + crop, image.width), min(top + crop, image.height)))
        resize = preprocessing.get("resize")
        if resize is not None:
            bicubic = getattr(Image, "Resampling", Image).BICUBIC
            image = image.resize((int(resize), int(resize)), bicubic)
        return np.asarray(image, dtype=np.uint8)


def _target_generated_dir(record: dict[str, str], root: Path, manifest: dict[str, Any]) -> Path:
    if "generated_dir" in record:
        return root / record["generated_dir"]
    input_root = manifest.get("input_root")
    if input_root is None:
        raise ValueError(
            "target_manifest requires input_root for file_path-only manifests "
            "(or generated_dir_column for an explicit mapping)"
        )
    try:
        relative = Path(record["input_path"]).relative_to(Path(input_root))
    except ValueError as error:
        raise ValueError(
            f"Target input {record['input_path']!r} is not below input_root {str(input_root)!r}"
        ) from error
    # Example: <input_root>/Subset1_Train_image/Subset1_Train_xx/patch.jpg
    #       -> <generated_root>/Subset1_Train_image/Subset1_Train_xx/patch/
    return root / relative.parent / relative.stem


def evaluate_target_root(config: dict[str, Any], generated_root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = config["target_manifest"] if isinstance(config["target_manifest"], dict) else {"path": config["target_manifest"]}
    layout = config.get("generated_frame_layout", "central_33")
    image_metrics = LegacyImageMetrics(device=config.get("device", "cuda"), use_lpips=bool(config.get("use_lpips", True)))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in _target_records(config):
        try:
            generated_dir = _target_generated_dir(record, Path(generated_root), manifest)
            generated = load_generated_stack(generated_dir, generated_frame_layout=layout)
            input_image = _target_image(record["input_path"], config.get("preprocessing", {}))
            middle = generated[5]
            rows.append({"sample_id": record["sample_id"], "input_path": record["input_path"], "generated_frame": f"z{generated_middle_index(layout):02d}", **image_metrics.target_central_frame(input_image, middle)})
        except Exception as error:
            failures.append({"sample_id": record["sample_id"], "input_path": record["input_path"], "reason": type(error).__name__, "details": str(error)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "per_image_metrics.csv", index=False)
    _summary(rows, skip={"sample_id", "input_path"}).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failed_samples.csv", index=False)
    return table


def evaluate_prior_retention(config: dict[str, Any]) -> None:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []
    failures: list[pd.DataFrame] = []
    for method, root in _methods(config).items():
        result = evaluate_source_root(config, root, output / method)
        paired = result["per_stack"].merge(result["st_per_stack"], on=["sample_id", "slide_name", "patch_name"], how="outer")
        paired.insert(0, "method", method)
        all_rows.append(paired)
        if not paired.empty:
            numeric = paired.select_dtypes(include=[np.number]).mean().to_dict()
            all_summaries.append({"method": method, "num_samples": len(paired), **numeric})
        if not result["failures"].empty:
            method_failures = result["failures"].copy()
            method_failures.insert(0, "method", method)
            failures.append(method_failures)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    result.to_csv(output / "source_prior_retention.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(output / "source_prior_retention_summary.csv", index=False)
    pd.concat(failures, ignore_index=True).to_csv(output / "failed_samples.csv", index=False) if failures else pd.DataFrame().to_csv(output / "failed_samples.csv", index=False)


def evaluate_prior_drift(config: dict[str, Any]) -> None:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = config["target_manifest"] if isinstance(config["target_manifest"], dict) else {"path": config["target_manifest"]}
    layout = config.get("generated_frame_layout", "central_33")
    methods = _methods(config)
    descriptors: dict[str, dict[str, np.ndarray]] = {method: {} for method in methods}
    failures: list[dict[str, str]] = []
    for record in _target_records(config):
        for method, root in methods.items():
            try:
                directory = _target_generated_dir(record, Path(root), manifest)
                volume = load_generated_stack(directory, generated_frame_layout=layout, grayscale=True).astype(np.float32) / 255.0
                descriptors[method][record["sample_id"]] = vz_descriptor(volume, **_st_kwargs(config))
            except Exception as error:
                failures.append({"sample_id": record["sample_id"], "method": method, "reason": type(error).__name__, "details": str(error)})
    reference = config.get("reference_method", "stage1")
    if reference not in descriptors:
        raise ValueError(f"reference_method {reference!r} is absent from methods")
    with (output / "prior_drift_descriptors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "method", "descriptor_name", "descriptor_value"])
        writer.writeheader()
        for method, samples in descriptors.items():
            for sample_id, values in samples.items():
                writer.writerows({"sample_id": sample_id, "method": method, "descriptor_name": "abs_v1_z", "descriptor_value": float(value)} for value in values)
    summary: list[dict[str, Any]] = []
    for method, samples in descriptors.items():
        if method == reference:
            continue
        common = sorted(set(descriptors[reference]) & set(samples))
        if not common:
            distance = float("nan")
            count = 0
        else:
            reference_values = np.concatenate([descriptors[reference][sample_id] for sample_id in common])
            target_values = np.concatenate([samples[sample_id] for sample_id in common])
            distance = float(wasserstein_distance(reference_values, target_values))
            count = len(common)
        summary.append({"method": method, "reference": reference, "metric": "st_prior_drift_w1", "wasserstein_distance": distance, "num_samples": count})
    pd.DataFrame(summary).to_csv(output / "prior_drift_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failed_samples.csv", index=False)
