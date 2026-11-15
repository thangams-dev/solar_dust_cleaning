"""Create reproducible, duplicate-group-aware main and cracked data splits."""

from __future__ import annotations

import csv
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN_MANIFEST = ROOT / "data" / "patches_manifest.csv"
CRACKED_MANIFEST = ROOT / "data" / "cracked_patches_manifest.csv"
SPLIT_ROOT = ROOT / "splits"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
LOGGER = logging.getLogger("split_data")


@dataclass
class Group:
    group_id: str
    label: str
    parent_paths: set[str] = field(default_factory=set)
    parents_by_source: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    rows: list[dict[str, str]] = field(default_factory=list)


def load_groups(manifest_path: Path, labels: set[str]) -> dict[str, list[Group]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Patch manifest is missing: {manifest_path}")
    groups: dict[tuple[str, str], Group] = {}
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "kept" or row["label"] not in labels:
                continue
            group_key = (row["label"], row["group_id"])
            group = groups.setdefault(
                group_key, Group(group_id=row["group_id"], label=row["label"])
            )
            group.parent_paths.add(row["parent_path"])
            group.parents_by_source[row["source"]].add(row["parent_path"])
            group.rows.append(row)
    by_label: dict[str, list[Group]] = defaultdict(list)
    for group in groups.values():
        by_label[group.label].append(group)
    return by_label


def allocation_cost(
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, float]],
    overall_counts: dict[str, int],
    overall_targets: dict[str, float],
    candidate: str,
    group: Group,
) -> float:
    cost = sum(
        (
            (
                counts[source][name]
                + (len(group.parents_by_source.get(source, set())) if name == candidate else 0)
                - targets[source][name]
            )
            ** 2
        )
        / max(targets[source][name], 1.1)
        for source in targets
        for name in SPLIT_NAMES
    )
    return cost + 0.25 * sum(
        (
            (
                overall_counts[name]
                + (len(group.parent_paths) if name == candidate else 0)
                - overall_targets[name]
            )
            ** 2
        )
        / max(overall_targets[name], 1.0)
        for name in SPLIT_NAMES
    )


def assign_groups(groups: list[Group], rng: random.Random) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError(f"Need at least three duplicate groups to split; found {len(groups)}")
    randomized = groups.copy()
    rng.shuffle(randomized)
    source_group_counts = Counter(
        source for group in groups for source in group.parents_by_source
    )
    randomized.sort(
        key=lambda group: (
            min(source_group_counts[source] for source in group.parents_by_source),
            -len(group.parent_paths),
        )
    )

    parent_totals = Counter(
        source
        for group in groups
        for source, parent_paths in group.parents_by_source.items()
        for _ in parent_paths
    )
    targets = {
        source: {
            split_name: parent_total * SPLIT_RATIOS[split_name]
            for split_name in SPLIT_NAMES
        }
        for source, parent_total in parent_totals.items()
    }
    counts = {
        source: {split_name: 0 for split_name in SPLIT_NAMES}
        for source in targets
    }
    overall_parent_total = sum(len(group.parent_paths) for group in groups)
    overall_targets = {
        split_name: overall_parent_total * SPLIT_RATIOS[split_name]
        for split_name in SPLIT_NAMES
    }
    overall_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    assigned: dict[str, str] = {}

    for index, group in enumerate(randomized):
        if index < len(SPLIT_NAMES):
            chosen = SPLIT_NAMES[index]
        else:
            choices = list(SPLIT_NAMES)
            rng.shuffle(choices)
            chosen = min(
                choices,
                key=lambda split_name: allocation_cost(
                    counts,
                    targets,
                    overall_counts,
                    overall_targets,
                    split_name,
                    group,
                ),
            )
        assigned[group.group_id] = chosen
        overall_counts[chosen] += len(group.parent_paths)
        for source, parent_paths in group.parents_by_source.items():
            counts[source][chosen] += len(parent_paths)
    return assigned


def write_split_files(
    group_sets: dict[str, list[Group]],
    output_directory: Path,
    rng: random.Random,
    model_name: str,
) -> dict[str, dict[str, int]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for label, groups in sorted(group_sets.items()):
        assignments = assign_groups(groups, rng)
        for group in groups:
            split_name = assignments[group.group_id]
            for row in group.rows:
                row["split"] = split_name
                row["model"] = model_name
                rows_by_split[split_name].append(row)

        label_stats: dict[str, int] = {}
        for split_name in SPLIT_NAMES:
            rows = [
                row
                for group in groups
                for row in group.rows
                if row["split"] == split_name
            ]
            unique_parents = {row["parent_path"] for row in rows}
            label_stats[f"{split_name}_patches"] = len(rows)
            label_stats[f"{split_name}_parents"] = len(unique_parents)
            label_stats[f"{split_name}_groups"] = sum(
                assignments[group.group_id] == split_name for group in groups
            )
        stats[label] = label_stats
    for split_name in SPLIT_NAMES:
        write_rows(output_directory / f"{split_name}.csv", rows_by_split[split_name])
    return stats


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "patch_path",
        "parent_path",
        "group_id",
        "label",
        "source",
        "crop_index",
        "crop_area_fraction",
        "left",
        "top",
        "right",
        "bottom",
        "status",
        "drop_reasons",
        "split",
        "model",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_no_leakage(directory: Path) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    for split_name in SPLIT_NAMES:
        path = directory / f"{split_name}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                group_splits[row["group_id"]].add(split_name)
                parent_splits[row["parent_path"]].add(split_name)
    leaked_groups = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    leaked_parents = {parent: splits for parent, splits in parent_splits.items() if len(splits) > 1}
    if leaked_groups or leaked_parents:
        raise RuntimeError(
            f"Split leakage: groups={len(leaked_groups)} parents={len(leaked_parents)}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main_groups = load_groups(MAIN_MANIFEST, {"clean", "dusty"})
    cracked_groups = load_groups(CRACKED_MANIFEST, {"cracked"})

    main_output = SPLIT_ROOT / "main"
    cracked_output = SPLIT_ROOT / "cracked_experimental"
    main_stats = write_split_files(
        main_groups,
        main_output,
        random.Random(SEED),
        "main",
    )
    cracked_stats = write_split_files(
        cracked_groups,
        cracked_output,
        random.Random(SEED),
        "cracked_experimental",
    )
    verify_no_leakage(main_output)
    verify_no_leakage(cracked_output)

    for label, stats in sorted(main_stats.items()):
        LOGGER.info("main label=%s stats=%s", label, stats)
    for label, stats in sorted(cracked_stats.items()):
        LOGGER.info("cracked label=%s stats=%s", label, stats)
    cracked_test_parents = cracked_stats.get("cracked", {}).get("test_parents", 0)
    LOGGER.warning(
        "Experimental cracked test has %d parent images; only about 10 are expected, "
        "so its metrics are not reliable.",
        cracked_test_parents,
    )
    LOGGER.info("Main model labels: clean,dusty (cracked is not included)")
    LOGGER.info("Split files: %s", SPLIT_ROOT)


if __name__ == "__main__":
    main()
