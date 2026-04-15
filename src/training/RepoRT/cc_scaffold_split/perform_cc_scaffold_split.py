"""
Split RepoRT by chromatography condition and Murcko scaffold.

Current strategy:
    1. Build a weighted dir_id conflict problem from scaffold overlaps.
    2. Generate diverse greedy assignments through multiple restarts.
    3. Prune scaffolds that would be shared across splits.
    4. Relabel the final splits by retained size so that the largest split is train,
       the second is val and the third is test.
    5. Keep only solutions that satisfy minimum split sizes and a minimum train fraction.
"""

from itertools import combinations
from pathlib import Path
import os

import numpy as np
import pandas as pd

from src.training.RepoRT.scaffold_split.perform_scaffold_split import ms_split


SPLIT_NAMES = ("train", "val", "test")
SPLIT_TO_INDEX = {name: index for index, name in enumerate(SPLIT_NAMES)}
NUM_SPLITS = len(SPLIT_NAMES)
HOLDOUT_SPLITS = (SPLIT_TO_INDEX["val"], SPLIT_TO_INDEX["test"])
TARGET_SPLIT_RATIOS = np.array([0.8, 0.1, 0.1], dtype=np.float64)
MIN_RETAINED_MOLECULES_PER_SPLIT = 1000
MIN_DIR_IDS_PER_SPLIT = 5
MIN_TRAIN_RETAINED_FRACTION = 0.5
NUM_RESTARTS = 25
CONFLICT_BUCKET_DECIMALS = 3
SEED_CANDIDATE_POOL_SIZE = 5
ORDER_SHUFFLE_WINDOW = 8
MINIMUM_PHASE_SPLIT_POOL_SIZE = 2
BALANCE_PHASE_SPLIT_POOL_SIZE = 2


def _clip_minimum(requested_minimum, max_possible):
    """
    Clip a global minimum to something feasible on small synthetic datasets.
    """
    return max(1, min(int(requested_minimum), int(max_possible)))


def _choose_from_ranked_candidates(ranked_candidates, pool_size, rng):
    """
    Sample from the top-k ranked candidates with a bias towards better ranks.
    """
    top_k = max(1, min(int(pool_size), len(ranked_candidates)))
    candidate_pool = ranked_candidates[:top_k]
    weights = np.arange(top_k, 0, -1, dtype=np.float64)
    probabilities = weights / weights.sum()
    selected_position = int(rng.choice(np.arange(top_k), p=probabilities))
    return candidate_pool[selected_position]


def _build_split_problem(df):
    """
    Build the compact representation used to evaluate assignments quickly.
    """
    clean_df = df[df["ms_smiles"].notna() & (df["ms_smiles"] != "")].reset_index(drop=True).copy()
    if clean_df.empty:
        raise ValueError("The input dataframe does not contain valid ms_smiles values.")

    pair_counts_df = (
        clean_df.groupby(["dir_id", "ms_smiles"], sort=False)
        .size()
        .reset_index(name="count")
    )

    dir_labels = pair_counts_df["dir_id"].drop_duplicates().to_list()
    scaffold_labels = pair_counts_df["ms_smiles"].drop_duplicates().to_list()
    dir_to_idx = {dir_id: index for index, dir_id in enumerate(dir_labels)}
    scaffold_to_idx = {
        scaffold: index for index, scaffold in enumerate(scaffold_labels)
    }

    pair_dir_idx = pair_counts_df["dir_id"].map(dir_to_idx).to_numpy(dtype=np.int32)
    pair_scaffold_idx = pair_counts_df["ms_smiles"].map(scaffold_to_idx).to_numpy(dtype=np.int32)
    pair_count = pair_counts_df["count"].to_numpy(dtype=np.int64)

    num_dirs = len(dir_labels)
    num_scaffolds = len(scaffold_labels)
    total_size = int(pair_count.sum())

    dir_sizes = np.zeros(num_dirs, dtype=np.int64)
    np.add.at(dir_sizes, pair_dir_idx, pair_count)

    dir_scaffold_counts = [dict() for _ in range(num_dirs)]
    scaffold_dir_counts = {}
    for dir_idx, scaffold_idx, count in zip(pair_dir_idx, pair_scaffold_idx, pair_count):
        dir_scaffold_counts[dir_idx][scaffold_idx] = int(count)
        scaffold_dir_counts.setdefault(scaffold_idx, []).append((dir_idx, int(count)))

    private_molecule_counts = np.zeros(num_dirs, dtype=np.int64)
    for scaffold_idx, dir_counts in scaffold_dir_counts.items():
        if len(dir_counts) == 1:
            dir_idx, count = dir_counts[0]
            private_molecule_counts[dir_idx] += count

    dir_conflict_matrix = np.zeros((num_dirs, num_dirs), dtype=np.float64)
    for dir_counts in scaffold_dir_counts.values():
        n_dir_ids = len(dir_counts)
        if n_dir_ids < 2:
            continue
        scale = 1.0 / n_dir_ids
        for (left_dir_idx, left_count), (right_dir_idx, right_count) in combinations(dir_counts, 2):
            weight = min(left_count, right_count) * scale
            dir_conflict_matrix[left_dir_idx, right_dir_idx] += weight
            dir_conflict_matrix[right_dir_idx, left_dir_idx] += weight

    conflict_degree = dir_conflict_matrix.sum(axis=1)

    row_dir_idx = clean_df["dir_id"].map(dir_to_idx).to_numpy(dtype=np.int32)
    row_scaffold_idx = clean_df["ms_smiles"].map(scaffold_to_idx).to_numpy(dtype=np.int32)

    split_targets = TARGET_SPLIT_RATIOS * total_size
    min_holdout_target = int(np.ceil(np.min(split_targets[list(HOLDOUT_SPLITS)])))
    min_holdout_dir_target = int(np.ceil(num_dirs * np.min(TARGET_SPLIT_RATIOS[list(HOLDOUT_SPLITS)])))
    min_retained_molecules = _clip_minimum(
        MIN_RETAINED_MOLECULES_PER_SPLIT,
        min_holdout_target,
    )
    min_dir_ids = _clip_minimum(
        MIN_DIR_IDS_PER_SPLIT,
        min_holdout_dir_target,
    )

    return {
        "df": clean_df,
        "pair_dir_idx": pair_dir_idx,
        "pair_scaffold_idx": pair_scaffold_idx,
        "pair_count": pair_count,
        "row_dir_idx": row_dir_idx,
        "row_scaffold_idx": row_scaffold_idx,
        "dir_labels": np.array(dir_labels, dtype=object),
        "scaffold_labels": np.array(scaffold_labels, dtype=object),
        "dir_sizes": dir_sizes,
        "dir_scaffold_counts": dir_scaffold_counts,
        "private_molecule_counts": private_molecule_counts,
        "dir_conflict_matrix": dir_conflict_matrix,
        "conflict_degree": conflict_degree,
        "split_targets": split_targets.astype(np.float64),
        "min_retained_molecules_per_split": int(min_retained_molecules),
        "min_dir_ids_per_split": int(min_dir_ids),
        "total_size": total_size,
        "num_dirs": num_dirs,
        "num_scaffolds": num_scaffolds,
    }


def _ordered_dir_indices(problem, rng):
    """
    Order dir_ids so that private and larger dir_ids are handled first.
    """
    dir_indices = np.arange(problem["num_dirs"], dtype=np.int32)
    rng.shuffle(dir_indices)
    ordered = sorted(
        dir_indices.tolist(),
        key=lambda dir_idx: (
            -int(problem["private_molecule_counts"][dir_idx]),
            -int(problem["dir_sizes"][dir_idx]),
            round(float(problem["conflict_degree"][dir_idx]), CONFLICT_BUCKET_DECIMALS),
        ),
    )
    for start in range(0, len(ordered), ORDER_SHUFFLE_WINDOW):
        stop = min(start + ORDER_SHUFFLE_WINDOW, len(ordered))
        shuffled_window = ordered[start:stop]
        rng.shuffle(shuffled_window)
        ordered[start:stop] = shuffled_window
    return np.array(ordered, dtype=np.int32)


def _select_seed_dir_ids(problem, rng):
    """
    Pick one seed dir_id for each split.
    Train receives a large dir_id, while val/test prefer low-conflict dir_ids with many private molecules.
    """
    if problem["num_dirs"] < NUM_SPLITS:
        raise ValueError(
            "At least three dir_ids are required to build train/val/test splits."
        )

    all_dir_indices = np.arange(problem["num_dirs"], dtype=np.int32)
    ranked_train_candidates = sorted(
        all_dir_indices.tolist(),
        key=lambda dir_idx: (
            -int(problem["dir_sizes"][dir_idx]),
            round(float(problem["conflict_degree"][dir_idx]), CONFLICT_BUCKET_DECIMALS),
            -int(problem["private_molecule_counts"][dir_idx]),
        ),
    )
    train_seed = int(
        _choose_from_ranked_candidates(
            ranked_candidates=ranked_train_candidates,
            pool_size=SEED_CANDIDATE_POOL_SIZE,
            rng=rng,
        )
    )

    selected = [train_seed]
    remaining = [dir_idx for dir_idx in all_dir_indices.tolist() if dir_idx != train_seed]

    def _best_holdout_seed(candidate_indices, selected_seeds):
        ranked_candidates = sorted(
            candidate_indices,
            key=lambda dir_idx: (
                float(problem["dir_conflict_matrix"][dir_idx, selected_seeds].sum()),
                -int(problem["private_molecule_counts"][dir_idx]),
                -int(problem["dir_sizes"][dir_idx]),
                round(float(problem["conflict_degree"][dir_idx]), CONFLICT_BUCKET_DECIMALS),
            ),
        )
        return int(
            _choose_from_ranked_candidates(
                ranked_candidates=ranked_candidates,
                pool_size=SEED_CANDIDATE_POOL_SIZE,
                rng=rng,
            )
        )

    val_seed = _best_holdout_seed(remaining, selected)
    selected.append(val_seed)
    remaining = [dir_idx for dir_idx in remaining if dir_idx != val_seed]
    test_seed = _best_holdout_seed(remaining, selected)
    return np.array([train_seed, val_seed, test_seed], dtype=np.int32)


def _initialize_assignment_state(problem):
    assignment = np.full(problem["num_dirs"], -1, dtype=np.int8)
    raw_split_sizes = np.zeros(NUM_SPLITS, dtype=np.int64)
    assigned_dir_counts = np.zeros(NUM_SPLITS, dtype=np.int64)
    split_scaffold_counts = [dict() for _ in range(NUM_SPLITS)]
    return assignment, raw_split_sizes, assigned_dir_counts, split_scaffold_counts


def _assign_dir_to_split(problem, assignment, raw_split_sizes, assigned_dir_counts, split_scaffold_counts, dir_idx, split_idx):
    assignment[dir_idx] = split_idx
    raw_split_sizes[split_idx] += int(problem["dir_sizes"][dir_idx])
    assigned_dir_counts[split_idx] += 1
    for scaffold_idx, count in problem["dir_scaffold_counts"][dir_idx].items():
        split_scaffold_counts[split_idx][scaffold_idx] = (
            split_scaffold_counts[split_idx].get(scaffold_idx, 0) + count
        )


def _estimated_dir_retention(problem, split_scaffold_counts, raw_split_sizes, dir_idx, candidate_split):
    """
    Estimate how many molecules of a dir_id are likely to survive pruning if assigned to a split.
    """
    retained_estimate = 0.0
    candidate_need = float(problem["split_targets"][candidate_split] - raw_split_sizes[candidate_split])

    for scaffold_idx, count in problem["dir_scaffold_counts"][dir_idx].items():
        candidate_total = split_scaffold_counts[candidate_split].get(scaffold_idx, 0) + count
        competitor_totals = []
        competitor_need = -np.inf
        for other_split in range(NUM_SPLITS):
            if other_split == candidate_split:
                continue
            other_total = split_scaffold_counts[other_split].get(scaffold_idx, 0)
            competitor_totals.append(other_total)
            if other_total >= candidate_total:
                competitor_need = max(
                    competitor_need,
                    float(problem["split_targets"][other_split] - raw_split_sizes[other_split]),
                )

        other_max = max(competitor_totals, default=0)
        if other_max == 0 or candidate_total > other_max:
            retained_estimate += count
        elif candidate_total == other_max:
            retained_estimate += count if candidate_need >= competitor_need else 0.5 * count

    return float(retained_estimate)


def _shortfalls(raw_split_sizes, dir_counts, problem, split_indices):
    molecule_shortfall = int(
        np.maximum(
            0,
            problem["min_retained_molecules_per_split"] - raw_split_sizes[list(split_indices)],
        ).sum()
    )
    dir_shortfall = int(
        np.maximum(
            0,
            problem["min_dir_ids_per_split"] - dir_counts[list(split_indices)],
        ).sum()
    )
    return molecule_shortfall, dir_shortfall


def _underfilled_holdout_splits(raw_split_sizes, assigned_dir_counts, problem):
    underfilled = []
    for split_idx in HOLDOUT_SPLITS:
        molecule_shortfall = max(
            0,
            problem["min_retained_molecules_per_split"] - int(raw_split_sizes[split_idx]),
        )
        dir_shortfall = max(
            0,
            problem["min_dir_ids_per_split"] - int(assigned_dir_counts[split_idx]),
        )
        if molecule_shortfall > 0 or dir_shortfall > 0:
            underfilled.append(
                (
                    split_idx,
                    molecule_shortfall,
                    dir_shortfall,
                )
            )
    underfilled.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [item[0] for item in underfilled]


def _candidate_assignment_score(
    problem,
    assignment,
    raw_split_sizes,
    assigned_dir_counts,
    split_scaffold_counts,
    dir_idx,
    split_idx,
    phase,
):
    projected_raw_sizes = raw_split_sizes.copy()
    projected_raw_sizes[split_idx] += int(problem["dir_sizes"][dir_idx])
    projected_dir_counts = assigned_dir_counts.copy()
    projected_dir_counts[split_idx] += 1

    assigned_mask = assignment != -1
    total_assigned_conflict = float(problem["dir_conflict_matrix"][dir_idx, assigned_mask].sum())
    same_split_conflict = float(
        problem["dir_conflict_matrix"][dir_idx, assignment == split_idx].sum()
    )
    cross_split_conflict = total_assigned_conflict - same_split_conflict
    estimated_retention = _estimated_dir_retention(
        problem=problem,
        split_scaffold_counts=split_scaffold_counts,
        raw_split_sizes=raw_split_sizes,
        dir_idx=dir_idx,
        candidate_split=split_idx,
    )
    private_bonus = int(problem["private_molecule_counts"][dir_idx])
    balance_penalty = float(
        np.abs(projected_raw_sizes - problem["split_targets"]).sum()
    )
    deficit = float(problem["split_targets"][split_idx] - raw_split_sizes[split_idx])

    if phase == "minimums":
        candidate_splits = HOLDOUT_SPLITS
    else:
        candidate_splits = range(NUM_SPLITS)

    molecule_shortfall, dir_shortfall = _shortfalls(
        raw_split_sizes=projected_raw_sizes,
        dir_counts=projected_dir_counts,
        problem=problem,
        split_indices=candidate_splits,
    )

    return (
        molecule_shortfall,
        dir_shortfall,
        cross_split_conflict,
        -estimated_retention,
        balance_penalty,
        -deficit,
        -private_bonus,
    )


def _initial_assignment(problem, seed):
    """
    Assign complete dir_ids in two stages:
        1. Fill val/test until their minimum raw sizes are covered.
        2. Assign the rest while minimizing conflict and preserving balance.
    """
    rng = np.random.default_rng(seed)
    assignment, raw_split_sizes, assigned_dir_counts, split_scaffold_counts = _initialize_assignment_state(problem)

    seed_dir_indices = _select_seed_dir_ids(problem, rng)
    for split_idx, dir_idx in enumerate(seed_dir_indices):
        _assign_dir_to_split(
            problem=problem,
            assignment=assignment,
            raw_split_sizes=raw_split_sizes,
            assigned_dir_counts=assigned_dir_counts,
            split_scaffold_counts=split_scaffold_counts,
            dir_idx=int(dir_idx),
            split_idx=split_idx,
        )

    ordered_dir_indices = _ordered_dir_indices(problem, rng)
    remaining_dir_indices = [
        int(dir_idx)
        for dir_idx in ordered_dir_indices.tolist()
        if assignment[int(dir_idx)] == -1
    ]

    while remaining_dir_indices:
        underfilled_splits = _underfilled_holdout_splits(
            raw_split_sizes=raw_split_sizes,
            assigned_dir_counts=assigned_dir_counts,
            problem=problem,
        )
        phase = "minimums" if underfilled_splits else "balance"
        candidate_splits = underfilled_splits if underfilled_splits else range(NUM_SPLITS)

        dir_idx = remaining_dir_indices.pop(0)
        split_scores = []
        for split_idx in candidate_splits:
            score = _candidate_assignment_score(
                problem=problem,
                assignment=assignment,
                raw_split_sizes=raw_split_sizes,
                assigned_dir_counts=assigned_dir_counts,
                split_scaffold_counts=split_scaffold_counts,
                dir_idx=dir_idx,
                split_idx=split_idx,
                phase=phase,
            )
            split_scores.append((score, split_idx))

        split_scores.sort(key=lambda item: item[0])
        if phase == "minimums":
            selected_pool_size = MINIMUM_PHASE_SPLIT_POOL_SIZE
        else:
            selected_pool_size = BALANCE_PHASE_SPLIT_POOL_SIZE
        best_split = int(
            _choose_from_ranked_candidates(
                ranked_candidates=[split_idx for _, split_idx in split_scores],
                pool_size=selected_pool_size,
                rng=rng,
            )
        )

        _assign_dir_to_split(
            problem=problem,
            assignment=assignment,
            raw_split_sizes=raw_split_sizes,
            assigned_dir_counts=assigned_dir_counts,
            split_scaffold_counts=split_scaffold_counts,
            dir_idx=dir_idx,
            split_idx=int(best_split),
        )

    return assignment


def _choose_owner_splits(scaffold_split_counts, assigned_split_sizes, problem):
    """
    Choose the owner split of each scaffold.
    Tie-break order:
        1. highest count of that scaffold
        2. split most below its minimum retained size
        3. split most below its target size
        4. stable order train > val > test
    """
    minimum_pressure = np.maximum(
        0,
        problem["min_retained_molecules_per_split"] - assigned_split_sizes,
    ).astype(np.float64)
    target_deficit = (problem["split_targets"] - assigned_split_sizes).astype(np.float64)
    split_priority = 2.0 * minimum_pressure + np.maximum(0.0, target_deficit)

    max_counts = scaffold_split_counts.max(axis=1, keepdims=True)
    max_mask = scaffold_split_counts == max_counts
    priority_scores = np.where(
        max_mask,
        np.broadcast_to(split_priority, scaffold_split_counts.shape),
        -np.inf,
    )
    best_priority = priority_scores.max(axis=1, keepdims=True)
    owner_mask = priority_scores == best_priority
    return owner_mask.argmax(axis=1).astype(np.int8)


def _evaluate_assignment(problem, assignment):
    """
    Evaluate a full dir_id -> split assignment after pruning scaffold conflicts.
    """
    split_idx_per_pair = assignment[problem["pair_dir_idx"]]

    scaffold_split_counts = np.zeros(
        (problem["num_scaffolds"], NUM_SPLITS),
        dtype=np.int64,
    )
    np.add.at(
        scaffold_split_counts,
        (problem["pair_scaffold_idx"], split_idx_per_pair),
        problem["pair_count"],
    )

    assigned_split_sizes = np.bincount(
        split_idx_per_pair,
        weights=problem["pair_count"],
        minlength=NUM_SPLITS,
    ).astype(np.int64)
    assigned_dir_counts = np.bincount(
        assignment,
        minlength=NUM_SPLITS,
    ).astype(np.int64)
    owner_splits = _choose_owner_splits(
        scaffold_split_counts=scaffold_split_counts,
        assigned_split_sizes=assigned_split_sizes,
        problem=problem,
    )

    keep_pair_mask = split_idx_per_pair == owner_splits[problem["pair_scaffold_idx"]]
    retained_by_dir = np.bincount(
        problem["pair_dir_idx"][keep_pair_mask],
        weights=problem["pair_count"][keep_pair_mask],
        minlength=problem["num_dirs"],
    ).astype(np.int64)
    removed_by_dir = problem["dir_sizes"] - retained_by_dir
    retained_split_sizes = np.bincount(
        assignment,
        weights=retained_by_dir,
        minlength=NUM_SPLITS,
    ).astype(np.int64)
    retained_dir_counts = np.bincount(
        assignment[retained_by_dir > 0],
        minlength=NUM_SPLITS,
    ).astype(np.int64)
    removed_total_by_scaffold = (
        scaffold_split_counts.sum(axis=1)
        - scaffold_split_counts[np.arange(problem["num_scaffolds"]), owner_splits]
    )
    molecule_shortfall = np.maximum(
        0,
        problem["min_retained_molecules_per_split"] - retained_split_sizes,
    ).astype(np.int64)
    dir_shortfall = np.maximum(
        0,
        problem["min_dir_ids_per_split"] - retained_dir_counts,
    ).astype(np.int64)
    valid_split_mask = (molecule_shortfall == 0) & (dir_shortfall == 0)

    return {
        "assignment": assignment.copy(),
        "owner_splits": owner_splits,
        "scaffold_split_counts": scaffold_split_counts,
        "retained_split_sizes": retained_split_sizes,
        "retained_dir_counts": retained_dir_counts,
        "retained_by_dir": retained_by_dir,
        "removed_by_dir": removed_by_dir,
        "removed_total_by_scaffold": removed_total_by_scaffold,
        "retained_total": int(retained_by_dir.sum()),
        "removed_total": int(problem["total_size"] - retained_by_dir.sum()),
        "balance_penalty": float(
            np.abs(retained_split_sizes - problem["split_targets"]).sum()
        ),
        "dropped_dir_count": int((retained_by_dir == 0).sum()),
        "molecule_shortfall": molecule_shortfall,
        "dir_shortfall": dir_shortfall,
        "valid_split_count": int(valid_split_mask.sum()),
    }


def _score_evaluation(evaluation):
    """
    Rank split candidates.
    Priority order:
        1. all splits valid,
        2. train large enough,
        3. larger train/val/test,
        4. larger total retention,
        5. lower balance penalty and fewer dropped dir_ids.
    """
    return (
        evaluation["valid_split_count"],
        int(evaluation["train_fraction_valid"]),
        -int(evaluation["molecule_shortfall"].sum()),
        -int(evaluation["dir_shortfall"].sum()),
        evaluation["retained_split_sizes"][SPLIT_TO_INDEX["train"]],
        evaluation["retained_split_sizes"][SPLIT_TO_INDEX["val"]],
        evaluation["retained_split_sizes"][SPLIT_TO_INDEX["test"]],
        evaluation["retained_total"],
        -evaluation["balance_penalty"],
        -evaluation["dropped_dir_count"],
    )


def _relabel_assignment(assignment, old_to_new_split_map):
    """
    Relabel split indices according to a given old->new mapping.
    """
    relabelled_assignment = np.asarray(assignment, dtype=np.int8).copy()
    for old_split_idx, new_split_idx in enumerate(old_to_new_split_map):
        relabelled_assignment[np.asarray(assignment) == old_split_idx] = new_split_idx
    return relabelled_assignment


def _relabel_evaluation_by_size(problem, evaluation):
    """
    Relabel the evaluated splits so that:
        largest retained split -> train
        second largest retained split -> val
        third largest retained split -> test
    """
    split_order = np.argsort(-evaluation["retained_split_sizes"], kind="stable")
    old_to_new_split_map = np.empty(NUM_SPLITS, dtype=np.int8)
    for new_split_idx, old_split_idx in enumerate(split_order):
        old_to_new_split_map[old_split_idx] = new_split_idx

    relabelled_evaluation = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in evaluation.items()
    }
    relabelled_evaluation["assignment"] = _relabel_assignment(
        assignment=evaluation["assignment"],
        old_to_new_split_map=old_to_new_split_map,
    )
    relabelled_evaluation["owner_splits"] = old_to_new_split_map[evaluation["owner_splits"]]
    for split_key in (
        "scaffold_split_counts",
        "retained_split_sizes",
        "retained_dir_counts",
        "molecule_shortfall",
        "dir_shortfall",
    ):
        if split_key == "scaffold_split_counts":
            relabelled_evaluation[split_key] = evaluation[split_key][:, split_order].copy()
        else:
            relabelled_evaluation[split_key] = evaluation[split_key][split_order].copy()

    relabelled_evaluation["old_to_new_split_map"] = old_to_new_split_map
    relabelled_evaluation["balance_penalty"] = float(
        np.abs(relabelled_evaluation["retained_split_sizes"] - problem["split_targets"]).sum()
    )
    relabelled_evaluation["valid_split_count"] = int(
        np.count_nonzero(
            (relabelled_evaluation["molecule_shortfall"] == 0)
            & (relabelled_evaluation["dir_shortfall"] == 0)
        )
    )
    relabelled_evaluation["train_retained_fraction"] = float(
        relabelled_evaluation["retained_split_sizes"][SPLIT_TO_INDEX["train"]]
        / relabelled_evaluation["retained_total"]
    )
    relabelled_evaluation["train_fraction_valid"] = (
        relabelled_evaluation["train_retained_fraction"] >= MIN_TRAIN_RETAINED_FRACTION
    )
    return relabelled_evaluation


def _materialize_split_dataframes(problem, evaluation):
    row_assignment = evaluation["assignment"][problem["row_dir_idx"]]
    keep_mask = row_assignment == evaluation["owner_splits"][problem["row_scaffold_idx"]]
    retained_df = problem["df"].loc[keep_mask].copy()
    retained_df.insert(0, "_split_idx", row_assignment[keep_mask])

    split_dataframes = {}
    for split_idx, split_name in enumerate(SPLIT_NAMES):
        split_df = retained_df[retained_df["_split_idx"] == split_idx].drop(columns="_split_idx")
        split_dataframes[split_name] = split_df.reset_index(drop=True)
    return split_dataframes


def _validate_final_split_dataframes(split_dataframes):
    for split_name, split_df in split_dataframes.items():
        if split_df.empty:
            raise ValueError(f"The final {split_name} split is empty after pruning.")

    for left_index, left_name in enumerate(SPLIT_NAMES):
        left_df = split_dataframes[left_name]
        left_dir_ids = set(left_df["dir_id"].unique())
        left_scaffolds = set(left_df["ms_smiles"].unique())
        for right_name in SPLIT_NAMES[left_index + 1:]:
            right_df = split_dataframes[right_name]
            overlapping_dir_ids = left_dir_ids.intersection(right_df["dir_id"].unique())
            overlapping_scaffolds = left_scaffolds.intersection(right_df["ms_smiles"].unique())
            if overlapping_dir_ids:
                raise ValueError(
                    f"Found shared dir_ids between {left_name} and {right_name}: "
                    f"{sorted(overlapping_dir_ids)[:5]}"
                )
            if overlapping_scaffolds:
                raise ValueError(
                    f"Found shared ms_smiles between {left_name} and {right_name}: "
                    f"{sorted(overlapping_scaffolds)[:5]}"
                )


def _build_dir_assignment_report(problem, original_assignment, final_evaluation):
    report_df = pd.DataFrame(
        {
            "dir_id": problem["dir_labels"],
            "original_split": [SPLIT_NAMES[index] for index in original_assignment],
            "final_split": [SPLIT_NAMES[index] for index in final_evaluation["assignment"]],
            "original_count": problem["dir_sizes"],
            "private_count": problem["private_molecule_counts"],
            "retained_count": final_evaluation["retained_by_dir"],
            "removed_count": final_evaluation["removed_by_dir"],
        }
    )
    report_df["retained_fraction"] = np.where(
        report_df["original_count"] > 0,
        report_df["retained_count"] / report_df["original_count"],
        0.0,
    )
    report_df["moved"] = report_df["original_split"] != report_df["final_split"]
    report_df["dropped"] = report_df["retained_count"] == 0
    return report_df.sort_values(
        by=["dropped", "removed_count", "dir_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _build_scaffold_pruning_report(problem, evaluation):
    report_df = pd.DataFrame(
        {
            "ms_smiles": problem["scaffold_labels"],
            "owner_split": [SPLIT_NAMES[index] for index in evaluation["owner_splits"]],
            "train_count": evaluation["scaffold_split_counts"][:, SPLIT_TO_INDEX["train"]],
            "val_count": evaluation["scaffold_split_counts"][:, SPLIT_TO_INDEX["val"]],
            "test_count": evaluation["scaffold_split_counts"][:, SPLIT_TO_INDEX["test"]],
            "removed_total": evaluation["removed_total_by_scaffold"],
        }
    )
    return report_df.sort_values(
        by=["removed_total", "ms_smiles"],
        ascending=[False, True],
    ).reset_index(drop=True)


def cc_ms_split(
    ms_complete_file,
    save_dir,
    random_seed,
    processed_file,
    save_complete_ms_dir,
    apply_low_grad_filter,
    drop_smrt,
):
    """
    Obtain train/val/test splits by assigning complete dir_ids to splits and then
    pruning scaffolds that would be shared across splits.
    """
    if not Path(ms_complete_file).exists():
        print("Creating the scaffold split file...")
        ms_split(
            input_path=processed_file,
            output_dir=save_complete_ms_dir,
            apply_low_grad_filter= apply_low_grad_filter,
            drop_smrt=drop_smrt,
        )

    print("Reading the input file...")
    input_df = pd.read_csv(ms_complete_file, sep="\t")

    print("Checking for the output dir...")
    os.makedirs(save_dir, exist_ok=True)

    print("Building split problem...")
    problem = _build_split_problem(input_df)
    print(
        f"Loaded {problem['total_size']} molecules, {problem['num_dirs']} dir_ids "
        f"and {problem['num_scaffolds']} scaffolds."
    )
    print(
        "Minimum retained split sizes:"
        f" molecules>={problem['min_retained_molecules_per_split']},"
        f" dir_ids>={problem['min_dir_ids_per_split']}"
    )

    print("Running deterministic restarts...")
    best_evaluation = None
    best_initial_assignment = None
    best_restart = None
    best_score = None
    for restart_idx in range(NUM_RESTARTS):
        initial_assignment = _initial_assignment(problem, random_seed + restart_idx)
        final_evaluation = _evaluate_assignment(problem, initial_assignment)
        final_evaluation = _relabel_evaluation_by_size(problem, final_evaluation)
        restart_score = _score_evaluation(final_evaluation)
        print(
            f"Restart {restart_idx + 1}/{NUM_RESTARTS}: "
            f"retained={final_evaluation['retained_total']}/{problem['total_size']} "
            f"({final_evaluation['retained_total'] / problem['total_size']:.2%}), "
            f"removed={final_evaluation['removed_total']}, "
            f"sizes={final_evaluation['retained_split_sizes'].tolist()}, "
            f"dir_ids={final_evaluation['retained_dir_counts'].tolist()}, "
            f"molecule_shortfall={final_evaluation['molecule_shortfall'].tolist()}, "
            f"dir_shortfall={final_evaluation['dir_shortfall'].tolist()}, "
            f"train_fraction={final_evaluation['train_retained_fraction']:.2%}, "
            f"score={restart_score}",
            flush=True,
        )
        if best_score is None or restart_score > best_score:
            best_evaluation = final_evaluation
            best_initial_assignment = initial_assignment.copy()
            best_restart = restart_idx
            best_score = restart_score
            print(
                f"  -> New best restart: {restart_idx + 1}/{NUM_RESTARTS}",
                flush=True,
            )

    if best_evaluation is None:
        raise ValueError(
            "Could not generate train/val/test splits under the current "
            "dir_id-first assignment and scaffold-pruning strategy."
        )

    if best_evaluation["valid_split_count"] < NUM_SPLITS or not best_evaluation["train_fraction_valid"]:
        raise ValueError(
            "Could not satisfy the minimum split constraints. "
            f"Best attempt retained sizes were {best_evaluation['retained_split_sizes'].tolist()} "
            f"and retained dir counts were {best_evaluation['retained_dir_counts'].tolist()}. "
            f"Train fraction was {best_evaluation['train_retained_fraction']:.2%}."
        )

    split_dataframes = _materialize_split_dataframes(problem, best_evaluation)
    _validate_final_split_dataframes(split_dataframes)

    retained_total = best_evaluation["retained_total"]
    removed_total = best_evaluation["removed_total"]
    print(
        f"Best restart: {best_restart}. "
        f"Retained {retained_total}/{problem['total_size']} molecules "
        f"({retained_total / problem['total_size']:.2%}), removed {removed_total}."
    )
    print(
        "Final split sizes:"
        f" train={split_dataframes['train'].shape[0]},"
        f" val={split_dataframes['val'].shape[0]},"
        f" test={split_dataframes['test'].shape[0]}"
    )
    print(
        "Final retained dir_ids:"
        f" train={best_evaluation['retained_dir_counts'][SPLIT_TO_INDEX['train']]},"
        f" val={best_evaluation['retained_dir_counts'][SPLIT_TO_INDEX['val']]},"
        f" test={best_evaluation['retained_dir_counts'][SPLIT_TO_INDEX['test']]}"
    )
    print(
        f"Final train fraction: {best_evaluation['train_retained_fraction']:.2%}"
    )

    print(f"Saving result files to {save_dir}")
    split_dataframes["train"].to_csv(f"{save_dir}train_data.tsv", sep="\t", index=False)
    split_dataframes["val"].to_csv(f"{save_dir}val_data.tsv", sep="\t", index=False)
    split_dataframes["test"].to_csv(f"{save_dir}test_data.tsv", sep="\t", index=False)

    relabelled_initial_assignment = _relabel_assignment(
        assignment=best_initial_assignment,
        old_to_new_split_map=best_evaluation["old_to_new_split_map"],
    )
    dir_assignment_report = _build_dir_assignment_report(
        problem=problem,
        original_assignment=relabelled_initial_assignment,
        final_evaluation=best_evaluation,
    )
    scaffold_pruning_report = _build_scaffold_pruning_report(
        problem=problem,
        evaluation=best_evaluation,
    )
    dir_assignment_report.to_csv(
        f"{save_dir}dir_assignment_report.tsv",
        sep="\t",
        index=False,
    )
    scaffold_pruning_report.to_csv(
        f"{save_dir}scaffold_pruning_report.tsv",
        sep="\t",
        index=False,
    )
    return
