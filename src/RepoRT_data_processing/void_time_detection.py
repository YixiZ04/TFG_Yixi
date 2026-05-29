"""Heuristic LC-MS early-elution front detection from empirical RT distributions.

This module estimates a putative void time / early-elution front for a single
LC-MS experiment using only observed retention times (RTs). It does not
determine the true chromatographic hold-up time, which would require a
non-retained marker or column metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from textwrap import shorten
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


VOID_DETECTION_METHOD = "kde_first_valley"


@dataclass(frozen=True)
class VoidDetectionParams:
    """Tunable parameters for KDE-based putative void-time detection."""

    min_n: int = 30
    kde_grid_size: int = 2000
    early_quantile: float = 0.35
    bandwidth_adjust: float = 1.0
    min_valley_depth: float = 0.25
    min_cluster_fraction: float = 0.01
    min_cluster_size: int = 5
    early_mode_search_fraction: float = 0.20
    min_early_peak_density_fraction: float = 0.02
    max_t0_fraction_of_run: float = 0.15
    max_t0_absolute: float = 3.0
    max_t0_quantile: float = 0.20
    max_cutoff_fraction_of_run: float = 0.20
    max_cutoff_quantile: float = 0.40
    weak_retention_k_threshold: float = 1.0
    allow_low_confidence_classification: bool = False
    use_log_transform: bool = True
    min_rt: float = 0.0
    enable_density_drop_fallback: bool = True
    fallback_density_drop_fraction: float = 0.30
    fallback_min_density_enrichment: float = 1.8
    fallback_min_peak_global_fraction: float = 0.75
    fallback_max_cutoff_quantile: float = 0.70
    fallback_min_initial_slope_fraction: float = 0.20
    fallback_slope_relax_fraction: float = 0.25
    fallback_shoulder_density_fraction: float = 0.60
    enable_histogram_first_bin_fallback: bool = True
    histogram_peak_ratio_to_second_bin: float = 2.5
    histogram_peak_ratio_to_rest_mean: float = 3.0
    histogram_first_two_bins_min_fraction: float = 0.20
    histogram_two_bin_ratio_to_rest_mean: float = 1.5
    histogram_two_bin_combined_ratio_to_rest_mean: float = 2.0
    histogram_sparse_first_bin_max_fraction: float = 0.02
    histogram_second_bin_ratio_to_rest_mean: float = 2.5
    histogram_drop_fraction: float = 0.20
    histogram_peak_min_fraction: float = 0.08
    histogram_valley_bin_limit: int = 4
    histogram_max_cutoff_fraction_of_run: float = 0.30


@dataclass(frozen=True)
class VoidDetectionResult:
    """Summary of a putative early-elution front estimate."""

    n_total: int
    n_valid_rt: int
    putative_t0: Optional[float]
    early_elution_cutoff: Optional[float]
    confidence: str
    detection_method: str
    flags: list[str] = field(default_factory=list)
    n_putatively_non_retained: int = 0
    n_weakly_retained: int = 0
    n_retained: int = 0
    n_uncertain: int = 0


@dataclass(frozen=True)
class _PreparedRTValues:
    rt_min: pd.Series
    valid_mask: pd.Series
    valid_rt_values: np.ndarray
    flags: list[str]


@dataclass(frozen=True)
class _ModeValleyDetection:
    mode_x: float
    mode_rt: float
    cutoff_x: float
    cutoff_rt: float
    valley_depth: float
    early_cluster_size: int
    method: str


def detect_putative_void_time(
    df: pd.DataFrame,
    rt_col: str,
    compound_id_col: Optional[str] = None,
    units: str = "minutes",
    params: Optional[VoidDetectionParams] = None,
) -> Tuple[pd.DataFrame, VoidDetectionResult]:
    """Estimate a putative void time / early-elution front from RTs alone.

    Estimate a putative void time / early-elution front from the empirical RT
    distribution of a single LC-MS experiment. This function does not determine
    the true chromatographic hold-up time. It uses a KDE-based first-valley
    heuristic to identify a possible early cluster of non-retained or poorly
    retained compounds.

    The output should be interpreted as a conservative screening aid for early
    elution. The label ``putatively_non_retained`` does not prove the compound
    did not interact with the column, and ``pseudo_k`` is not a formal
    chromatographic retention factor.
    """

    resolved_params = params or VoidDetectionParams()
    _validate_inputs(df, rt_col, units, compound_id_col)

    prepared = _prepare_rt_values(df, rt_col, units, resolved_params)
    annotated_df = df.copy(deep=True)
    annotated_df["rt_min"] = prepared.rt_min

    base_flags = list(prepared.flags)
    n_total = len(df)
    n_valid_rt = int(prepared.valid_mask.sum())

    putative_t0: Optional[float] = None
    early_elution_cutoff: Optional[float] = None
    confidence = "low"
    flags = list(base_flags)

    if n_valid_rt < resolved_params.min_n:
        confidence = "insufficient_data"
        _append_flag(flags, "insufficient_data")
    else:
        x_values = _transform_rt_values(prepared.valid_rt_values, resolved_params)
        kde_result = _fit_kde(x_values, resolved_params)
        if kde_result is None:
            confidence = "low"
            _append_flag(flags, "kde_failed")
        else:
            x_grid, density = kde_result
            detection = _find_first_mode_and_valley(
                x_grid=x_grid,
                density=density,
                x_values=x_values,
                rt_values=prepared.valid_rt_values,
                params=resolved_params,
            )
            if detection is not None:
                _append_flag(flags, "cutoff_estimated_from_first_valley")
            elif resolved_params.enable_density_drop_fallback:
                detection = _find_early_density_drop_front(
                    x_grid=x_grid,
                    density=density,
                    x_values=x_values,
                    rt_values=prepared.valid_rt_values,
                    flags=flags,
                    params=resolved_params,
                )
            if detection is None and resolved_params.enable_histogram_first_bin_fallback:
                detection = _find_histogram_first_bin_front(
                    rt_values=prepared.valid_rt_values,
                    flags=flags,
                    params=resolved_params,
                )
            if detection is None:
                confidence = "low"
                if "no_early_mode_detected" not in flags:
                    _append_flag(flags, "no_clear_early_valley")
            else:
                early_cluster_mask = prepared.valid_rt_values <= detection.cutoff_rt
                early_cluster_rts = prepared.valid_rt_values[early_cluster_mask]
                if detection.method == "histogram_first_bin" and detection.mode_rt <= _max_t0_rt(
                    prepared.valid_rt_values,
                    resolved_params,
                ):
                    if "putative_t0_candidate_too_late" in flags:
                        flags.remove("putative_t0_candidate_too_late")
                if (
                    detection.method == "histogram_first_bin"
                    and "t0_estimated_as_median_of_second_histogram_bin" in flags
                    and "possible_early_outlier" in flags
                ):
                    flags.remove("possible_early_outlier")
                putative_t0 = detection.mode_rt
                if "many_duplicate_rt_values" in flags and early_cluster_rts.size >= resolved_params.min_cluster_size:
                    putative_t0 = float(np.median(early_cluster_rts))
                    _append_flag(flags, "t0_estimated_as_median_of_early_cluster")
                early_elution_cutoff = detection.cutoff_rt
                confidence = _estimate_confidence(
                    mode_rt=putative_t0,
                    cutoff_rt=early_elution_cutoff,
                    valley_depth=detection.valley_depth,
                    early_cluster_size=detection.early_cluster_size,
                    rt_values=prepared.valid_rt_values,
                    flags=flags,
                    params=resolved_params,
                    detection_method=detection.method,
                )
    annotated_df["putative_t0"] = putative_t0
    annotated_df["early_elution_cutoff"] = early_elution_cutoff
    annotated_df["pseudo_k"] = np.nan
    if putative_t0 is not None and putative_t0 > 0:
        annotated_df["pseudo_k"] = (annotated_df["rt_min"] - putative_t0) / putative_t0

    annotated_df["void_status"] = _classify_rt_values(
        rt_values=annotated_df["rt_min"].to_numpy(dtype=float),
        valid_mask=prepared.valid_mask.to_numpy(dtype=bool),
        putative_t0=putative_t0,
        cutoff=early_elution_cutoff,
        confidence=confidence,
        params=resolved_params,
    )
    annotated_df["void_confidence"] = confidence
    annotated_df["void_detection_method"] = VOID_DETECTION_METHOD
    annotated_df["void_detection_flags"] = "; ".join(flags)

    status_counts = annotated_df["void_status"].value_counts(dropna=False)
    result = VoidDetectionResult(
        n_total=n_total,
        n_valid_rt=n_valid_rt,
        putative_t0=putative_t0,
        early_elution_cutoff=early_elution_cutoff,
        confidence=confidence,
        detection_method=VOID_DETECTION_METHOD,
        flags=flags,
        n_putatively_non_retained=int(status_counts.get("putatively_non_retained", 0)),
        n_weakly_retained=int(status_counts.get("weakly_retained", 0)),
        n_retained=int(status_counts.get("retained", 0)),
        n_uncertain=int(status_counts.get("uncertain", 0)),
    )
    return annotated_df, result


def _validate_inputs(
    df: pd.DataFrame,
    rt_col: str,
    units: str,
    compound_id_col: Optional[str] = None,
) -> None:
    """Validate the public API inputs."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")
    if rt_col not in df.columns:
        raise ValueError(f"rt_col '{rt_col}' is not present in the DataFrame.")
    if compound_id_col is not None and compound_id_col not in df.columns:
        raise ValueError(f"compound_id_col '{compound_id_col}' is not present in the DataFrame.")
    if units not in {"minutes", "seconds"}:
        raise ValueError("units must be either 'minutes' or 'seconds'.")

    numeric_rt = pd.to_numeric(df[rt_col], errors="coerce")
    invalid_numeric = df[rt_col].notna() & numeric_rt.isna()
    if bool(invalid_numeric.any()):
        raise ValueError(f"rt_col '{rt_col}' contains non-numeric values that cannot be converted.")


def _prepare_rt_values(
    df: pd.DataFrame,
    rt_col: str,
    units: str,
    params: VoidDetectionParams,
) -> _PreparedRTValues:
    """Convert RTs to minutes and derive valid values used for estimation."""

    rt_numeric = pd.to_numeric(df[rt_col], errors="coerce").astype(float)
    rt_min = rt_numeric / 60.0 if units == "seconds" else rt_numeric.copy()
    valid_mask = rt_min.notna() & np.isfinite(rt_min.to_numpy(dtype=float)) & (rt_min > params.min_rt)
    valid_rt_values = np.sort(rt_min.loc[valid_mask].to_numpy(dtype=float))

    flags: list[str] = []
    if valid_rt_values.size:
        duplicate_counts = pd.Series(valid_rt_values).value_counts(dropna=False)
        if float(duplicate_counts.max()) / float(valid_rt_values.size) > 0.20:
            _append_flag(flags, "many_duplicate_rt_values")

        rt_range = float(valid_rt_values[-1] - valid_rt_values[0])
        if rt_range < 0.25:
            _append_flag(flags, "narrow_rt_range")

        if _has_possible_early_outlier(valid_rt_values):
            _append_flag(flags, "possible_early_outlier")

    return _PreparedRTValues(
        rt_min=rt_min,
        valid_mask=valid_mask,
        valid_rt_values=valid_rt_values,
        flags=flags,
    )


def _transform_rt_values(rt_values: np.ndarray, params: VoidDetectionParams) -> np.ndarray:
    """Transform RT values for KDE fitting."""

    if params.use_log_transform:
        return np.log1p(rt_values)
    return rt_values.astype(float, copy=True)


def _inverse_transform_rt_values(x_values: np.ndarray | float, params: VoidDetectionParams) -> np.ndarray | float:
    """Map transformed values back to minutes."""

    if params.use_log_transform:
        return np.expm1(x_values)
    return x_values


def _fit_kde(x_values: np.ndarray, params: VoidDetectionParams) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Fit a KDE on the transformed RT values and evaluate it on a regular grid."""

    if x_values.size < 2 or np.allclose(x_values, x_values[0]):
        return None

    try:
        kde = gaussian_kde(x_values)
        kde.set_bandwidth(bw_method=kde.factor * params.bandwidth_adjust)
        x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), params.kde_grid_size)
        density = kde.evaluate(x_grid)
    except Exception:
        return None

    if not np.all(np.isfinite(density)):
        return None
    return x_grid, density


def _early_mode_search_limit(x_values: np.ndarray, params: VoidDetectionParams) -> float:
    """Return an early search boundary that is robust to duplicated minimum RTs."""

    quantile_limit = float(np.quantile(x_values, params.early_quantile))
    range_limit = float(np.min(x_values) + params.early_mode_search_fraction * (np.max(x_values) - np.min(x_values)))
    return max(quantile_limit, range_limit)


def _max_t0_rt(rt_values: np.ndarray, params: VoidDetectionParams) -> float:
    """Return the maximum plausible putative t0 in minutes."""

    run_limit = float(np.max(rt_values) * params.max_t0_fraction_of_run)
    return min(run_limit, float(params.max_t0_absolute))


def _max_cutoff_rt(rt_values: np.ndarray, params: VoidDetectionParams) -> float:
    """Return the maximum plausible early-elution cutoff in minutes."""

    return float(np.max(rt_values) * params.max_cutoff_fraction_of_run)


def _find_first_mode_and_valley(
    x_grid: np.ndarray,
    density: np.ndarray,
    x_values: np.ndarray,
    rt_values: np.ndarray,
    params: VoidDetectionParams,
) -> Optional[_ModeValleyDetection]:
    """Find the first relevant early mode and the first acceptable downstream valley."""

    if x_grid.size < 3 or density.size < 3:
        return None

    density_max = float(np.max(density))
    if not np.isfinite(density_max) or density_max <= 0:
        return None

    early_zone_limit = _early_mode_search_limit(x_values, params)
    max_t0_rt = _max_t0_rt(rt_values, params)
    cutoff_limit_rt = min(
        float(np.quantile(rt_values, params.max_cutoff_quantile)),
        _max_cutoff_rt(rt_values, params),
    )
    required_cluster_size = max(
        params.min_cluster_size,
        int(ceil(params.min_cluster_fraction * rt_values.size)),
    )

    peak_indices = set(int(idx) for idx in find_peaks(density)[0].tolist())
    if density[0] > density[1]:
        peak_indices.add(0)
    if density[-1] > density[-2]:
        peak_indices.add(density.size - 1)

    valley_indices = sorted(int(idx) for idx in find_peaks(-density)[0].tolist())
    early_peaks = sorted(
        idx
        for idx in peak_indices
        if x_grid[idx] <= early_zone_limit and density[idx] >= params.min_early_peak_density_fraction * density_max
    )

    if not early_peaks:
        return None

    for peak_idx in early_peaks:
        peak_rt = float(_inverse_transform_rt_values(x_grid[peak_idx], params))
        if peak_rt <= params.min_rt:
            continue
        if peak_rt > max_t0_rt:
            continue

        downstream_valleys = [idx for idx in valley_indices if idx > peak_idx]
        for valley_idx in downstream_valleys:
            valley_rt = float(_inverse_transform_rt_values(x_grid[valley_idx], params))
            if valley_rt > cutoff_limit_rt:
                break

            early_cluster_size = int(np.sum(rt_values <= valley_rt))
            if early_cluster_size < required_cluster_size:
                continue

            valley_depth = 1.0 - float(density[valley_idx] / density[peak_idx])
            if valley_depth < params.min_valley_depth:
                continue

            return _ModeValleyDetection(
                mode_x=float(x_grid[peak_idx]),
                mode_rt=peak_rt,
                cutoff_x=float(x_grid[valley_idx]),
                cutoff_rt=valley_rt,
                valley_depth=valley_depth,
                early_cluster_size=early_cluster_size,
                method="first_valley",
            )

    return None


def _find_early_density_drop_front(
    x_grid: np.ndarray,
    density: np.ndarray,
    x_values: np.ndarray,
    rt_values: np.ndarray,
    flags: list[str],
    params: VoidDetectionParams,
) -> Optional[_ModeValleyDetection]:
    """Fallback detection for dominant early peaks without a clean local valley."""

    if x_grid.size < 5 or density.size < 5:
        return None

    density_max = float(np.max(density))
    if not np.isfinite(density_max) or density_max <= 0:
        return None

    early_zone_limit = _early_mode_search_limit(x_values, params)
    max_t0_rt = _max_t0_rt(rt_values, params)
    cutoff_limit_rt = min(
        float(np.quantile(rt_values, params.fallback_max_cutoff_quantile)),
        _max_cutoff_rt(rt_values, params),
    )
    required_cluster_size = max(
        params.min_cluster_size,
        int(ceil(params.min_cluster_fraction * rt_values.size)),
    )

    peak_indices = set(int(idx) for idx in find_peaks(density)[0].tolist())
    if density[0] > density[1]:
        peak_indices.add(0)
    early_peaks = sorted(
        idx
        for idx in peak_indices
        if x_grid[idx] <= early_zone_limit and density[idx] >= 0.10 * density_max
    )
    if not early_peaks:
        _append_flag(flags, "no_early_mode_detected")
        return None

    gradient = np.gradient(density, x_grid)
    curvature = np.gradient(gradient, x_grid)

    for peak_idx in early_peaks:
        peak_rt = float(_inverse_transform_rt_values(x_grid[peak_idx], params))
        if peak_rt <= params.min_rt:
            continue
        if peak_rt > max_t0_rt:
            _append_flag(flags, "putative_t0_candidate_too_late")
            continue

        after_early_mask = x_grid > early_zone_limit
        after_density = density[after_early_mask]
        if after_density.size == 0:
            continue

        reference_density = float(np.median(after_density))
        if reference_density <= 0:
            continue

        density_enrichment = float(density[peak_idx] / reference_density)
        peak_global_fraction = float(density[peak_idx] / density_max)
        if (
            density_enrichment < params.fallback_min_density_enrichment
            and peak_global_fraction < params.fallback_min_peak_global_fraction
        ):
            _append_flag(flags, "no_dominant_early_density")
            continue

        initial_end = min(density.size, peak_idx + max(4, density.size // 20))
        initial_gradient = gradient[peak_idx:initial_end]
        if initial_gradient.size == 0:
            continue
        strongest_negative_slope = abs(float(np.min(initial_gradient)))
        global_slope_scale = float(np.max(np.abs(gradient))) if gradient.size else 0.0
        if global_slope_scale <= 0:
            continue
        if strongest_negative_slope < params.fallback_min_initial_slope_fraction * global_slope_scale:
            _append_flag(flags, "insufficient_initial_density_drop")
            continue

        target_density = float(density[peak_idx] * params.fallback_density_drop_fraction)
        downstream_indices = np.arange(peak_idx + 1, density.size)
        cutoff_candidates = downstream_indices[density[downstream_indices] <= target_density]
        cutoff_idx: Optional[int] = None
        if cutoff_candidates.size:
            cutoff_idx = int(cutoff_candidates[0])

        density_drop_curvature_candidates = downstream_indices[
            (density[downstream_indices] <= target_density)
            & (gradient[downstream_indices] < 0)
            & (curvature[downstream_indices] > 0)
        ]
        if density_drop_curvature_candidates.size:
            cutoff_idx = int(density_drop_curvature_candidates[0])

        if cutoff_idx is None:
            _append_flag(flags, "insufficient_density_drop")
        else:
            cutoff_rt = float(_inverse_transform_rt_values(x_grid[cutoff_idx], params))
            if cutoff_rt > cutoff_limit_rt:
                cutoff_idx = None
                _append_flag(flags, "fallback_density_drop_cutoff_too_late")

        if cutoff_idx is None:
            slope_relax_candidates = downstream_indices[
                (gradient[downstream_indices] < 0)
                & (np.abs(gradient[downstream_indices]) <= params.fallback_slope_relax_fraction * strongest_negative_slope)
                & (curvature[downstream_indices] > 0)
            ]
            if slope_relax_candidates.size:
                slope_cutoff_idx = int(slope_relax_candidates[0])
                slope_cutoff_rt = float(_inverse_transform_rt_values(x_grid[slope_cutoff_idx], params))
                if slope_cutoff_rt <= cutoff_limit_rt:
                    cutoff_idx = slope_cutoff_idx
                    _append_flag(flags, "cutoff_estimated_from_slope_relaxation")

        if cutoff_idx is None:
            shoulder_density = float(density[peak_idx] * params.fallback_shoulder_density_fraction)
            shoulder_candidates = downstream_indices[density[downstream_indices] <= shoulder_density]
            if shoulder_candidates.size == 0:
                continue
            cutoff_idx = int(shoulder_candidates[0])
            _append_flag(flags, "cutoff_estimated_from_early_shoulder")

        cutoff_rt = float(_inverse_transform_rt_values(x_grid[cutoff_idx], params))
        if cutoff_rt > cutoff_limit_rt:
            _append_flag(flags, "fallback_cutoff_too_late")
            continue

        early_cluster_size = int(np.sum(rt_values <= cutoff_rt))
        if early_cluster_size < required_cluster_size:
            _append_flag(flags, "fallback_early_cluster_too_small")
            continue

        valley_depth = 1.0 - float(density[cutoff_idx] / density[peak_idx])
        _append_flag(flags, "cutoff_estimated_from_density_drop")
        return _ModeValleyDetection(
            mode_x=float(x_grid[peak_idx]),
            mode_rt=peak_rt,
            cutoff_x=float(x_grid[cutoff_idx]),
            cutoff_rt=cutoff_rt,
            valley_depth=valley_depth,
            early_cluster_size=early_cluster_size,
            method="density_drop",
        )

    return None


def _find_histogram_first_bin_front(
    rt_values: np.ndarray,
    flags: list[str],
    params: VoidDetectionParams,
) -> Optional[_ModeValleyDetection]:
    """Fallback detection for strongly enriched first histogram bins."""

    if rt_values.size < max(params.min_cluster_size, 10):
        return None

    bins = min(60, max(10, int(np.sqrt(rt_values.size))))
    counts, edges = np.histogram(rt_values, bins=bins)
    if counts.size < 3:
        return None

    first_count = int(counts[0])
    second_count = int(counts[1])
    rest_counts = counts[1:]
    rest_mean = float(np.mean(rest_counts)) if rest_counts.size else 0.0
    rest_after_second_counts = counts[2:]
    rest_after_second_mean = float(np.mean(rest_after_second_counts)) if rest_after_second_counts.size else 0.0
    if rest_mean <= 0:
        return None

    first_fraction = float(first_count / rt_values.size)
    if first_fraction < params.histogram_peak_min_fraction and "possible_early_outlier" not in flags:
        return None

    use_two_bin_front = False
    use_second_bin_t0 = False
    first_two_bins_fraction = float((first_count + second_count) / rt_values.size)
    first_two_mean = float((first_count + second_count) / 2.0)
    single_bin_front = (
        first_count >= params.histogram_peak_ratio_to_second_bin * max(second_count, 1)
        and first_count >= params.histogram_peak_ratio_to_rest_mean * rest_mean
    )
    two_bin_front = (
        rest_after_second_mean > 0
        and first_two_bins_fraction >= params.histogram_first_two_bins_min_fraction
        and first_two_mean >= params.histogram_two_bin_combined_ratio_to_rest_mean * rest_after_second_mean
        and first_count >= params.histogram_two_bin_ratio_to_rest_mean * rest_after_second_mean
        and second_count >= params.histogram_two_bin_ratio_to_rest_mean * rest_after_second_mean
    )
    outlier_two_bin_front = (
        "possible_early_outlier" in flags
        and rest_after_second_mean > 0
        and first_fraction <= params.histogram_sparse_first_bin_max_fraction
        and first_two_bins_fraction >= params.histogram_first_two_bins_min_fraction
        and first_two_mean >= params.histogram_two_bin_combined_ratio_to_rest_mean * rest_after_second_mean
        and second_count >= params.histogram_second_bin_ratio_to_rest_mean * rest_after_second_mean
    )
    if not single_bin_front and not two_bin_front and not outlier_two_bin_front:
        _append_flag(flags, "no_histogram_first_bin_enrichment")
        return None
    if two_bin_front and not single_bin_front:
        use_two_bin_front = True
    if outlier_two_bin_front and not single_bin_front:
        use_two_bin_front = True
        use_second_bin_t0 = True

    cutoff_limit_rt = min(
        float(np.quantile(rt_values, params.fallback_max_cutoff_quantile)),
        float(np.max(rt_values) * params.histogram_max_cutoff_fraction_of_run),
    )
    required_cluster_size = max(
        params.min_cluster_size,
        int(ceil(params.min_cluster_fraction * rt_values.size)),
    )

    if use_second_bin_t0:
        peak_rt = float(np.median(rt_values[(rt_values >= edges[1]) & (rt_values <= edges[2])]))
    else:
        peak_rt = float(np.median(rt_values[(rt_values >= edges[0]) & (rt_values <= edges[1])]))
    valley_depth = 0.0
    cutoff_rt: Optional[float] = None
    reference_count = float((first_count + second_count) / 2.0) if use_two_bin_front else float(first_count)

    if use_two_bin_front:
        conservative_rt = float(edges[2])
        if conservative_rt <= cutoff_limit_rt:
            cutoff_rt = conservative_rt
            third_count = int(counts[2]) if counts.size > 2 else second_count
            valley_depth = 1.0 - float(third_count / max(reference_count, 1.0))
            _append_flag(flags, "cutoff_estimated_from_first_two_histogram_bins")
    else:
        search_stop = min(counts.size - 1, params.histogram_valley_bin_limit + 1)
        for idx in range(1, search_stop):
            if counts[idx] <= counts[idx - 1] and counts[idx] <= counts[idx + 1]:
                candidate_rt = float(edges[idx + 1])
                if candidate_rt <= cutoff_limit_rt:
                    cutoff_rt = candidate_rt
                    valley_depth = 1.0 - float(counts[idx] / max(reference_count, 1.0))
                    _append_flag(flags, "cutoff_estimated_from_histogram_valley")
                    break

        if cutoff_rt is None:
            for idx in range(1, search_stop):
                if counts[idx] <= params.histogram_drop_fraction * first_count:
                    candidate_rt = float(edges[idx + 1])
                    if candidate_rt <= cutoff_limit_rt:
                        cutoff_rt = candidate_rt
                        valley_depth = 1.0 - float(counts[idx] / max(reference_count, 1.0))
                        _append_flag(flags, "cutoff_estimated_from_histogram_drop")
                        break

        if cutoff_rt is None:
            conservative_rt = float(edges[1])
            if conservative_rt <= cutoff_limit_rt:
                cutoff_rt = conservative_rt
                valley_depth = 1.0 - float(second_count / max(reference_count, 1.0))
                _append_flag(flags, "cutoff_estimated_from_first_histogram_bin")

    if cutoff_rt is None:
        return None

    early_cluster_size = int(np.sum(rt_values <= cutoff_rt))
    if early_cluster_size < required_cluster_size:
        _append_flag(flags, "histogram_early_cluster_too_small")
        return None

    if use_second_bin_t0:
        _append_flag(flags, "t0_estimated_as_median_of_second_histogram_bin")
    else:
        _append_flag(flags, "t0_estimated_as_median_of_first_histogram_bin")
    return _ModeValleyDetection(
        mode_x=peak_rt,
        mode_rt=peak_rt,
        cutoff_x=cutoff_rt,
        cutoff_rt=cutoff_rt,
        valley_depth=valley_depth,
        early_cluster_size=early_cluster_size,
        method="histogram_first_bin",
    )


def _estimate_confidence(
    mode_rt: float,
    cutoff_rt: float,
    valley_depth: float,
    early_cluster_size: int,
    rt_values: np.ndarray,
    flags: list[str],
    params: VoidDetectionParams,
    detection_method: str = "first_valley",
) -> str:
    """Estimate qualitative confidence for the accepted mode/valley pair."""

    required_cluster_size = max(
        params.min_cluster_size,
        int(ceil(params.min_cluster_fraction * rt_values.size)),
    )
    cutoff_quantile = float(np.mean(rt_values <= cutoff_rt))
    mode_quantile = float(np.mean(rt_values <= mode_rt))
    max_t0_rt = _max_t0_rt(rt_values, params)
    max_cutoff_rt = (
        float(np.max(rt_values) * params.histogram_max_cutoff_fraction_of_run)
        if detection_method == "histogram_first_bin"
        else _max_cutoff_rt(rt_values, params)
    )

    max_cutoff_quantile = (
        params.fallback_max_cutoff_quantile
        if detection_method in {"density_drop", "histogram_first_bin"}
        else params.max_cutoff_quantile
    )

    if (
        valley_depth < params.min_valley_depth
        or early_cluster_size < required_cluster_size
        or cutoff_quantile > max_cutoff_quantile
        or mode_rt > max_t0_rt
        or cutoff_rt > max_cutoff_rt
        or "possible_early_outlier" in flags
    ):
        return "low"

    if detection_method in {"density_drop", "histogram_first_bin"}:
        if (
            valley_depth >= max(params.min_valley_depth, 0.40)
            and early_cluster_size >= params.min_cluster_size
            and cutoff_quantile <= params.fallback_max_cutoff_quantile
            and "narrow_rt_range" not in flags
            and mode_rt < cutoff_rt
        ):
            return "medium"
        return "low"

    if (
        valley_depth >= 0.40
        and early_cluster_size >= params.min_cluster_size
        and mode_quantile <= params.max_t0_quantile
        and cutoff_quantile <= params.max_cutoff_quantile
        and "narrow_rt_range" not in flags
        and mode_rt < cutoff_rt
    ):
        return "high"

    return "medium"


def _classify_rt_values(
    rt_values: np.ndarray,
    valid_mask: np.ndarray,
    putative_t0: Optional[float],
    cutoff: Optional[float],
    confidence: str,
    params: VoidDetectionParams,
) -> pd.Series:
    """Classify RT values relative to the putative early-elution front."""

    pseudo_k = np.full(rt_values.shape, np.nan, dtype=float)
    if putative_t0 is not None and putative_t0 > 0:
        pseudo_k = (rt_values - putative_t0) / putative_t0

    statuses: list[str] = []
    for rt_value, is_valid, pk in zip(rt_values, valid_mask, pseudo_k, strict=False):
        if not is_valid or not np.isfinite(rt_value):
            statuses.append("uncertain")
            continue
        if putative_t0 is None or cutoff is None:
            statuses.append("uncertain")
            continue
        if confidence == "low" and not params.allow_low_confidence_classification:
            statuses.append("uncertain")
            continue
        if rt_value <= cutoff:
            statuses.append("putatively_non_retained")
            continue
        if np.isfinite(pk) and pk < params.weak_retention_k_threshold:
            statuses.append("weakly_retained")
            continue
        if np.isfinite(pk) and pk >= params.weak_retention_k_threshold:
            statuses.append("retained")
            continue
        statuses.append("uncertain")

    return pd.Series(statuses, dtype="object")


def plot_void_detection(
    rt_values: np.ndarray,
    result: VoidDetectionResult,
    params: Optional[VoidDetectionParams] = None,
):
    """Plot a histogram and KDE diagnostic for RT-based putative void detection."""

    import matplotlib.pyplot as plt

    resolved_params = params or VoidDetectionParams()
    rt_values = np.asarray(rt_values, dtype=float)
    valid_rt_values = np.sort(rt_values[np.isfinite(rt_values) & (rt_values > resolved_params.min_rt)])

    fig, ax = plt.subplots(figsize=(9, 5))
    if valid_rt_values.size:
        bins = min(60, max(10, int(np.sqrt(valid_rt_values.size))))
        ax.hist(valid_rt_values, bins=bins, color="#9ecae1", edgecolor="#4a6fa5", alpha=0.75)

        kde_result = _fit_kde(_transform_rt_values(valid_rt_values, resolved_params), resolved_params)
        if kde_result is not None:
            x_grid, density = kde_result
            rt_grid = _inverse_transform_rt_values(x_grid, resolved_params)
            ax_density = ax.twinx()
            ax_density.plot(rt_grid, density, color="#d62728", linewidth=2.0, label="KDE")
            ax_density.set_ylabel("KDE density")
            ax_density.legend(loc="upper right")

    if result.putative_t0 is not None:
        ax.axvline(result.putative_t0, color="#2ca02c", linestyle="--", linewidth=2, label="putative_t0")
    if result.early_elution_cutoff is not None:
        ax.axvline(
            result.early_elution_cutoff,
            color="#ff7f0e",
            linestyle=":",
            linewidth=2,
            label="early_elution_cutoff",
        )

    flags_text = "; ".join(result.flags) if result.flags else "none"
    subtitle = shorten(f"flags: {flags_text}", width=115, placeholder="...")
    ax.set_xlabel("Retention time (min)")
    ax.set_ylabel("Count")
    fig.suptitle(f"Putative void detection ({result.confidence} confidence)", fontsize=14, y=0.98)
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=8, color="#555555")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return fig


def _has_possible_early_outlier(rt_values: np.ndarray) -> bool:
    """Detect a single unusually low RT separated from the remaining values."""

    if rt_values.size < 4:
        return False

    gaps = np.diff(rt_values)
    first_gap = float(gaps[0])
    remaining_gaps = gaps[1:]
    positive_remaining_gaps = remaining_gaps[remaining_gaps > 0]
    baseline_gap = float(np.median(positive_remaining_gaps)) if positive_remaining_gaps.size else 0.0
    if baseline_gap == 0.0:
        baseline_gap = 1e-6

    early_tail_fraction = float(np.mean(rt_values <= rt_values[0]))
    return early_tail_fraction <= (1.0 / rt_values.size) and first_gap > max(0.5, 5.0 * baseline_gap)


def _append_flag(flags: list[str], flag: str) -> None:
    """Append a flag while preserving insertion order and uniqueness."""

    if flag not in flags:
        flags.append(flag)
