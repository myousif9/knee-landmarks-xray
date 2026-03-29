import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from dataclasses import dataclass

from src.landmarks.pipeline import LandmarkResult


@dataclass
class QCResult:
    """Weighted quality control assessment of a single PTS pipeline result.

    Attributes:
        checks: Per-check pass/fail flags keyed by check name.
        weights: Per-check integer weights used to compute the score.
        thresholds: Threshold values used for each check.
        mask_area: Number of foreground pixels in the bone mask.
        mask_solidity: Ratio of mask area to its convex hull area. Close to 1
            for a compact, well-segmented bone.
        pca_aspect_ratio: Ratio of bone length to bone width along PCA axes.
            Low values indicate the bone is unusually wide relative to its length.
        superior_span_ratio: Width of the superior boundary relative to the
            full bone width. Low values indicate the plateau is too narrow.
        inferior_span_ratio: Width of the inferior boundary relative to the
            full bone width.
        shaft_pt_count: Number of pixels used to fit the shaft axis line.
        plateau_pt_count: Number of pixels used to fit the plateau line.
        shaft_residual: RMS pixel residual of the shaft line fit. High values
            indicate a curved or noisy shaft region.
        plateau_residual: RMS pixel residual of the plateau line fit.
        ap_slope_diff: Absolute difference between anterior and posterior
            boundary slopes. High values indicate non-parallel AP boundaries.
        ap_length_ratio: Ratio of shorter to longer AP boundary length. Low
            values indicate asymmetric anterior/posterior coverage.
        pts_angle: Measured PTS angle in degrees.
        score: Weighted fraction of checks passed, in the range [0, 1].
        passed: True if score meets the pass threshold.
    """

    checks: dict[str, bool]  # label -> passed
    weights: dict[str, int]
    thresholds: dict
    mask_area: int
    mask_solidity: float
    pca_aspect_ratio: float
    superior_span_ratio: float
    inferior_span_ratio: float
    shaft_pt_count: int
    plateau_pt_count: int
    shaft_residual: float
    plateau_residual: float
    ap_slope_diff: float
    ap_length_ratio: float
    pts_angle: float
    score: float  # 0 - 1
    passed: bool  # score >= threshold

    def plot(self, ax=None):
        """Plot a QC summary table overlaid on a matplotlib axes.

        Renders a table with one row per check showing the metric name, measured
        value, threshold, and a pass/fail indicator. Passing rows are highlighted
        green, failing rows red. The axes title shows the overall score and
        pass/fail status.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new
                figure is created.

        Returns:
            matplotlib.axes.Axes: Axes containing the table.
        """

        def _threshold_str(key, val):
            if isinstance(val, tuple):
                return f"{val[0]}-{val[1]}"
            if key in ("ap_slope_diff", "shaft_residual", "plateau_residual"):
                return f"< {val}"
            return f"> {val}"

        values = {
            "mask_area": self.mask_area,
            "mask_solidity": round(self.mask_solidity, 3),
            "pca_aspect_ratio": round(self.pca_aspect_ratio, 2),
            "superior_span_ratio": round(self.superior_span_ratio, 3),
            "inferior_span_ratio": round(self.inferior_span_ratio, 3),
            "ap_slope_diff": round(self.ap_slope_diff, 3),
            "ap_length_ratio": round(self.ap_length_ratio, 3),
            "shaft_pt_count": self.shaft_pt_count,
            "plateau_pt_count": self.plateau_pt_count,
            "shaft_residual": round(self.shaft_residual, 3),
            "plateau_residual": round(self.plateau_residual, 3),
            "pts_in_range": round(self.pts_angle, 2),
        }

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        ax.axis("off")
        keys = list(values.keys())
        rows = [
            [
                k,
                str(v),
                _threshold_str(k, self.thresholds[k]),
                "✓" if self.checks[k] else "✗",
            ]
            for k, v in values.items()
        ]
        table = ax.table(
            cellText=rows,
            colLabels=["Check", "Value", "Threshold", ""],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(True)

        for (row, col), cell in table.get_celld().items():
            if col == 3 and row > 0:
                cell.set_facecolor(
                    "#d4edda" if self.checks[keys[row - 1]] else "#f8d7da"
                )

        ax.set_title(
            f"QC score: {self.score:.2f} — {'PASSED' if self.passed else 'FAILED'}"
        )
        return ax

    def to_dict(self) -> dict:
        """Return QC metrics as a flat dictionary suitable for CSV output.

        Returns:
            dict: Metric names mapped to rounded values. Includes ``qc_passed``,
                ``qc_score``, and all per-check measurements. ``pts_angle`` is
                included so this dict can be used as a complete result row without
                needing to merge with the PTS result separately.
        """

        return {
            "qc_passed": self.passed,
            "qc_score": round(self.score, 3),
            "mask_area": self.mask_area,
            "mask_solidity": round(self.mask_solidity, 3),
            "pca_aspect_ratio": round(self.pca_aspect_ratio, 2),
            "superior_span_ratio": round(self.superior_span_ratio, 3),
            "inferior_span_ratio": round(self.inferior_span_ratio, 3),
            "ap_slope_diff": round(self.ap_slope_diff, 3),
            "ap_length_ratio": round(self.ap_length_ratio, 3),
            "shaft_pt_count": self.shaft_pt_count,
            "plateau_pt_count": self.plateau_pt_count,
            "shaft_residual": round(self.shaft_residual, 3),
            "plateau_residual": round(self.plateau_residual, 3),
            "pts_angle": round(self.pts_angle, 2),
        }


def compute_qc(
    result: LandmarkResult,
    pass_threshold: float = 0.75,
    weights: dict | None = None,
    thresholds: dict | None = None,
) -> QCResult:
    """Compute a weighted quality control score for a landmark pipeline result.

    Evaluates a set of checks against the segmentation mask, PCA axes,
    boundary conditions, and PTS line fits. Each check is assigned a weight;
    the overall score is the weighted fraction of passing checks.

    Weights and thresholds can be partially overridden — only the keys
    provided are replaced; all others fall back to defaults.

    Args:
        result (LandmarkResult): Output of ``run_pipeline``.
        pass_threshold (float, optional): Minimum score to mark the result as
            passed. Defaults to 0.75.
        weights (dict | None, optional): Per-check integer weights to override
            defaults. Partial overrides are supported.
        thresholds (dict | None, optional): Per-check threshold values to
            override defaults. Partial overrides are supported.

    Returns:
        QCResult: QC metrics, per-check results, weighted score, and pass flag.
    """

    mask = result.bc.mask
    pts_result = result.pts

    # mask area
    mask_area = int(mask.sum())

    # mask_solidity
    coords = np.argwhere(mask)
    hull = ConvexHull(coords)
    mask_solidity = mask_area / hull.volume

    # PCA aspect ratio
    proj_pc1, proj_pc2 = result.pca.project(coords)
    pca_aspect_ratio = np.ptp(proj_pc1) / (np.ptp(proj_pc2) + 1e-8)
    bone_width = np.ptp(proj_pc2)

    # superior span relative to bone width
    sup_coords = np.argwhere(result.bc.superior)
    sup_span = np.ptp(result.pca.project(sup_coords)[1]) if len(sup_coords) > 1 else 0.0
    superior_span_ratio = sup_span / (bone_width + 1e-8)

    # inferior span relative to bone width
    inf_coords = np.argwhere(result.bc.inferior)
    inf_span = np.ptp(result.pca.project(inf_coords)[1]) if len(inf_coords) > 1 else 0.0
    inferior_span_ratio = inf_span / (bone_width + 1e-8)

    # anterior/posterior parallelism
    ant_pts = np.argwhere(result.bc.anterior)
    post_pts = np.argwhere(result.bc.posterior)
    ant_slope = np.polyfit(ant_pts[:, 0], ant_pts[:, 1], 1)[0]
    post_slope = np.polyfit(post_pts[:, 0], post_pts[:, 1], 1)[0]
    ap_slope_diff = float(abs(ant_slope - post_slope))

    # anterior/posterior length similarity
    ant_length = np.ptp(result.pca.project(ant_pts)[0])
    post_length = np.ptp(result.pca.project(post_pts)[0])
    ap_length_ratio = float(
        min(ant_length, post_length) / (max(ant_length, post_length) + 1e-8)
    )

    # shaft fit residual
    shaft_pt_count = len(pts_result.shaft_pts)
    predicted_x = np.polyval(pts_result.shaft_coeffs, pts_result.shaft_pts[:, 0])
    shaft_residual = float(
        np.sqrt(np.mean((pts_result.shaft_pts[:, 1] - predicted_x) ** 2))
    )

    # plateau fit residual
    plateau_pt_count = len(pts_result.plateau_pts)
    predicted_y = np.polyval(pts_result.plateau_coeffs, pts_result.plateau_pts[:, 1])
    plateau_residual = float(
        np.sqrt(np.mean((pts_result.plateau_pts[:, 0] - predicted_y) ** 2))
    )

    _default_weights = {
        "mask_area": 2,
        "mask_solidity": 3,
        "pca_aspect_ratio": 3,
        "superior_span_ratio": 2,
        "inferior_span_ratio": 1,
        "ap_slope_diff": 1,
        "ap_length_ratio": 1,
        "shaft_pt_count": 2,
        "plateau_pt_count": 2,
        "shaft_residual": 3,
        "plateau_residual": 1,
        "pts_in_range": 3,
    }

    _default_thresholds = {
        "mask_area": 5000,
        "mask_solidity": 0.85,
        "pca_aspect_ratio": 1.5,
        "superior_span_ratio": 0.5,
        "inferior_span_ratio": 0.5,
        "ap_slope_diff": 0.3,
        "ap_length_ratio": 0.6,
        "shaft_pt_count": 50,
        "plateau_pt_count": 20,
        "shaft_residual": 15.0,
        "plateau_residual": 15.0,
        "pts_in_range": (0.0, 20.0),
    }

    if weights is None:
        weights = _default_weights
    else:
        weights = {**_default_weights, **weights}

    if thresholds is None:
        thresholds = _default_thresholds
    else:
        thresholds = {**_default_thresholds, **thresholds}

    checks = {
        "mask_area": mask_area > thresholds["mask_area"],
        "mask_solidity": mask_solidity > thresholds["mask_solidity"],
        "pca_aspect_ratio": pca_aspect_ratio > thresholds["pca_aspect_ratio"],
        "superior_span_ratio": superior_span_ratio > thresholds["superior_span_ratio"],
        "inferior_span_ratio": inferior_span_ratio > thresholds["inferior_span_ratio"],
        "ap_slope_diff": ap_slope_diff < thresholds["ap_slope_diff"],
        "ap_length_ratio": ap_length_ratio > thresholds["ap_length_ratio"],
        "shaft_pt_count": shaft_pt_count > thresholds["shaft_pt_count"],
        "plateau_pt_count": plateau_pt_count > thresholds["plateau_pt_count"],
        "shaft_residual": shaft_residual < thresholds["shaft_residual"],
        "plateau_residual": plateau_residual < thresholds["plateau_residual"],
        "pts_in_range": thresholds["pts_in_range"][0]
        <= pts_result.angle
        <= thresholds["pts_in_range"][1],
    }

    score = sum(w for k, w in weights.items() if checks[k]) / sum(weights.values())

    return QCResult(
        checks=checks,
        weights=weights,
        thresholds=thresholds,
        mask_area=mask_area,
        mask_solidity=mask_solidity,
        pca_aspect_ratio=pca_aspect_ratio,
        superior_span_ratio=superior_span_ratio,
        inferior_span_ratio=inferior_span_ratio,
        shaft_pt_count=shaft_pt_count,
        plateau_pt_count=plateau_pt_count,
        shaft_residual=shaft_residual,
        plateau_residual=plateau_residual,
        ap_slope_diff=ap_slope_diff,
        ap_length_ratio=ap_length_ratio,
        pts_angle=pts_result.angle,
        score=score,
        passed=score >= pass_threshold,
    )
