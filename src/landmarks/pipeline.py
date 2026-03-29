import numpy as np
from dataclasses import dataclass
from src.landmarks.shape import (
    PCAAxes,
    BoundaryConditions,
    compute_pca_axes,
    process_boundary_conditions,
    _ORIENTATION,
)
from src.landmarks.eikonal import EikonalMaps, compute_eikonal
from src.landmarks.pts import PTSResult, compute_pts, _method as _pts_method


@dataclass
class LandmarkResult:
    """Aggregated outputs from the full landmark detection pipeline.

    Attributes:
        pca: PCA axes fitted to the bone mask.
        bc: Post-processed anatomical boundary conditions.
        em: Eikonal travel time maps computed from the boundary conditions.
        pts: Posterior tibial slope measurement.
    """

    pca: PCAAxes
    bc: BoundaryConditions
    em: EikonalMaps
    pts: PTSResult


def run_pipeline(
    mask: np.ndarray,
    orientation: _ORIENTATION = "left",
    pts_method: _pts_method = "medial",
):
    """Run the full tibial shape analysis and PTS measurement pipeline.

    Computes PCA axes, boundary conditions, eikonal coordinate maps, and
    PTS angle from a binary bone segmentation mask.

    Args:
        mask (np.ndarray): 2D binary segmentation mask of the tibia.
        orientation (_ORIENTATION, optional): Laterality of the bone —
            ``"left"`` or ``"right"``. Defaults to ``"left"``.
        pts_method (_pts_method, optional): Shaft region selection method for
            PTS computation. Defaults to ``"medial"``.

    Returns:
        LandmarkResult: PCA axes, boundary conditions, eikonal maps, and PTS result.
    """

    pca = compute_pca_axes(mask, orientation)
    bc = process_boundary_conditions(mask, pca)
    em = compute_eikonal(bc)
    pts = compute_pts(em, bc, method=pts_method)

    return LandmarkResult(pca=pca, bc=bc, em=em, pts=pts)
