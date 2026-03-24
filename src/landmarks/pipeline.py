import numpy as np
from dataclasses import dataclass
from src.landmarks.shape import (
    BoundaryConditions,
    compute_pca_axes,
    process_boundary_conditions,
    _ORIENTATION,
)
from src.landmarks.eikonal import EikonalMaps, compute_eikonal
from src.landmarks.pts import PTSResult, compute_pts, _method as _pts_method


@dataclass
class LandmarkResult:
    bc: BoundaryConditions
    em: EikonalMaps
    pts: PTSResult


def run_pipeline(
    mask: np.ndarray,
    orientation: _ORIENTATION = "left",
    pts_method: _pts_method = "medial",
):
    centroid, pc1, pc2 = compute_pca_axes(mask, orientation)
    bc = process_boundary_conditions(mask, pc1, pc2, centroid)
    em = compute_eikonal(bc)
    pts = compute_pts(em, bc, method=pts_method)

    return LandmarkResult(bc=bc, em=em, pts=pts)
