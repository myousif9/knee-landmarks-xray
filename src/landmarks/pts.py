import numpy as np
from src.landmarks.shape import BoundaryConditions
from src.landmarks.eikonal import EikonalMaps
import matplotlib.pyplot as plt
from typing import Literal
from dataclasses import dataclass

_method = Literal["medial", "lateral", "posterior_cortex"]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1))))


@dataclass
class PTSResult:
    angle: float
    shaft_coeffs: np.ndarray
    plateau_coeffs: np.ndarray
    shaft_pts: np.ndarray
    plateau_pts: np.ndarray
    method: _method
    mask: np.ndarray

    def plot(self, ax=None):

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))

        # plat_x = self.plateau_pts[:, 1]
        plat_y = self.plateau_pts[:, 0]
        # x_range = np.linspace(plat_x.min(), plat_x.max(), 100)
        bone_x = np.where(self.mask)[1]
        x_range = np.linspace(bone_x.min(), bone_x.max(), 100)
        plat_line = np.poly1d(self.plateau_coeffs)

        y_range = np.linspace(plat_y.mean(), self.shaft_pts[:, 0].max(), 100)
        shaft_line = np.poly1d(self.shaft_coeffs)

        # perpendicular to shaft at plateau level

        shaft_slope = self.shaft_coeffs[0]

        y_intersect = plat_y.mean()
        x_intersect = shaft_line(y_intersect)

        # perpendicular direction: rotate shaft direction 90 degrees
        perp_slope = -1.0 / shaft_slope if shaft_slope != 0 else np.inf
        perp_line = np.poly1d([perp_slope, x_intersect - perp_slope * y_intersect])

        perp_y = np.linspace(y_intersect - 100, y_intersect + 100, 100)

        ax.imshow(self.mask, cmap="gray")
        ax.plot(shaft_line(y_range), y_range, "b-", lw=2, label="shaft axis")
        ax.plot(x_range, plat_line(x_range), "r-", lw=2, label="plateau")
        ax.plot(perp_line(perp_y), perp_y, "g--", lw=2, label="shaft normal")
        ax.set_title(f"PTS ({self.method}): {self.angle:.2f} degrees")
        ax.legend()
        return ax


def compute_pts(
    em: EikonalMaps, bc: BoundaryConditions, method: _method = "medial"
) -> PTSResult:
    t_si = em.t_si
    t_ap = em.t_ap
    mask = em.mask

    # shaft axis

    if method == "medial":
        shaft_mask = (np.abs(t_ap - 0.5) < 0.05) & (t_si > 0.6) & (t_si < 0.9) & mask
    elif method == "lateral":
        shaft_mask = (t_ap < 0.15) & (t_si > 0.6) & (t_si < 0.9) & mask
    elif method == "posterior_cortex":
        shaft_mask = (t_ap > 0.85) & (t_si > 0.6) & (t_si < 0.9) & mask

    shaft_pts = np.argwhere(shaft_mask)
    shaft_coeffs = np.polyfit(shaft_pts[:, 0], shaft_pts[:, 1], 1)

    plateau_mask = bc.superior
    plateau_pts = np.argwhere(plateau_mask)
    plateau_coeffs = np.polyfit(plateau_pts[:, 1], plateau_pts[:, 0], 1)

    shaft_dir = np.array([1.0, shaft_coeffs[0]])
    plat_dir = np.array([plateau_coeffs[0], 1.0])

    angle = _angle_between(shaft_dir, plat_dir) - 90

    return PTSResult(
        angle=abs(angle),
        shaft_coeffs=shaft_coeffs,
        plateau_coeffs=plateau_coeffs,
        shaft_pts=shaft_pts,
        plateau_pts=plateau_pts,
        method=method,
        mask=mask,
    )
