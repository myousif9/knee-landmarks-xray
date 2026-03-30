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
    """Result of a posterior tibial slope (PTS) measurement.

    Attributes:
        angle (float): Measured PTS angle in degrees.
        shaft_coeffs (np.ndarray): Polynomial coefficients (degree 1) of the shaft axis
            line fit — ``x = shaft_coeffs[0] * y + shaft_coeffs[1]``.
        plateau_coeffs (np.ndarray): Polynomial coefficients (degree 1) of the tibial
            plateau line fit — ``y = plateau_coeffs[0] * x + plateau_coeffs[1]``.
        shaft_pts (np.ndarray): Array of shape (N, 2) of (y, x) pixels used to fit the shaft axis.
        plateau_pts (np.ndarray): Array of shape (M, 2) of (y, x) pixels used to fit the plateau line.
        method: PTS method used — ``"medial"``, ``"lateral"``, or ``"posterior_cortex"``.
        mask (np.ndarray): Binary bone mask used as the plot background.
    """

    angle: float
    shaft_coeffs: np.ndarray
    plateau_coeffs: np.ndarray
    shaft_pts: np.ndarray
    plateau_pts: np.ndarray
    method: _method
    mask: np.ndarray

    def plot(self, ax=None, image=None):
        """Plot the PTS measurement overlaid on the bone mask or a raw image.

        Draws three lines and the source point clouds:
        - Shaft axis (blue) — from plateau level to the inferior shaft.
        - Plateau line (red) — spanning the full bone width.
        - Shaft normal (green dashed) — perpendicular to the shaft at plateau level.
        - Shaft points (blue scatter) and plateau points (red scatter).
        - Intersection point (white dot) at plateau level on the shaft axis.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new
                figure is created.
            image (np.ndarray, optional): Raw image array to use as background.
                If None, the binary bone mask is used.

        Returns:
            matplotlib.axes.Axes: Axes with the plot.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))

        plat_y = self.plateau_pts[:, 0]
        bone_x = np.where(self.mask)[1]
        x_range = np.linspace(bone_x.min(), bone_x.max(), 100)
        plat_line = np.poly1d(self.plateau_coeffs)

        y_range = np.linspace(plat_y.mean(), self.shaft_pts[:, 0].max(), 100)
        shaft_line = np.poly1d(self.shaft_coeffs)

        shaft_slope = self.shaft_coeffs[0]
        y_intersect = plat_y.mean()
        x_intersect = shaft_line(y_intersect)

        perp_slope = -1.0 / shaft_slope if shaft_slope != 0 else np.inf
        perp_line = np.poly1d([perp_slope, x_intersect - perp_slope * y_intersect])
        perp_y = np.linspace(y_intersect - 100, y_intersect + 100, 100)

        bg = image if image is not None else self.mask
        # cmap = "gray" if image is None else None
        ax.imshow(bg, cmap="gray")

        ax.scatter(
            self.shaft_pts[:, 1],
            self.shaft_pts[:, 0],
            s=1,
            c="cyan",
            alpha=0.4,
            label="shaft pts",
        )
        ax.scatter(
            self.plateau_pts[:, 1],
            self.plateau_pts[:, 0],
            s=1,
            c="orange",
            alpha=0.4,
            label="plateau pts",
        )

        ax.plot(shaft_line(y_range), y_range, "b-", lw=2, label="shaft axis")
        ax.plot(x_range, plat_line(x_range), "r-", lw=2, label="plateau")
        ax.plot(perp_line(perp_y), perp_y, "g--", lw=2, label="shaft normal")
        ax.plot(x_intersect, y_intersect, "wo", ms=6)

        ax.set_title(f"PTS ({self.method}): {self.angle:.2f}°")
        ax.legend(markerscale=5)

        h, w = bg.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # inverted — image y goes top to bottom

        return ax


def compute_pts(
    em: EikonalMaps, bc: BoundaryConditions, method: _method = "medial"
) -> PTSResult:
    """Compute the posterior tibial slope (PTS) angle from eikonal coordinate maps.

    Selects shaft pixels using eikonal coordinates, fits a line to define the
    tibial shaft axis, then fits a second line to the superior boundary pixels
    to define the plateau. The PTS angle is the deviation of the plateau from
    perpendicular to the shaft.

    Shaft region definitions by method:
    - ``"medial"``: pixels near the AP midline (t_ap ≈ 0.5), mid-shaft (t_si 0.6–0.9).
    - ``"lateral"``: pixels near the anterior cortex (t_ap < 0.10), mid-shaft.
    - ``"posterior_cortex"``: pixels near the posterior cortex (t_ap > 0.90), mid-shaft.

    Args:
        em (EikonalMaps): Eikonal coordinate maps for the bone.
        bc (BoundaryConditions): Boundary conditions providing the superior boundary
            pixels for the plateau line fit.
        method (_method, optional): Shaft region selection method. Defaults to ``"medial"``.

    Returns:
        PTSResult: Measured angle, line coefficients, source points, and bone mask.
    """

    t_si = em.t_si
    t_ap = em.t_ap
    mask = em.mask

    # shaft axis

    if method == "medial":
        shaft_mask = (np.abs(t_ap - 0.5) < 0.05) & (t_si > 0.6) & (t_si < 0.9) & mask
    elif method == "lateral":
        shaft_mask = (t_ap < 0.10) & (t_si > 0.6) & (t_si < 0.9) & mask
    elif method == "posterior_cortex":
        shaft_mask = (t_ap > 0.90) & (t_si > 0.6) & (t_si < 0.9) & mask
    else:
        raise ValueError(f"Unknown method '{method}'.")

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
