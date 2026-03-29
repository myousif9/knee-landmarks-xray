import numpy as np
import skfmm
from dataclasses import dataclass
from src.landmarks.shape import BoundaryConditions

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class EikonalMaps:
    """Travel time maps computed from each anatomical boundary via the eikonal equation.

    Stores raw travel times from each of the four boundary conditions, and exposes
    normalised coordinate maps as properties for anatomical localisation within the bone.

    Attributes:
        t_superior (np.ndarray): Travel time from the superior boundary at each mask pixel.
        t_inferior (np.ndarray): Travel time from the inferior boundary at each mask pixel.
        t_anterior (np.ndarray): Travel time from the anterior boundary at each mask pixel.
        t_posterior (np.ndarray): Travel time from the posterior boundary at each mask pixel.
        mask: Binary bone mask.

    Properties:
        t_si (np.ndarray): Normalised superior-inferior coordinate in [0, 1].
            0 = superior, 1 = inferior.
        t_ap (np.ndarray): Normalised anterior-posterior coordinate in [0, 1].
            0 = anterior, 1 = posterior.
        label_map (np.ndrray): Per-pixel label of the nearest boundary (0=superior,
            1=anterior, 2=posterior, 3=inferior). -1 outside the mask.
    """

    t_superior: np.ndarray
    t_inferior: np.ndarray
    t_anterior: np.ndarray
    t_posterior: np.ndarray
    mask: np.ndarray

    @property
    def t_si(self) -> np.ndarray:
        denom = self.t_superior + self.t_inferior
        return np.where(self.mask, self.t_superior / (denom + 1e-8), np.nan)

    @property
    def t_ap(self) -> np.ndarray:
        denom = self.t_anterior + self.t_posterior
        return np.where(self.mask, self.t_anterior / (denom + 1e-8), np.nan)

    @property
    def label_map(self) -> np.ndarray:
        stacked = np.stack(
            [self.t_superior, self.t_anterior, self.t_posterior, self.t_inferior]
        )
        labels = np.argmin(stacked, axis=0)
        return np.where(self.mask, labels, -1)

    def plot(self, ax=None):
        """Plot the nearest-boundary label map overlaid on the mask.

        Colours each pixel by its closest boundary: blue=superior, green=anterior,
        orange=posterior, red=inferior.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.

        Returns:
            matplotlib.axes.Axes: Axes with the plot.
        """

        cmap = mcolors.ListedColormap(["blue", "green", "orange", "red"])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))

        display = np.where(self.mask, self.label_map, np.nan)
        ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
        return ax

    def plot_si(self, ax=None):
        """Plot the normalised superior–inferior coordinate map.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.

        Returns:
            matplotlib.axes.Axes: Axes with the plot.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))
        display = np.where(self.mask, self.t_si, np.nan)
        ax.imshow(display, cmap="RdBu", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("SI coordinate (0=superior, 1=inferior)")
        return ax

    def plot_ap(self, ax=None):
        """Plot the normalised anterior–posterior coordinate map.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.

        Returns:
            matplotlib.axes.Axes: Axes with the plot.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))
        display = np.where(self.mask, self.t_ap, np.nan)
        ax.imshow(display, cmap="RdBu", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("AP coordinate (0=anterior, 1=posterior)")
        return ax

    def plot_coords(self):
        """Plot SI and AP coordinate maps side by side.

        Returns:
            tuple[matplotlib.figure.Figure, np.ndarray]: Figure and array of two Axes.
        """

        fig, axes = plt.subplots(1, 2, figsize=(12, 10))
        self.plot_si(axes[0])
        self.plot_ap(axes[1])
        plt.tight_layout()
        return fig, axes


def travel_time(boundary, mask):
    """Compute eikonal travel time from a boundary region within a mask.

    Encodes the boundary as the zero level set of a signed distance field,
    masks the domain to the bone region, then solves the eikonal equation
    with unit speed to obtain travel times.

    Args:
        boundary (np.ndarray): 2D boolean array where True marks the source boundary.
        mask (np.ndarray): 2D boolean array defining the domain.

    Returns:
        np.ndarray: Float array of travel times from the boundary, defined within the mask.
    """

    phi = np.where(boundary, -1.0, 1.0)
    phi = np.ma.MaskedArray(phi, mask=~mask)
    dist = skfmm.travel_time(phi, np.ones_like(mask, dtype=float))
    return np.array(dist)


def compute_eikonal(bc: BoundaryConditions) -> EikonalMaps:
    """Compute travel time maps from all four anatomical boundaries.

    Solves the eikonal equation independently from each boundary in
    ``BoundaryConditions`` using unit speed within the bone mask.

    Args:
        bc (BoundaryConditions): Post-processed boundary conditions defining
            the source regions and bone mask.

    Returns:
        EikonalMaps: Travel times from each boundary and the bone mask.
    """

    t_superior = travel_time(bc.superior, bc.mask)
    t_inferior = travel_time(bc.inferior, bc.mask)
    t_anterior = travel_time(bc.anterior, bc.mask)
    t_posterior = travel_time(bc.posterior, bc.mask)

    return EikonalMaps(
        t_superior=t_superior,
        t_inferior=t_inferior,
        t_anterior=t_anterior,
        t_posterior=t_posterior,
        mask=bc.mask,
    )
