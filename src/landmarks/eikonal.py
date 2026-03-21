import numpy as np
import skfmm
from dataclasses import dataclass
from src.landmarks.shape import BoundaryConditions

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class EikonalMaps:
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
        cmap = mcolors.ListedColormap(["blue", "green", "orange", "red"])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))

        display = np.where(self.mask, self.label_map, np.nan)
        ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
        return ax

    def plot_si(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))
        display = np.where(self.mask, self.t_si, np.nan)
        ax.imshow(display, cmap="RdBu", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("SI coordinate (0=superior, 1=inferior)")
        return ax

    def plot_ap(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))
        display = np.where(self.mask, self.t_ap, np.nan)
        ax.imshow(display, cmap="RdBu", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("AP coordinate (0=anterior, 1=posterior)")
        return ax

    def plot_coords(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 10))
        self.plot_si(axes[0])
        self.plot_ap(axes[1])
        plt.tight_layout()
        return fig, axes


def travel_time(boundary, mask):
    phi = np.where(boundary, -1.0, 1.0)
    phi = np.ma.MaskedArray(phi, mask=~mask)
    dist = skfmm.travel_time(phi, np.ones_like(mask, dtype=float))
    return np.array(dist)


def compute_eikonal(bc: BoundaryConditions) -> EikonalMaps:

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
