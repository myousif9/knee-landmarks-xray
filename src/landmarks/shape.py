import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.decomposition import PCA
from skimage.measure import find_contours
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Literal

_mode = Literal["or", "replace"]


@dataclass
class BoundaryConditions:
    superior: np.ndarray
    inferior: np.ndarray
    anterior: np.ndarray
    posterior: np.ndarray
    mask: np.ndarray

    def dilate_and_clean(self, iterations=2):
        return BoundaryConditions(
            superior=largest_component(
                binary_dilation(self.superior, iterations=iterations) & self.mask
            ),
            inferior=largest_component(
                binary_dilation(self.inferior, iterations=iterations) & self.mask
            ),
            anterior=largest_component(
                binary_dilation(self.anterior, iterations=iterations) & self.mask
            ),
            posterior=largest_component(
                binary_dilation(self.posterior, iterations=iterations) & self.mask
            ),
            mask=self.mask,
        )

    def apply_border(self, distance=50, mode: _mode = "replace"):
        h, w = self.mask.shape
        near_border = np.zeros_like(self.mask, dtype=bool)
        near_border[:distance, :] = True
        near_border[-distance:, :] = True
        near_border[:, :distance] = True
        near_border[:, -distance:] = True

        band = near_border & self.mask
        if mode == "or":
            inferior = self.inferior | band
        else:
            inferior = band

        return BoundaryConditions(
            superior=self.superior,
            inferior=inferior,
            anterior=self.anterior,
            posterior=self.posterior,
            mask=self.mask,
        )

    def apply_inferior_band(self, pc1, centroid, fraction=0.15):
        all_y, all_x = np.where(self.mask)
        all_coords = np.stack([all_y, all_x], axis=1)
        proj = (all_coords - centroid) @ pc1
        bone_height = proj.max() - proj.min()

        band = np.zeros_like(self.mask, dtype=bool)
        band_mask = proj < proj.min() + bone_height * fraction
        band[all_y[band_mask], all_x[band_mask]] = True
        band &= self.mask

        return BoundaryConditions(
            superior=self.superior,
            inferior=self.inferior | band,
            anterior=self.anterior,
            posterior=self.posterior,
            mask=self.mask,
        )

    def apply_heirarchy(self):
        inferior = self.inferior
        anterior = self.anterior & ~inferior
        posterior = self.posterior & ~inferior
        superior = self.superior & ~inferior & ~anterior & ~posterior

        return BoundaryConditions(
            superior=superior,
            inferior=inferior,
            anterior=anterior,
            posterior=posterior,
            mask=self.mask,
        )

    def inferior_bone_width(self, pc1, pc2, centroid, fraction=0.05):
        all_y, all_x = np.where(self.mask)
        all_coords = np.stack([all_y, all_x], axis=1)
        proj_pc1 = (all_coords - centroid) @ pc1
        proj_pc2 = (all_coords - centroid) @ pc2
        bone_height = proj_pc1.max() - proj_pc1.min()
        near_inf = proj_pc1 < proj_pc1.min() + bone_height * fraction
        if near_inf.sum() > 0:
            return np.ptp(proj_pc2[near_inf])
        return 0.0

    def shrink_mask(self, label="inferior"):
        regions = {
            "superior": self.superior,
            "inferior": self.inferior,
            "anterior": self.anterior,
            "posterior": self.posterior,
        }
        region = regions[label]
        new_mask = self.mask & ~region
        new_boundary = new_mask & binary_dilation(region)

        new_labels = {k: v & new_mask for k, v in regions.items()}
        new_labels[label] = new_boundary

        return BoundaryConditions(
            superior=new_labels["superior"],
            inferior=new_labels["inferior"],
            anterior=new_labels["anterior"],
            posterior=new_labels["posterior"],
            mask=new_mask,
        )

    def plot(self, ax=None, mode="all"):
        if mode == "si":
            imgs = [self.superior, self.inferior]
            colors = ["blue", "red"]
            labels = ["superior", "inferior"]
        elif mode == "ap":
            imgs = [self.anterior, self.posterior]
            colors = ["green", "orange"]
            labels = ["anterior", "posterior"]
        else:
            imgs = [self.superior, self.anterior, self.posterior, self.inferior]
            colors = ["blue", "green", "orange", "red"]
            labels = ["superior", "anterior", "posterior", "inferior"]

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))

        ax.imshow(self.mask, cmap="gray")
        for img, color, label in zip(imgs, colors, labels):
            y, x = np.where(img)
            ax.scatter(x, y, c=color, s=2, label=label)
        ax.legend()
        return ax

    def plot_all(self):

        fig, axes = plt.subplots(1, 2, figsize=(12, 10))

        for ax, mode in zip(axes, ["si", "ap"]):
            self.plot(ax=ax, mode=mode)

        plt.tight_layout()
        return fig, axes

    def to_contour(self, ignore=None):
        ignore = ignore or []
        if isinstance(ignore, str):
            ignore = [ignore]

        mask_contour = self.mask & ~binary_erosion(self.mask)

        def maybe_thin(img, name):
            if name in ignore:
                return img
            return img & mask_contour

        return BoundaryConditions(
            superior=maybe_thin(self.superior, "superior"),
            inferior=maybe_thin(self.inferior, "inferior"),
            anterior=maybe_thin(self.anterior, "anterior"),
            posterior=maybe_thin(self.posterior, "posterior"),
            mask=self.mask,
        )


_ORIENTATION = Literal["left", "right"]


def compute_pca_axes(mask, orientation_: _ORIENTATION = "left"):
    points = np.argwhere(mask)
    centroid = points.mean(axis=0)

    pca = PCA(n_components=2)
    pca.fit(points - centroid)

    pc1 = pca.components_[0]
    pc2 = pca.components_[1]

    # ensure PC1 points toward smaller y (superior/tibial head)
    if pc1[0] > 0:  # pc1[0] is the y component
        pc1 = -pc1

    # ensure PC2 points in correct anterior direction based on laterality
    if orientation_ == "left":
        if pc2[1] < 0:  # pc2[1] is the x component, anterior = larger x for left
            pc2 = -pc2
    else:
        if pc2[1] > 0:  # anterior = smaller x for right
            pc2 = -pc2

    return centroid, pc1, pc2


def plot_pca_axes(mask, centroid, pc1, pc2, scale_l=200, scale_w=100, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 10))

    ax.imshow(mask, cmap="gray")
    c = centroid
    ax.annotate(
        "",
        xy=c[::-1] + scale_l * pc1[::-1],
        xytext=c[::-1],
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.annotate(
        "",
        xy=c[::-1] + scale_w * pc2[::-1],
        xytext=c[::-1],
        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    )
    ax.scatter(*c[::-1], c="yellow", s=50, zorder=5)
    return ax


def largest_component(binary_img: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(binary_img)
    if n <= 1:
        return binary_img
    sizes = ndimage.sum(binary_img, labeled, range(1, n + 1))
    return labeled == (np.argmax(sizes) + 1)


def compute_boundary_conditions(mask, pc1, pc2, centroid, threshold=0.8, debug=False):
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    contour = find_contours(padded.astype(float), level=0.5)
    contour = max(contour, key=len) - 1

    cy, cx = contour[:, 0], contour[:, 1]
    contour_points = np.stack([cy, cx], axis=1)

    tangent = np.gradient(contour, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)

    cos_pc1 = normal @ pc1
    cos_pc2 = normal @ pc2

    proj_pc1_contour = (contour_points - centroid) @ pc1
    proj_pc2_contour = (contour_points - centroid) @ pc2

    superior_bool = cos_pc1 < -threshold
    superior_bool &= proj_pc1_contour > 0

    anterior_bool = cos_pc2 < -threshold
    posterior_bool = cos_pc2 > threshold

    def local_spatial_filter(
        label_bool, proj_pc_contour, side="lower", thresh_std: int = 2
    ):
        if label_bool.any():
            proj = proj_pc_contour[label_bool]
            if side == "lower":
                cutoff = proj.mean() - thresh_std * proj.std()
                label_bool = label_bool & (proj_pc_contour >= cutoff)
            else:
                cutoff = proj.mean() + thresh_std * proj.std()
                label_bool = label_bool & (proj_pc_contour <= cutoff)
        return label_bool

    superior_bool = local_spatial_filter(superior_bool, proj_pc1_contour, side="lower")

    n = len(cx)
    most_inf_proj = proj_pc1_contour[int(np.argmin(proj_pc1_contour))]
    bone_height = proj_pc1_contour.max() - proj_pc1_contour.min()
    tolerance = bone_height * 0.05

    inferior_bool = (
        (cos_pc1 > 0)
        & (proj_pc1_contour < most_inf_proj + tolerance)
        & (np.abs(cos_pc2) < threshold)
    )

    # fill gaps between labelled contour points
    labels = np.full(n, -1)
    labels[superior_bool] = 0
    labels[anterior_bool] = 1
    labels[posterior_bool] = 2
    labels[inferior_bool] = 3

    labeled_idx = np.where(labels >= 0)[0]
    for i in np.where(labels < 0)[0]:
        dists = np.minimum(np.abs(i - labeled_idx), n - np.abs(i - labeled_idx))
        labels[i] = labels[labeled_idx[np.argmin(dists)]]

    superior_bool = labels == 0
    anterior_bool = labels == 1
    posterior_bool = labels == 2
    inferior_bool = labels == 3

    def to_img(mask_bool):
        img = np.zeros_like(mask, dtype=bool)
        ys = cy[mask_bool].round().astype(int).clip(0, mask.shape[0] - 1)
        xs = cx[mask_bool].round().astype(int).clip(0, mask.shape[1] - 1)
        img[ys, xs] = True
        return img

    bc = BoundaryConditions(
        superior=to_img(superior_bool),
        inferior=to_img(inferior_bool),
        anterior=to_img(anterior_bool),
        posterior=to_img(posterior_bool),
        mask=mask.astype(bool),
    )

    if debug:
        return (
            bc,
            {
                "cy": cy,
                "cx": cx,
                "cos_pc1": cos_pc1,
                "cos_pc2": cos_pc2,
                "proj_pc1_contour": proj_pc1_contour,
                "proj_pc2_contour": proj_pc2_contour,
                "superior_bool": superior_bool,
                "inferior_bool": inferior_bool,
                "anterior_bool": anterior_bool,
                "posterior_bool": posterior_bool,
            },
        )
    return bc


def process_boundary_conditions(
    mask,
    pc1,
    pc2,
    centroid,
    threshold=0.8,
    border_distance=50,
    inferior_band_fraction=0.25,
    dilation_iterations=2,
) -> BoundaryConditions:
    bc = compute_boundary_conditions(mask, pc1, pc2, centroid, threshold=threshold)
    width = bc.inferior_bone_width(pc1, pc2, centroid)

    inferior_count = bc.inferior.sum()
    if inferior_count < width * 0.5 or inferior_count > width * 2.0:
        return (
            bc.apply_border(distance=border_distance)
            .apply_inferior_band(pc1, centroid, fraction=inferior_band_fraction)
            .dilate_and_clean(iterations=dilation_iterations)
            .to_contour(ignore="inferior")
            .apply_heirarchy()
            .shrink_mask()
        )
    else:
        return (
            bc.apply_border(distance=border_distance, mode="or")
            .dilate_and_clean(iterations=dilation_iterations)
            .to_contour(ignore="inferior")
            .apply_heirarchy()
            .shrink_mask()
        )
