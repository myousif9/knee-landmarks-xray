from __future__ import annotations

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
class PCAAxes:
    centroid: np.ndarray
    pc1: np.ndarray
    pc2: np.ndarray

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project points onto PCA axes.

        Args:
            points (np.ndarray): Array of shape (N, 2) in image coordinates (y, x).

        Returns:
            tuple[np.ndarray, np.ndarray]: Projection along first principal component and
                second principal component.
        """
        centered = points - self.centroid
        return centered @ self.pc1, centered @ self.pc2

    def cosine_similarity(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute cosine similarity of unit vectors with each PCA axis.

        Args:
            vectors (np.ndarray): Array of shape (N, 2) of unit vectors.

        Returns:
            tuple[np.ndarray, np.ndarray]: Cosine similarity with first principal component
                and second principal component.
        """
        return vectors @ self.pc1, vectors @ self.pc2

    def to_img_coords(self, proj_pc1: np.ndarray, proj_pc2: np.ndarray) -> np.ndarray:
        """Convert PCA projections back to image coordinates.

        Args:
            proj_pc1 (np.ndarray): Projections along first principal component.
            proj_pc2 (np.ndarray): Projections along second principal component.

        Returns:
            np.ndarray: Array of shape (N, 2) in image coordinates (y, x)
        """
        return (
            self.centroid + proj_pc1[:, None] * self.pc1 + proj_pc2[:, None] * self.pc2
        )

    def plot(self, mask: np.ndarray, scale_l=200, scale_w=100, ax=None):
        """Plot PCA axes overlaid on bone mask or raw image pixel array.

        Args:
            mask (np.ndarray): 2D array to display as background — either the binary
                segmentation mask or the raw image pixel array.
            scale_l (int, optional): Length of PC1 arrow in pixels. Defaults to 200.
            scale_w (int, optional): Length of PC2 arrow in pixels. Defaults to 100.
            ax (matplotlib.axes.Axes, optional): Matplotlib axes to plot on.
                If None, new figure is created.

        Returns:
            matplotlib.axes.Axes: Matplotlib axes with plot.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 10))

        ax.imshow(mask, cmap="gray")
        c = self.centroid
        ax.annotate(
            "",
            xy=c[::-1] + scale_l * self.pc1[::-1],
            xytext=c[::-1],
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )
        ax.annotate(
            "",
            xy=c[::-1] + scale_w * self.pc2[::-1],
            xytext=c[::-1],
            arrowprops=dict(arrowstyle="->", color="blue", lw=2),
        )
        ax.scatter(*c[::-1], c="yellow", s=50, zorder=5)
        return ax


_label = Literal["superior", "inferior", "anterior", "posterior"]


@dataclass
class BoundaryConditions:
    """
    Represents the four anatomical boundary regions of a segmented bone mask.

    Each boundary is a boolean array of the same shape as a mask, where True indicates pixels
    belonging to that boundary region. Used as boundary conditions for eikonal equation
    to compute anatomical coordinate maps.

    Attributes:
        superior: Tibial plateau / superior surface boundary.
        inferior: Inferior cut surface boundary.
        anterior: Anterior cortex boundary.
        posterior: Posterior cortex boundary.
        mask: Binary segmentation mask of the bone
    """

    superior: np.ndarray
    inferior: np.ndarray
    anterior: np.ndarray
    posterior: np.ndarray
    mask: np.ndarray

    def dilate_and_clean(self, iterations: int = 2) -> BoundaryConditions:
        """
        Binary dilation of boundary along with connected component selection of the largest label.

        Args:
            iterations (int, optional): Dilation is repeated `iterations` times. Defaults to 2.

        Returns:
            BoundaryConditions: Dilated and cleaned boundary conditions
        """
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

    def apply_border(self, distance=50, mode: _mode = "replace") -> BoundaryConditions:
        """Apply image border pixels within the mask as inferior boundary.

        Pixels within  `distance` of any image edge and inside the mask are treated as
        the inferior boundary — useful for tibia cut surfaces that extend to the image border

        Args:
            distance (int, optional): Width of the border band in pixels. Defaults to 50.
            mode (_mode, optional): How to combine with existing inferior boundary.
                ``"replace"`` discards the exisitng inferior and uses only the border band.
                ``"or"`` unions the border band with the existing inferior. Defaults to "replace".

        Returns:
            BoundaryConditions: New instance with the inferior boundary expanded by the anatomical band.
        """

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

    def apply_inferior_band(self, pca: PCAAxes, fraction=0.10) -> BoundaryConditions:
        """Union the existing inferior boundary with the bottom fraction of the bone along PC1.

        Computes each mask pixel's projection onto PC1, then marks the lowest `fraction`
        of the bone height as part of the inferior boundary. Useful for ensuring the inferior region
        is well-covered when the border band alone misses interior cut-surface pixels.

        Args:
            pca (PCAAxes): PCA axes used to project mask pixels along the superior-inferior axis (PC1).
            fraction (float, optional): Fraction of the total bone height (along PC1) to include in
                the inferior band. Defaults to 0.10.

        Returns:
            BoundaryConditions: New instance with the inferior boundary expanded by the anatomical band.
        """

        all_y, all_x = np.where(self.mask)
        all_coords = np.stack([all_y, all_x], axis=1)
        proj = pca.project(all_coords)[0]
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

    def apply_hierarchy(self) -> BoundaryConditions:
        """Resolve overlapping boundary regions using anatomical priority.

        Inferior takes the highest priority, followed by anterior and posterior,
        with superior being the lowest. Each region has conflicting pixels from
        higher-priority regions removed, ensuring boundaries are mutually exclusive.

        Priority order: inferior > anterior = posterior > superior.

        Returns:
            BoundaryConditions: New instance with non-overlapping boundaries.
        """

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

    def inferior_bone_width(
        self,
        pca: PCAAxes,
        fraction: float = 0.05,
    ) -> float:
        """Measure bone width near the inferior end along PC2

        Projects all mask pixels onto PC1 and PC2, selects the bottom `fraction`
        of bone height along PC1, and returns the peak-to-peak spread of those
        pixels along PC2 (the medial-lateral width).

        Args:
            pca (PCAAxes): PCA axes defining superior-inferior (PC1) and
                anterior-posterior (PC2) directions.
            fraction (float, optional): Fraction of total bone height used to
                define the inferior region. Defaults to 0.05.

        Returns:
            float: Width of the bone near the inferior end in pixels along PC2.
                Returns 0.0 if no pixels fall within the inferior band.
        """

        all_y, all_x = np.where(self.mask)
        all_coords = np.stack([all_y, all_x], axis=1)
        proj_pc1, proj_pc2 = pca.project(all_coords)
        bone_height = proj_pc1.max() - proj_pc1.min()
        near_inf = proj_pc1 < proj_pc1.min() + bone_height * fraction
        if near_inf.sum() > 0:
            return np.ptp(proj_pc2[near_inf])
        return 0.0

    def shrink_mask(self, label: _label = "inferior") -> BoundaryConditions:
        """Remove a boundary region from the mask and redefine its boundary at the new edge.

        Excise the named region from the mask, then dilates that region by one
        pixel into the remaining mask to form a new tight boundary at the cut
        surface. All other boundaries are clipped to the shrunken mask.

        Primarily used to remove the inferior cut surface from the bone interior
        so eikonal distances are computed within the cortical shell only.

        Args:
            label (_label, optional): Which boundary region to excise. Defaults to "inferior".

        Returns:
            BoundaryConditions: New instance with the shrunken mask and recomputed boundary
                for the excised region.
        """
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
        """Plot boundary regions overlaid on the bone mask.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None,
                new figure is created. Defaults to None.
            mode (str, optional): Which boundaries to display.
                ``"si"`` - superior (blue) and inferior (red).
                ``"ap"`` - anterior (green) and posterior (orange).
                ``"all"`` - Defaults to "all".

        Returns:
            matplotlib.axes.Axes: Axes with the plot
        """
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
        """Plot all boundary regions in two side-by-side panels (SI and AP).

        Returns:
            tuple[matplotlib.figure.Figure, np.ndarray]: Figure and array of two Axes,
                left showing superior/inferior and right showing anterior/posterior.
        """

        fig, axes = plt.subplots(1, 2, figsize=(12, 10))

        for ax, mode in zip(axes, ["si", "ap"]):
            self.plot(ax=ax, mode=mode)

        plt.tight_layout()
        return fig, axes

    def to_contour(self, ignore=None) -> BoundaryConditions:
        """Restrict boundary region to the outer contour of the mask.

        Intersects each boundary with the single-pixel contour of the mask
        (mask minus its erosion), ensuring boundaries lie on the bone surface
        rather than in the interior. Regions listed in `ignore` are left as-is.

        Typically used to keep inferior boundary as a filled interior region
        (``ignore="inferior"``) while thinning the other boundaries to the cortex.

        Args:
            ignore (str | list[str] | None, optional): Boundary name(s) to skip.
                Accepts a single string or a list. Defaults to None.

        Returns:
            BoundaryConditions: _description_
        """

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


def compute_pca_axes(mask: np.ndarray, orientation_: _ORIENTATION = "left") -> PCAAxes:
    """Fit PCA axes to a binary bone mask with anatomically consistent orientation.

    Computes the centroid and two prinicple components of the mask pixels,
    then flips axes so that PC1 always points superiorly (decreasing y) and PC2
    always points anteriorly with anterior defined as increasing x for a left
    bone and decreasing x for a right bone.

    Args:
        mask (np.ndarray): 2D binary segmentation mask of the bone.
        orientation_ (_ORIENTATION, optional): Laterality of the bone.
            ``"left"`` or ``"right"``. Determines the anterior direction of PC2.
            Defaults to "left".

    Returns:
        PCAAxes: Centroid and unit vectors for the superior-inferior (PC1)
            and anterior posterior (PC2) axes
    """

    if orientation_ not in ("left", "right"):
        raise ValueError(
            f"Invalid orientation '{orientation_}. Expected 'left' or 'right'."
        )

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

    return PCAAxes(centroid=centroid, pc1=pc1, pc2=pc2)


def largest_component(binary_img: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary image.

    Args:
        binary_img (np.ndarray): 2D binary array.

    Returns:
        np.ndarray: Boolean array of the same shape containing only the
            largest component. Returns the input unchanged if
            there is one component or fewer.
    """

    labeled, n = ndimage.label(binary_img)
    if n <= 1:
        return binary_img
    sizes = ndimage.sum(binary_img, labeled, range(1, n + 1))
    return labeled == (np.argmax(sizes) + 1)


def compute_boundary_conditions(
    mask: np.ndarray, pca: PCAAxes, threshold: float = 0.8, debug=False
):
    """Label the bone contour into four anatomical boundary regions.

    Extracts the outer contour of the mask, computes outward-facing normals,
    and classifies each contour point by cosine similarity with the PCA axes:

    - **Superior**: normals strongly aligned with -PC1 (pointing superiorly),
      restricted to the upper portion of the bone.
    - **Anterior**: normals strongly aligned with -PC2.
    - **Posterior**: normals strongly aligned with +PC2.
    - **Inferior**: normals weakly aligned with PC2 and near the most inferior
      point along PC1.

    Unlabelled contour points are assigned to the nearest labelled neighbour
    along the contour to fill gaps.

    Args:
        mask (np.ndarray): 2D binary segmentation mask of the bone.
        pca (PCAAxes): PCA axes defining the superior-inferior and
            anterior-posterior directions
        threshold (float, optional): Cosine similarity threshold for classifyiing
            a contour normal as belonging to a boundary region. Defaults to 0.8.
        debug (bool, optional): If True, returns a second value - a dict of
            intermediate arrays (contour coordinates, cosine similarities,
            projections, per-region boolean masks). Defaults to False.

    Returns:
        BoundaryConditions: Raw contour-level boundary labels.
        tuple[BoundaryConditions, dict]: If ``debug=True``, also returns the
            intermediate debug arrays.
    """

    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    contour = find_contours(padded.astype(float), level=0.5)
    contour = max(contour, key=len) - 1

    cy, cx = contour[:, 0], contour[:, 1]
    contour_points = np.stack([cy, cx], axis=1)

    tangent = np.gradient(contour, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)

    cos_pc1, cos_pc2 = pca.cosine_similarity(normal)

    proj_pc1_contour, proj_pc2_contour = pca.project(contour_points)

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
    mask: np.ndarray,
    pca: PCAAxes,
    threshold: float = 0.8,
    border_distance: int = 50,
    inferior_band_fraction: float = 0.10,
    dilation_iterations: int = 2,
) -> BoundaryConditions:
    """Run the full boundary condition pipeline from raw mask to eikonal-ready labels.

    Computes raw contour-level boundary labels then applies a fixed post-processing chain:
    border band -> inferior band -> dilation + cleanup -> contour thinning ->
    priority hierarchy -> mask shrink.

    Args:
        mask (np.ndarray): 2D binary segmentation mask of the bone.
        pca (PCAAxes): PCA axes defining anatomical directions.
        threshold (float, optional): Cosine similarity threshold for contour
            classification. Defaults to 0.8.
        border_distance (int, optional): Pixel width of the image-edge band
            applied as the inferior boundary. Defaults to 50.
        inferior_band_fraction (float, optional): Fraction of bone height along
            PC1 added to the inferior boundary. Defaults to 0.10.
        dilation_iterations (int, optional): Number of dilation iteration in
            ``dilate_and_clean``. Defaults to 2.

    Returns:
        BoundaryConditions: Post-processed boundary conditions ready for eikonal computation
    """

    bc = compute_boundary_conditions(mask, pca, threshold=threshold)

    return (
        bc.apply_border(distance=border_distance)
        .apply_inferior_band(pca, fraction=inferior_band_fraction)
        .dilate_and_clean(iterations=dilation_iterations)
        .to_contour(ignore="inferior")
        .apply_hierarchy()
        .shrink_mask()
    )
