import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA
from skimage.measure import find_contours

from typing import Literal


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


def largest_component(binary_img: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(binary_img)
    if n <= 1:
        return binary_img
    sizes = ndimage.sum(binary_img, labeled, range(1, n + 1))
    return labeled == (np.argmax(sizes) + 1)


def largest_n_components(binary_img, n=2):
    labeled, num = ndimage.label(binary_img)
    if num <= n:
        return binary_img
    sizes = ndimage.sum(binary_img, labeled, range(1, num + 1))
    top_n = np.argsort(sizes)[-n:] + 1
    return np.isin(labeled, top_n)


def compute_boundary_conditions(mask, pc1, pc2, centroid, threshold=0.8, debug=False):
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
    contour = find_contours(padded.astype(float), level=0.5)
    contour = contour[0] - 1

    cy, cx = contour[:, 0], contour[:, 1]

    tangent = np.gradient(contour, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)

    cos_pc1 = normal @ pc1
    cos_pc2 = normal @ pc2

    def make_img(mask_bool, exclude_imgs=None):
        img = np.zeros_like(mask, dtype=bool)
        for y, x in zip(
            cy[mask_bool].round().astype(int), cx[mask_bool].round().astype(int)
        ):
            img[y, x] = True
        img = largest_component(binary_dilation(img, iterations=2) & mask.astype(bool))
        if exclude_imgs:
            for excl in exclude_imgs:
                img &= ~excl
        return img

    superior_bool = cos_pc1 < -threshold
    anterior_bool = cos_pc2 < -threshold
    posterior_bool = cos_pc2 > threshold

    unlabeled = ~superior_bool & ~anterior_bool & ~posterior_bool
    posterior_bool = posterior_bool | (unlabeled & (cos_pc2 > 0))

    unlabeled = ~superior_bool & ~anterior_bool & ~posterior_bool
    proj_pc1_contour = (np.stack([cy, cx], axis=1) - centroid) @ pc1
    inferior_bool = unlabeled & (cos_pc1 > 0) & (proj_pc1_contour < 0)

    superior_img = make_img(superior_bool)
    anterior_img = make_img(anterior_bool, exclude_imgs=[superior_img])
    posterior_img = make_img(posterior_bool, exclude_imgs=[superior_img, anterior_img])
    inferior_img = make_img(
        inferior_bool, exclude_imgs=[superior_img, anterior_img, posterior_img]
    )

    if debug:
        return (
            superior_img,
            inferior_img,
            anterior_img,
            posterior_img,
            {
                "cy": cy,
                "cx": cx,
                "cos_pc1": cos_pc1,
                "cos_pc2": cos_pc2,
                "superior_bool": superior_bool,
                "inferior_bool": inferior_bool,
                "anterior_bool": anterior_bool,
                "posterior_bool": posterior_bool,
            },
        )
    return superior_img, inferior_img, anterior_img, posterior_img
