import numpy as np
import cv2 as cv


def clip_img(
    img: np.ndarray, lower_centile: int = 1, upper_centile: int = 99
) -> tuple[np.ndarray, float, float]:
    """Clip image intensities to percentile bounds.

    Args:
        img (np.ndarray): 2D float image array.
        lower_centile (int, optional): Lower percentile for clipping. Defaults to 1.
        upper_centile (int, optional): Upper percentile for clipping. Defaults to 99.

    Returns:
        tuple[np.ndarray, float, float]: Clipped image, lower bound value, upper bound value.
    """

    lower_val, upper_val = np.percentile(img, [lower_centile, upper_centile])
    img_clip = np.clip(img, lower_val, upper_val)

    return img_clip, lower_val, upper_val


def normalize_img(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Clip and min-max normalize image to uint8 [0, 255].

    Args:
        img (np.ndarray): 2D float image array.

    Returns:
        tuple[np.ndarray, float, float]: Normalized uint8 image, lower clip value (p1) and
            upper clip value (p99).
    """
    img_clip, p1, p99 = clip_img(img)

    # Inew = (I - I.min) * (newmax - newmin)/(I.max - I.min)  + newmin

    denom = img_clip.max() - img_clip.min()
    if denom == 0:
        img_norm = np.zeros_like(img_clip)
    else:
        img_norm = (img_clip - img_clip.min()) / denom

    img_uint8 = (img_norm * 255).astype(np.uint8)

    return img_uint8, p1, p99


def zscore_normalize(img: np.ndarray) -> np.ndarray:
    """Standardize image intensities to zero mean and unit variance.

    Returns zeros if the standard deviation is zero (flat image).

    Args:
        img (np.ndarray): 2D float image array.

    Returns:
        np.ndarray: Float array of the same shape with mean approximately 0 and std approximately 1.
    """
    mean = img.mean()
    std = img.std()

    if std == 0:
        return np.zeros_like(img)

    return (img - mean) / std


def burned_text_removal(
    float_img: np.ndarray,
    max_area_ratio: float = 0.01,
    min_area_ratio: float = 1e-5,
    dilation_iter: int = 1,
) -> np.ndarray:
    """Detect and inpaint burned-in text annotation from DICOM image.

    Applies adaptive thresholding and connected component analysis to identify
    small high-intensity regions consistent with burned-in text, then removes
    them via inpainting. Components outside the area ratio are skipped
    (too large = anatomy, too small = noise).

    Args:
        float_img (np.ndarray): 2D float image array.
        max_area_ratio (float, optional): Upper bound on component area as a
            fraction of total image area. Components above this are treated as
            anatomy and ignored. Defaults to 0.01.
        min_area_ratio (float, optional): Lower bound on component area as a
            fraction of total image area. Components below this are treated as
            noise and ignored. Defaults to 1e-5.
        dilation_iter (int, optional): Number of dilation iterations applied to
            the text mask before inpainting, to ensure full coverage of each
            character. Defaults to 1.

    Returns:
        np.ndarray: Float image with burned-in text removed.
    """

    # image normalization

    img = float_img.copy()
    h, w = img.shape

    img_uint8, _, _ = normalize_img(img)

    # image smoothing
    img_blur = cv.blur(img_uint8, (5, 5))

    # thresholding
    thresh_img = cv.adaptiveThreshold(
        img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
    )

    # connected component analysis
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        thresh_img, connectivity=8
    )

    text_mask = np.zeros_like(img_uint8)

    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]

        # Skip very large regions (likely anatomy)
        area_ratio = area / (h * w)
        if area_ratio > max_area_ratio:
            continue
        elif area_ratio < min_area_ratio:
            continue
        else:
            text_mask[labels == i] = 255

    # dilate mask
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv.dilate(text_mask, kernel, iterations=dilation_iter)

    # masking
    mask_bool = (text_mask > 0).astype(np.uint8)

    cleaned_float = cv.inpaint(img, mask_bool, 5, cv.INPAINT_TELEA)

    return cleaned_float


def crop_border(img: np.ndarray, margin: int = 10) -> tuple[np.ndarray, dict]:
    """Crop a uniform border from all sides of an image.

    Args:
        img (np.ndarray): 2D image array.
        margin (int, optional): Number of pixels to remove from each edge.
            if 0, the image is returned unchanged. Defaults to 10.

    Returns:
        tuple[np.ndarray, dict]: Cropped image and metadata dict with keys
            ``crop_margin``, ``orig_h``, ``orig_w`` needed to reverse the crop.
    """
    if margin == 0:
        metadata = {"crop_margin": 0, "orig_h": img.shape[0], "orig_w": img.shape[1]}
        return img, metadata

    h, w = img.shape

    cropped = img[margin : h - margin, margin : w - margin]

    metadata = {"crop_margin": margin, "orig_h": h, "orig_w": w}

    return cropped, metadata


def invert_img(img: np.ndarray) -> np.ndarray:
    """Invert image intensities relative to the array maximum.

    Use to convert MONOCHROME1 DICOM images (where low values = bright)
    to the standard convention (high values = bright).

    Args:
        img (np.ndarray): 2D float image array.

    Returns:
        np.ndarray: Inverted image of the same shape and dtype.
    """
    return img.max() - img


class ResizeTransform:
    """Resize an image to a square target size while preserving aspect ratio.

    Scales the image so its longest side equals `target_size`, then pads
    symmetrically with a constant value to produce a square output. Stores
    the transform metadata needed to reverse the operation.
    """

    def __init__(self, target_size: int = 512):
        """
        Args:
            target_size (int, optional): Side length of the output square in pixels.
                Defaults to 512.
        """

        self.target_size = target_size

    def forward(
        self, img: np.ndarray, padding_constant: int = 0
    ) -> tuple[np.ndarray, dict]:
        """Resize and pad an image to a square of `target_size`.

        Args:
            img (np.ndarray): 2D float image array.
            padding_constant (int, optional): Constant value used for padding. Defaults to 0.

        Returns:
            tuple[np.ndarray, dict]: Resized and padded image, and metadata dict containing
                original dimensions, scale factor, and padding amounts needed for ``reverse``.
        """

        h, w = img.shape

        scale = self.target_size / max(h, w)

        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        resized_img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = np.pad(
            resized_img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=padding_constant,
        )

        metadata = {
            "orig_h": h,
            "orig_w": w,
            "scale": scale,
            "new_h": new_h,
            "new_w": new_w,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }

        return padded, metadata

    def forward_mask(
        self, img: np.ndarray, metadata: dict, padding_constant: int = 0
    ) -> np.ndarray:
        """Apply the same resize and pad to a mask using nearest-neighbour interpolation.

        Args:
            img (np.ndarray): 2D mask array.
            metadata (dict): Metadata returned by ``forward`` for the corresponding image.
            padding_constant (int, optional): Constant value used for padding. Defaults to 0.

        Returns:
            np.ndarray: Resized and padded mask of the same square size.
        """

        resized_img = cv.resize(
            img, (metadata["new_w"], metadata["new_h"]), interpolation=cv.INTER_NEAREST
        )

        padded = np.pad(
            resized_img,
            (
                (metadata["pad_top"], metadata["pad_bottom"]),
                (metadata["pad_left"], metadata["pad_right"]),
            ),
            mode="constant",
            constant_values=padding_constant,
        )

        return padded

    def reverse(self, img: np.ndarray, metadata: dict) -> np.ndarray:
        """Undo the forward transform — remove padding and resize back to original dimensions.

        Uses nearest-neighbour interpolation, suitable for masks.

        Args:
            img (np.ndarray): Square padded array produced by ``forward`` or ``forward_mask``.
            metadata (dict): Metadata returned by ``forward`` for the corresponding image.

        Returns:
            np.ndarray: Array restored to the original ``(orig_h, orig_w)`` dimensions.
        """

        h, w = img.shape

        orig_h = metadata["orig_h"]
        orig_w = metadata["orig_w"]
        pad_top = metadata["pad_top"]
        pad_bottom = metadata["pad_bottom"]
        pad_left = metadata["pad_left"]
        pad_right = metadata["pad_right"]

        unpadded = img[
            pad_top : h - pad_bottom,
            pad_left : w - pad_right,
        ]

        unpadded_resized = cv.resize(
            unpadded, (orig_w, orig_h), interpolation=cv.INTER_NEAREST
        )

        return unpadded_resized
