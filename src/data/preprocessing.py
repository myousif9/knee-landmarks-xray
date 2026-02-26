import numpy as np
import cv2 as cv


def clip_img(
    img: np.ndarray, lower_centile: int = 1, upper_centile: int = 99
) -> tuple[np.ndarray, float, float]:

    lower_val, upper_val = np.percentile(img, [lower_centile, upper_centile])
    img_clip = np.clip(img, lower_val, upper_val)

    return img_clip, lower_val, upper_val


def normalize_img(img: np.ndarray) -> tuple[np.ndarray, float, float]:

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

    cleaned_float = img.copy()

    cleaned_float = cv.inpaint(cleaned_float, mask_bool, 5, cv.INPAINT_TELEA)

    return cleaned_float


def crop_border(img: np.ndarray, margin: int = 10) -> tuple[np.ndarray, dict]:

    if margin == 0:
        return img

    h, w = img.shape

    cropped = img[margin : h - margin, margin : w - margin]

    metadata = {"crop_margin": margin, "orig_h": h, "orig_w": w}

    return cropped, metadata


def invert_img(img: np.ndarray) -> np.ndarray:
    return img.max() - img


# def invert_monochrome1(dcm_img):
#     return


class ResizeTransform:
    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def forward(
        self, img: np.ndarray, padding_constant: int = 0
    ) -> tuple[np.ndarray, dict]:
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

    def reverse(self, img: np.ndarray, metadata: dict) -> np.ndarray:
        # return to original

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
