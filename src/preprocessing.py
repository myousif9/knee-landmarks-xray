import numpy as np
import cv2 as cv
import sys


def normalize_img(img_float: np.float32) -> np.uint8:

    p1, p99 = np.percentile(img_float, [1, 99])
    img_clip = np.clip(img_float, p1, p99)

    # Inew = (I - I.min) * (newmax - newmin)/(I.max - I.min)  + newmin
    img_norm = (img_clip - img_clip.min()) / (img_clip.max() - img_clip.min())

    img_uint8 = (img_norm * 255).astype(np.uint8)

    return img_uint8, p1, p99


def burned_text_removal(
    float_img: np.float32, max_area_ratio=0.01, dilation_iter=1
) -> np.float32:

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
        thresh_img.astype(np.uint8), connectivity=8
    )

    text_mask = np.zeros_like(img_uint8)

    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]

        # Skip very large regions (likely anatomy)
        if area > max_area_ratio * (h * w):
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


if __name__ == "__main__":
    burned_text_removal(sys.argv[1])
