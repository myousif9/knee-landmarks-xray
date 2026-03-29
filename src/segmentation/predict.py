import pydicom
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

import argparse
import shutil
import os

from src.segmentation.model import load_model
from src.data.preprocessing import (
    invert_img,
    burned_text_removal,
    clip_img,
    zscore_normalize,
    ResizeTransform,
)


# Preprocess Dicom Manually
def preprocess(dcm_path: str, laterality: str) -> tuple[torch.Tensor, dict, bool]:
    """Load and preprocess a DICOM image for model inference.

    Applies the following pipeline: MONOCHROME1 inversion → horizontal flip
    for right-laterality → burned-in text removal → intensity clipping →
    z-score normalisation → resize and pad to 512x512.

    Args:
        dcm_path (str): Path to the DICOM file.
        laterality (str): Laterality of the image — ``"R"`` or ``"L"``.
            Right images are flipped horizontally so the model always sees
            a left-oriented bone.

    Returns:
        tuple[torch.Tensor, dict, bool]: Preprocessed image tensor of shape (1, 512, 512),
            resize metadata needed for postprocessing, and a flag indicating whether
            the image was flipped.
    """

    flipped = laterality is not None and laterality.strip().upper() == "R"

    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    # Fix Monochrome 1 (invert image)
    if (
        getattr(ds, "PhotometricInterpretation", "").lower().replace(" ", "")
        == "monochrome1"
    ):
        img = invert_img(img)

    if flipped:
        img = np.fliplr(img)

    # 2. Remove burned in text
    img = burned_text_removal(img)

    # 3. Clip image intensities
    img, _, _ = clip_img(img)

    # 4. Z-score normalize image
    img = zscore_normalize(img)

    # 5. Resize and pad
    resize = ResizeTransform(target_size=512)
    img, metadata = resize.forward(img)

    tensor_img = torch.from_numpy(img).unsqueeze(0).float()

    return tensor_img, metadata, flipped


def run_inference(model: torch.nn.Module, img: torch.Tensor, device: str) -> np.ndarray:
    """Run a single forward pass and return a binary segmentation mask.

    Args:
        model (torch.nn.Module): Segmentation model in eval mode.
        img (torch.Tensor): Preprocessed image tensor of shape (1, H, W).
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Binary uint8 mask of shape (H, W) with values 0 or 1.
    """

    img = img.to(device)  # send image to device
    img = img.unsqueeze(0)  # add batch dim

    model = model.to(device)
    with torch.no_grad():
        pred = model(img)

    pred = (torch.sigmoid(pred) > 0.5).float()

    return pred.squeeze().cpu().numpy().astype(np.uint8)


def fill_and_close(binary: np.ndarray):
    """Fill interior holes and close small gaps in a binary mask.

    Applies binary hole filling followed by morphological closing to produce
    a clean, solid segmentation mask.

    Args:
        binary (np.ndarray): Binary uint8 mask.

    Returns:
        np.ndarray: Cleaned binary uint8 mask.
    """

    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    binary = ndimage.binary_closing(binary, iterations=2).astype(np.uint8)
    return binary.astype(np.uint8)


def connected_component_filter(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask.

    Args:
        binary (np.ndarray): Binary uint8 mask.

    Returns:
        np.ndarray: Binary uint8 mask containing only the largest component.
            Returns the input unchanged if there is only one component.
    """

    labeled, num_features = ndimage.label(binary)

    if num_features > 1:
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        binary = (labeled == np.argmax(sizes) + 1).astype(np.uint8)

    return binary


# Reverse resize back to original dimensions
def postprocess(
    mask: np.ndarray,
    metadata: dict,
    flipped: bool,
    apply_fill_close: bool = True,
) -> np.ndarray:
    """Reverse preprocessing and clean up the predicted segmentation mask.

    Reverses the resize transform, un-flips right-laterality images, applies
    Gaussian smoothing to soften inference artefacts, then optionally keeps
    only the largest connected component and fills/closes the mask.

    Args:
        mask (np.ndarray): Raw binary mask output from ``run_inference``.
        metadata (dict): Resize metadata returned by ``preprocess``.
        flipped (bool): Whether the image was horizontally flipped during preprocessing.
        apply_fill_close (bool, optional): If True, applies connected component
            filtering and morphological fill-and-close. Defaults to True.

    Returns:
        np.ndarray: Binary uint8 mask in the original image dimensions.
    """

    resize = ResizeTransform()
    mask = resize.reverse(mask.astype(np.float32), metadata)
    if flipped:
        mask = np.fliplr(mask)

    binary = (ndimage.gaussian_filter(mask.astype(float), sigma=5) > 0.5).astype(
        np.uint8
    )

    # keep largest connected component
    if apply_fill_close:
        binary = connected_component_filter(binary)
        binary = fill_and_close(binary)

    return binary


# Save segmentation to output dir as .nrrd
def save_nrrd(mask: np.ndarray, dcm_path: str, out_path: str):
    """Save a 2D segmentation mask as a 3D NRRD file with DICOM spatial metadata.

    Expands the mask to 3D, copies spacing, origin, and direction from the
    source DICOM, and writes to disk as an NRRD file.

    Args:
        mask (np.ndarray): 2D binary segmentation mask.
        dcm_path (str): Path to the source DICOM file — used to copy spatial metadata.
        out_path (str): Output path for the NRRD file.
    """

    mask_3d = mask[np.newaxis, ...]
    sitk_img = sitk.GetImageFromArray(mask_3d)

    dcm_img = sitk.ReadImage(dcm_path)

    sitk_img.SetSpacing(dcm_img.GetSpacing())
    sitk_img.SetOrigin(dcm_img.GetOrigin())
    sitk_img.SetDirection(dcm_img.GetDirection())

    sitk.WriteImage(sitk_img, out_path)


def predict(
    model: torch.nn.Module,
    dcm_path: str,
    laterality: str,
    device: str,
    fill_close: bool = True,
) -> np.ndarray:
    """Run the full segmentation pipeline on a single DICOM file.

    Args:
        model (torch.nn.Module): Loaded segmentation model in eval mode.
        dcm_path (str): Path to the DICOM file.
        laterality (str): Laterality of the image — ``"R"`` or ``"L"``.
        device (str): Device to run inference on.
        fill_close (bool, optional): Apply connected component filtering and
            morphological fill-and-close during postprocessing. Defaults to True.

    Returns:
        np.ndarray: Binary uint8 segmentation mask in the original image dimensions.
    """

    img, metadata, flipped = preprocess(dcm_path, laterality)
    mask = run_inference(model, img, device)

    return postprocess(mask, metadata, flipped, apply_fill_close=fill_close)


def predict_batch(
    checkpoint_path: str,
    csv_path: str,
    data_dir: str,
    output_dir: str,
    fill_close: bool = True,
    device: str = None,
):
    """Run segmentation on a batch of DICOM files listed in a CSV.

    Loads the model once, iterates over each row, saves each mask as an NRRD
    file, and copies the source DICOM to the output directory.

    The CSV must contain ``dicom_path`` and ``laterality`` columns.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        csv_path (str): Path to the CSV file listing images to process.
        data_dir (str): Root directory prepended to each ``dicom_path`` in the CSV.
        output_dir (str): Directory to write NRRD masks and copied DICOMs.
        fill_close (bool, optional): Apply connected component filtering and
            morphological fill-and-close during postprocessing. Defaults to True.
        device (str, optional): Device for inference. Auto-detected if None.
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    os.makedirs(output_dir, exist_ok=True)
    model = load_model(checkpoint_path, device)
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        dcm_path = os.path.join(data_dir, row["dicom_path"])
        mask = predict(
            model, dcm_path, row["laterality"], device, fill_close=fill_close
        )
        stem = os.path.splitext(os.path.basename(dcm_path))[0]
        save_nrrd(mask, dcm_path, os.path.join(output_dir, f"{stem}.nrrd"))
        shutil.copy(dcm_path, os.path.join(output_dir, os.path.basename(dcm_path)))


def main():

    parser = argparse.ArgumentParser(
        description="Batch tibia segmentation from DICOM files."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint file."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file with 'dicom_path' and 'laterality' columns.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory prepended to each dicom_path in the CSV.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write NRRD masks and copied DICOMs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. 'cuda', 'mps', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--fill_close",
        action="store_true",
        default=True,
        help="Apply morphological fill-and-close to the predicted mask.",
    )

    args = parser.parse_args()

    predict_batch(
        args.checkpoint,
        args.csv,
        args.data_dir,
        args.output_dir,
        device=args.device,
        fill_close=args.fill_close,
    )


if __name__ == "__main__":
    main()
