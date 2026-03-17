import pydicom
import SimpleITK as sitk
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage
from tqdm import tqdm

import argparse
import shutil
import os

from src.data.preprocessing import (
    invert_img,
    burned_text_removal,
    clip_img,
    zscore_normalize,
    ResizeTransform,
)


# load model
def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    architecture = checkpoint.get("architecture", "unet++")
    model = build_model(model_name, architecture=architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


# build model
def build_model(model_name: str, architecture: str = "unet") -> torch.nn.Module:

    encoders = {"unet_resnet34": "resnet34", "unet_resnet50": "resnet50"}

    if model_name not in encoders:
        raise ValueError(f"Unknown model_name {model_name}")

    encoder = encoders[model_name]
    kwargs = dict(
        encoder_name=encoder, encoder_weights="imagenet", in_channels=1, classes=1
    )

    if architecture == "unet":
        return smp.Unet(**kwargs)
    elif architecture == "unet++":
        return smp.UnetPlusPlus(**kwargs)
    else:
        raise ValueError(f"Unknown architecture {architecture}")


# Preprocess Dicom Manually
def preprocess(dcm_path: str, laterality: str) -> tuple[torch.Tensor, dict, bool]:
    flipped = laterality is not None and laterality.strip().upper() == "R"

    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    # Fix Monochrome 1 (invert image)
    if ds.PhotometricInterpretation.lower() == "monochrome1":
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


# Run inference
def run_inference(model: torch.nn.Module, img: torch.Tensor, device: str) -> np.ndarray:

    img = img.to(device)  # send image to device
    img = img.unsqueeze(0)  # add batch dim

    model = model.to(device)
    with torch.no_grad():
        pred = model(img)

    pred = (torch.sigmoid(pred) > 0.5).float()

    return pred.squeeze().cpu().numpy().astype(np.uint8)


# Reverse resize back to original dimensions
def postprocess(
    mask: np.uint8,
    metadata: dict,
    flipped: bool,
    smooth: bool = False,
) -> np.ndarray:

    resize = ResizeTransform()
    mask = resize.reverse(mask.astype(np.float32), metadata)
    if flipped:
        mask = np.fliplr(mask)

    binary = (mask > 0.5).astype(np.uint8)

    # keep largest connected component
    if smooth:
        labeled, num_features = ndimage.label(binary)
        if num_features > 1:
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            binary = (labeled == np.argmax(sizes) + 1).astype(np.uint8)

        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
        binary = ndimage.binary_closing(binary, iterations=2).astype(np.uint8)

    return binary


# Save segmentation to output dir as .nrrd
def save_nrrd(mask: np.uint8, dcm_path: str, out_path: str):

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
    smooth: bool = False,
) -> np.ndarray:
    img, metadata, flipped = preprocess(dcm_path, laterality)
    mask = run_inference(model, img, device)

    return postprocess(mask, metadata, flipped, smooth=smooth)


def predict_batch(
    checkpoint_path: str,
    csv_path: str,
    data_dir: str,
    output_dir: str,
    smooth: bool = False,
    device: str = None,
):
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
        mask = predict(model, dcm_path, row["laterality"], device, smooth=smooth)
        stem = os.path.splitext(os.path.basename(dcm_path))[0]
        save_nrrd(mask, dcm_path, os.path.join(output_dir, f"{stem}.nrrd"))
        shutil.copy(dcm_path, os.path.join(output_dir, os.path.basename(dcm_path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smooth", action="store_true", default=False)

    args = parser.parse_args()

    predict_batch(
        args.checkpoint,
        args.csv,
        args.data_dir,
        args.output_dir,
        device=args.device,
        smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
