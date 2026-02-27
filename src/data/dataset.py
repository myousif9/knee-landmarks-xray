import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import decode_image
import pydicom
import SimpleITK as sitk
from tqdm import tqdm
from preprocessing import (
    invert_img,
    burned_text_removal,
    clip_img,
    zscore_normalize,
    ResizeTransform,
)
import numpy as np
import pandas as pd
import os


class KneeDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, target_size=512):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize = ResizeTransform(target_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # 1. Load dicom image
        ds = pydicom.dcmread(self.image_paths[index])
        img = ds.pixel_array.astype(np.float32)

        # Fix Monochrome 1 (invert image)
        if ds.PhotometricInterpretation.lower() == "monochrome1":
            img = invert_img(img)

        # 2. Remove burned in text
        img = burned_text_removal(img)

        # 3. Clip image intensities
        img, _, _ = clip_img(img)

        # 4. Z-score normalize image
        img = zscore_normalize(img)

        # 5. Resize and pad
        img, metadata = self.resize.forward(img)

        # 6. Load, resize and pad mask
        if self.mask_paths is not None:
            mask_nrrd = sitk.ReadImage(self.mask_paths[index])
            mask = sitk.GetArrayFromImage(mask_nrrd).astype(np.float32)

            mask = self.resize.forward_mask(mask, metadata)
        else:
            return tensor_img

        # 7. Convert to tensor

        tensor_img = torch.from_numpy(img).unsqueeze(0).float()
        tensor_mask = torch.from_numpy(mask).unsqueeze(0).float()

        return tensor_img, tensor_mask
