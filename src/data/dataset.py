import torch
from torch.utils.data import Dataset
import pydicom
import SimpleITK as sitk
import numpy as np
import os
from preprocessing import (
    invert_img,
    burned_text_removal,
    clip_img,
    zscore_normalize,
    ResizeTransform,
)


class KneeDataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths=None,
        cache_dir: str = None,
        target_size: int = 512,
    ):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize = ResizeTransform(target_size)

        self.cache_dir = cache_dir

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_paths)

    def _cache_path(self, index) -> str:
        stem = os.path.splitext(os.path.basename(self.image_paths[index]))[0]
        return os.path.join(self.cache_dir, f"{stem}.npz")

    def __getitem__(self, index):

        # 0. Loading cached image and mask and making them tensors
        if self.cache_dir is not None:
            cache_file = self._cache_path(index)
            if os.path.exists(cache_file):
                with np.load(cache_file) as data:
                    img = data["img"].copy()
                    mask = data["mask"].copy() if "mask" in data else None

                tensor_img = torch.from_numpy(img).unsqueeze(0).float()
                tensor_mask = (
                    torch.from_numpy(mask).unsqueeze(0).float()
                    if mask is not None
                    else None
                )

                return tensor_img, tensor_mask

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

        # 6. Convert img to tensor
        tensor_img = torch.from_numpy(img).unsqueeze(0).float()

        # 7. Load, resize and pad mask
        if self.mask_paths is not None:
            mask_nrrd = sitk.ReadImage(self.mask_paths[index])
            mask = sitk.GetArrayFromImage(mask_nrrd).astype(np.float32)

            mask = self.resize.forward_mask(mask, metadata)

            tensor_mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            tensor_mask = None

        if self.cache_dir is not None:
            if self.mask_paths is not None:
                np.savez(self._cache_path(index), img=img, mask=mask)
            else:
                np.savez(self._cache_path(index), img=img)

        return tensor_img, tensor_mask
