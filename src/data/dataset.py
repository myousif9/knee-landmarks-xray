import torch
from torch.utils.data import Dataset
import pydicom
import SimpleITK as sitk
import numpy as np
import os
from .preprocessing import (
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
        laterality=None,  # list of "L"/"R", one label per image
        transform=None,
        cache_dir: str = None,
        target_size: int = 512,
    ):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.laterality = laterality
        self.resize = ResizeTransform(target_size)
        self.transform = transform
        self.cache_dir = cache_dir

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_paths)

    def _cache_path(self, index) -> str:
        stem = os.path.splitext(os.path.basename(self.image_paths[index]))[0]
        return os.path.join(self.cache_dir, f"{stem}.npz")

    def __getitem__(self, index):
        flipped = (
            self.laterality is not None
            and self.laterality[index].strip().upper() == "R"
        )

        # 0. Loading cached image and mask and making them tensors
        if self.cache_dir is not None:
            cache_file = self._cache_path(index)
            if os.path.exists(cache_file):
                with np.load(cache_file) as data:
                    img = data["img"].copy()
                    mask = data["mask"].copy() if "mask" in data else None

                # 1. Augmentations
                if self.transform is not None and mask is not None:
                    augmented = self.transform(image=img, mask=mask)
                    img, mask = augmented["image"], augmented["mask"]

                # 2. Convert to tensor
                tensor_img = torch.from_numpy(img).unsqueeze(0).float()
                tensor_mask = (
                    torch.from_numpy(mask).unsqueeze(0).float()
                    if mask is not None
                    else None
                )

                return tensor_img, tensor_mask, flipped

        # 1. Load dicom image
        ds = pydicom.dcmread(self.image_paths[index])
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
        img, metadata = self.resize.forward(img)

        # 7. Load, resize and pad mask
        if self.mask_paths is not None:
            mask_nrrd = sitk.ReadImage(self.mask_paths[index])
            mask = (sitk.GetArrayFromImage(mask_nrrd).squeeze(0) == 1).astype(
                np.float32
            )

            if flipped:
                mask = np.fliplr(mask)

            mask = self.resize.forward_mask(mask, metadata)

        if self.cache_dir is not None:
            if self.mask_paths is not None:
                np.savez(self._cache_path(index), img=img, mask=mask)
            else:
                np.savez(self._cache_path(index), img=img)

        # Augmentations
        if self.transform is not None and self.mask_paths is not None:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        tensor_mask = (
            torch.from_numpy(mask).unsqueeze(0).float()
            if self.mask_paths is not None
            else None
        )

        tensor_img = torch.from_numpy(img).unsqueeze(0).float()

        return tensor_img, tensor_mask, flipped
