from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import os

from src.data.dataset import KneeDataset


def make_mock_dicom(shape=(256, 256), photometric="MONOCHROME2"):
    ds = MagicMock()
    ds.pixel_array = np.random.randint(0, 4096, shape, dtype=np.uint16).astype(
        np.float32
    )
    ds.PhotometricInterpretation = photometric
    return ds


def make_mock_mask(shape=(1, 256, 256)):
    arr = np.zeros(shape, dtype=np.uint8)
    arr[0, 50:100, 50:100] = 1
    return arr


@pytest.fixture
def image_paths(tmp_path):
    paths = [str(tmp_path / f"img_{i}.dcm") for i in range(3)]
    for p in paths:
        open(p, "w").close()
    return paths


@pytest.fixture
def mask_paths(tmp_path):
    paths = [str(tmp_path / f"img_{i}.nrrd") for i in range(3)]
    for p in paths:
        open(p, "w").close()
    return paths


@pytest.fixture
def cache_dir(tmp_path):
    cache_path = str(tmp_path / "cache")
    return cache_path


def test_len(image_paths):
    ds = KneeDataset(image_paths, target_size=512)
    assert len(ds) == len(image_paths)


@patch("pydicom.dcmread")
def test_output_shape_no_mask(mock_dcmread, image_paths):
    mock_dcmread.return_value = make_mock_dicom()

    ds = KneeDataset(image_paths, target_size=512)
    img, mask, _ = ds[0]

    assert img.shape == (1, 512, 512)
    assert mask is None


@patch("SimpleITK.GetArrayFromImage")
@patch("SimpleITK.ReadImage")
@patch("pydicom.dcmread")
def test_output_shape_with_mask(
    mock_dcmread, mock_read_image, mock_get_array, image_paths, mask_paths
):
    mock_dcmread.return_value = make_mock_dicom()
    mock_get_array.return_value = make_mock_mask()

    ds = KneeDataset(image_paths, mask_paths=mask_paths, target_size=512)
    img, mask, _ = ds[0]

    assert img.shape == (1, 512, 512)
    assert mask.shape == (1, 512, 512)


@patch("pydicom.dcmread")
def test_cache_file_created(mock_dcmread, image_paths, cache_dir):
    mock_dcmread.return_value = make_mock_dicom()

    ds = KneeDataset(image_paths, cache_dir=cache_dir)

    ds[0]

    assert os.path.exists(
        os.path.join(cache_dir, os.path.basename(image_paths[0]).replace("dcm", "npz"))
    )


@patch("pydicom.dcmread")
def test_cache_hit_skips_dicom(mock_dcmread, image_paths, cache_dir):
    mock_dcmread.return_value = make_mock_dicom()

    img = np.random.randint(0, 4096, (512, 512), dtype=np.uint16).astype(np.float32)

    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[50:100, 50:100] = 1

    cache_path = os.path.join(
        cache_dir, os.path.basename(image_paths[0]).replace("dcm", "npz")
    )

    os.makedirs(cache_dir, exist_ok=True)

    np.savez(cache_path, img=img, mask=mask)

    ds = KneeDataset(image_paths, cache_dir=cache_dir)

    img_load, mask_load, _ = ds[0]

    mock_dcmread.assert_not_called()

    assert img_load.shape == (1, 512, 512)
    assert mask_load.shape == (1, 512, 512)


@patch("pydicom.dcmread")
def test_right_laterality_returns_flipped_true(mock_dcmread, image_paths):
    mock_dcmread.return_value = make_mock_dicom()
    lateraltiy = ["R", "R", "R"]

    ds = KneeDataset(image_paths, laterality=lateraltiy)
    _, _, flipped = ds[0]

    assert flipped is True


@patch("pydicom.dcmread")
def test_right_laterality_returns_flipped_false(mock_dcmread, image_paths):
    mock_dcmread.return_value = make_mock_dicom()
    lateraltiy = ["L", "L", "L"]

    ds = KneeDataset(image_paths, laterality=lateraltiy)
    _, _, flipped = ds[0]

    assert flipped is False
