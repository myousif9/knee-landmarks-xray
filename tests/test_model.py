import torch

from src.segmentation.model import build_model, load_model


def test_build_model_uses_requested_classes():
    model = build_model("unet_resnet34", architecture="unet", classes=2)

    assert model.segmentation_head[0].out_channels == 2


def test_load_model_uses_checkpoint_classes(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    model = build_model("unet_resnet34", architecture="unet", classes=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "unet_resnet34",
            "architecture": "unet",
            "classes": 2,
            "target_labels": ["femur", "tibia"],
            "mask_format": "multilabel_npz",
        },
        checkpoint_path,
    )

    loaded = load_model(str(checkpoint_path), "cpu")

    assert loaded.segmentation_head[0].out_channels == 2


def test_load_model_defaults_old_checkpoints_to_one_class(tmp_path):
    checkpoint_path = tmp_path / "old_model.pt"
    model = build_model("unet_resnet34", architecture="unet", classes=1)

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "unet_resnet34",
            "architecture": "unet",
        },
        checkpoint_path,
    )

    loaded = load_model(str(checkpoint_path), "cpu")

    assert loaded.segmentation_head[0].out_channels == 1
