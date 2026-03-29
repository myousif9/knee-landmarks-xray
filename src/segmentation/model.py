import torch
import segmentation_models_pytorch as smp


# build model
def build_model(model_name: str, architecture: str = "unet++") -> torch.nn.Module:
    """Instantiate a segmentation model with ImageNet-pretrained encoder weights.

    Args:
        model_name (str): Encoder identifier. Supported values:
            ``"unet_resnet34"``, ``"unet_resnet50"``.
        architecture (str, optional): Decoder architecture — ``"unet"`` or ``"unet++"``.
            Defaults to ``"unet++"``.

    Raises:
        ValueError: If ``model_name`` or ``architecture`` is not recognised.

    Returns:
        torch.nn.Module: Uninitialised segmentation model (weights not loaded).
    """

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


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load a segmentation model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to load the model onto.

    Returns:
        torch.nn.Module: Model in eval mode with weights loaded from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    architecture = checkpoint.get("architecture", "unet++")
    model = build_model(model_name, architecture=architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
