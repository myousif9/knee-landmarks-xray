import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import albumentations as A

import argparse

from src.data.dataset import KneeDataset
from src.segmentation.model import build_model


def boundary_loss(pred, target):
    """Compute boundary-weighted binary cross-entropy loss.

    Detects mask boundaries using a Laplacian filter, then computes BCE
    between the predicted logits and the boundary map. Encourages the model
    to focus on accurately delineating edges rather than just filling the mask interior.

    Args:
        pred (torch.Tensor): Raw logit predictions of shape (B, 1, H, W).
        target (torch.Tensor): Binary segmentation masks of shape (B, 1, H, W).

    Returns:
        torch.Tensor: Scalar boundary loss.
    """

    laplacian = (
        torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(pred.device)
    )
    boundary = F.conv2d(target.float(), laplacian, padding=1).abs().clamp(0, 1)
    return F.binary_cross_entropy_with_logits(pred, boundary)


def train(
    data_path: str,
    cache_dir: str,
    checkpoint_dir: str,
    model_name: str = "unet_resnet34",
    architecture: str = "unet",
    version: str = "v1",
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 0.001,
    boundary_loss_weight: float = 0.0,
    early_stopping_patience: int = 20,
):
    """Train a tibia segmentation model and save checkpoints.

    Loads segmented images from the MRKR dataset metadata CSV, applies an
    80/20 train/val split, and trains with BCE + optional boundary loss.
    Saves the best checkpoint by validation loss and applies early stopping.
    TensorBoard logs are written to ``checkpoint_dir/logs``.

    Args:
        data_path (str): Root directory of the dataset.
        cache_dir (str): Directory for caching preprocessed images.
        checkpoint_dir (str): Directory to save model checkpoints and logs.
        model_name (str, optional): Encoder identifier. Defaults to ``"unet_resnet34"``.
        architecture (str, optional): Decoder architecture — ``"unet"`` or ``"unet++"``.
            Defaults to ``"unet"``.
        version (str, optional): Version tag appended to the checkpoint filename.
            Defaults to ``"v1"``.
        epochs (int, optional): Maximum number of training epochs. Defaults to 200.
        batch_size (int, optional): Training batch size. Defaults to 16.
        lr (float, optional): Initial learning rate. Defaults to 0.001.
        boundary_loss_weight (float, optional): Weight of the boundary loss term.
            Set to 0 to use BCE only. Defaults to 0.0.
        early_stopping_patience (int, optional): Number of epochs without validation
            improvement before stopping. Defaults to 20.
    """

    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "logs"))

    NUM_WORKERS = 2 if torch.cuda.is_available() else 0

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load CSV, filter to is_segmented == True
    lateral_path = os.path.join(data_path, "tables", "MRKR_lateral_image_metadata.csv")
    lateral_df = pd.read_csv(lateral_path)
    segmented_df = lateral_df[lateral_df["is_segmented"]].reset_index(drop=True)

    # Build image_paths, mask_paths, laterality lists
    image_paths = [
        os.path.join(data_path, "images", p)
        for p in segmented_df["dicom_path"].tolist()
    ]
    mask_paths = segmented_df["mask_path"].tolist()
    laterality = segmented_df["laterality"].tolist()

    # Split train/val indicies
    train_idx, val_idx = train_test_split(
        range(len(image_paths)), test_size=0.2, random_state=42
    )

    # dataset augmentations
    train_aug = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.ElasticTransform(alpha=60, sigma=6, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(p=0.2),
        ]
    )

    # Create KneeDataset
    train_ds = KneeDataset(
        image_paths=[image_paths[i] for i in train_idx],
        mask_paths=[mask_paths[i] for i in train_idx],
        laterality=[laterality[i] for i in train_idx],
        transform=train_aug,
        cache_dir=cache_dir,
        target_size=512,
    )

    val_ds = KneeDataset(
        image_paths=[image_paths[i] for i in val_idx],
        mask_paths=[mask_paths[i] for i in val_idx],
        laterality=[laterality[i] for i in val_idx],
        cache_dir=cache_dir,
        target_size=512,
    )

    # Create DataLoader

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    model = build_model(model_name, architecture=architecture).to(device=DEVICE)

    # loss/criterion
    criterion = DiceCELoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )
    scaler = GradScaler()

    best_val_dice = 0.0
    early_stopping_counter = 0

    for epoch in range(epochs):
        print(f"EPOCH: {epoch + 1}")
        model.train(True)

        train_loss = 0.0

        for imgs, masks, _ in tqdm(train_loader, "train"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type=DEVICE):
                ypred = model(imgs)

                loss = criterion(ypred, masks)
                if boundary_loss_weight > 0.0:
                    loss = loss + boundary_loss_weight * boundary_loss(ypred, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        print(f" train_loss: {train_loss / len(train_loader):.4f}")
        writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)

        model.eval()
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc="val"):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

                ypred = model(imgs)
                preds = (torch.sigmoid(ypred) > 0.5).float()
                dice_metric(y_pred=preds, y=masks)

        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        scheduler.step(val_dice)

        print(f" val_dice: {val_dice:.4f}")
        writer.add_scalar("Dice/val", val_dice, epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "architecture": architecture,
                    "epoch": epoch,
                    "val_dice": val_dice,
                },
                os.path.join(checkpoint_dir, f"{model_name}_{version}_best.pt"),
            )
            print(" -> saved checkpoint")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stoppping at epoch {epoch + 1}")
                break

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", required=True, help="Directory for caching preprocessed images."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory to save checkpoints and TensorBoard logs.",
    )
    parser.add_argument(
        "--architecture",
        default="unet",
        help="Decoder architecture: 'unet' or 'unet++'. Default: unet.",
    )
    parser.add_argument(
        "--model_name",
        default="unet_resnet34",
        help="Encoder identifier. Default: unet_resnet34.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Version tag appended to the checkpoint filename. Default: v1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs. Default: 200.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size. Default: 16."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate. Default: 0.001."
    )
    parser.add_argument(
        "--boundary_loss_weight",
        type=float,
        default=0.0,
        help="Weight of boundary loss term. 0 disables it. Default: 0.0.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Epochs without val improvement before stopping. Default: 20.",
    )

    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
