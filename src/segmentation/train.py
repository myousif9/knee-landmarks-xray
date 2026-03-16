import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from src.data.dataset import KneeDataset
from tqdm import tqdm
import albumentations as A

import argparse


# model
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


def boundary_loss(pred, target):

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
            early_stopping_counter = 0
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
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
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--architecture", default="unet")
    parser.add_argument("--model_name", default="unet_resnet34")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--boundary_loss_weight", type=float, default=0.0)
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
