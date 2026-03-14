import os
import pandas as pd
import torch
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
def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "unet_resnet34":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )
    elif model_name == "unet_resnet50":
        return smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )
    else:
        raise ValueError(f"Unknown model {model_name}")


def train(
    data_path: str,
    cache_dir: str,
    checkpoint_dir: str,
    model_name: str = "unet_resnet34",
    version: str = "v1",
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 0.001,
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

    model = build_model(model_name).to(device=DEVICE)

    # loss/criterion
    criterion = DiceCELoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_dice = 0.0

    for epoch in range(epochs):
        print(f"EPOCH: {epoch + 1}")
        model.train(True)

        train_loss = 0.0

        for imgs, masks, _ in tqdm(train_loader, "train"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            ypred = model(imgs)
            loss = criterion(ypred, masks)

            optimizer.zero_grad()
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

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

        print(f" val_dice: {val_dice:.4f}")
        writer.add_scalar("Dice/val", val_dice, epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
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

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model_name", default="unet_resnet34")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
