import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from src.data.dataset import KneeDataset
from tqdm import tqdm
import albumentations as A

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
ON_COLAB = "COLAB_GPU" in os.environ

MODEL_NAME = "unet_resnet34"

EPOCHS = 200
BATCH_SIZE = 16 if ON_COLAB else 4
NUM_WORKERS = 0 if ON_COLAB is False and torch.backends.mps.is_available() else 2

LR = 0.001

if ON_COLAB:
    MRKR_DATA_PATH = "/content/drive/MyDrive/projects/knee-landmarks-xray"
    CACHE_DIR = (
        "/content/drive/MyDrive/projects/knee-landmarks-xray/cache/segmentations"
    )
    CHECKPOINT_DIR = (
        "/content/drive/MyDrive/projects/knee-landmarks-xray/checkpoints/segmentations"
    )
else:
    MRKR_DATA_PATH = "../../../../../Volumes/HDD_02/datasets/emory_mrkr"
    CACHE_DIR = "cache/segmentations"
    CHECKPOINT_DIR = "checkpoints/segmentations"


# Load CSV, filter to is_segmented == True
lateral_path = os.path.join(MRKR_DATA_PATH, "tables", "MRKR_lateral_image_metadata.csv")
lateral_df = pd.read_csv(lateral_path)
segmented_df = lateral_df[lateral_df["is_segmented"]].reset_index(drop=True)

# Build image_paths, mask_paths, laterality lists
image_paths = [
    os.path.join(MRKR_DATA_PATH, "images", p)
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
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
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
    cache_dir=CACHE_DIR,
    target_size=512,
)

val_ds = KneeDataset(
    image_paths=[image_paths[i] for i in val_idx],
    mask_paths=[mask_paths[i] for i in val_idx],
    laterality=[laterality[i] for i in val_idx],
    cache_dir=CACHE_DIR,
    target_size=512,
)

# Create DataLoader

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


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


model = build_model(MODEL_NAME).to(device=DEVICE)

# loss/criterion
criterion = DiceCELoss(sigmoid=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_dice = 0.0

for epoch in range(EPOCHS):
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

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": MODEL_NAME,
                "epoch": epoch,
                "val_dice": val_dice,
            },
            os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.pt"),
        )
        print(" -> saved checkpoint")
