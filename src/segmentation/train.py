import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from monai.losses import DiceBCELoss
from monai.metrics import DiceMetric
from src.data.dataset import KneeDataset

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
ON_COLAB = "COLAB_GPU" in os.environ

EPOCHS = 50
BATCH_SIZE = 4
NUM_WORKERS = 0 if ON_COLAB is False and torch.backends.mps.is_available() else 2

LR = 0.001

if ON_COLAB:
    MRKR_DATA_PATH = "/content/drive/MyDrive/knee-landmarks-xray"
    CACHE_DIR = "/content/drive/MyDrive/knee-landmarks-xray/cache/segmentations"
    CHECKPOINT_DIR = "/content/drive/MyDrive/knee-landmarks-xray/cache/segmentation"
else:
    MRKR_DATA_PATH = "../../../../../Volumes/HDD_02/datasets/emory_mrkr"
    CACHE_DIR = "cache/segmentations"
    CHECKPOINT_DIR = "checkpoints/segmentations"


# Load CSV, filter to is_segmented == True
lateral_path = os.path.join(MRKR_DATA_PATH, "tables", "MRKR_lateral_image_metadata.csv")
lateral_df = pd.read_csv(lateral_path)
segmented_df = lateral_df[lateral_df["is_segmented"] == True].reset_index(drop=True)

# Build image_paths, mask_paths, laterality lists
image_paths = [
    os.path.join(MRKR_DATA_PATH, "images", p)
    for p in segmented_df["dicom_path"].tolist()
]
mask_paths = segmented_df["mask_path"].tolist()
laterality = segmented_df["laterality"].tolist()

# Create KneeDataset
ds = KneeDataset(
    image_paths=image_paths,
    mask_paths=mask_paths,
    laterality=laterality,
    cache_dir=CACHE_DIR,
    target_size=512,
)

# Split train/val
generator1 = torch.Generator().manual_seed(42)
train, validate = random_split(ds, [0.8, 0.2], generator=generator1)

# Create DataLoader

train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    validate, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# model

model = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1
).to(device=DEVICE)

# loss/criterion
criterion = DiceBCELoss(sigmoid=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_dice = 0.0

for epoch in range(EPOCHS):
    print(f"EPOCH: {epoch + 1}")
    model.train(True)

    train_loss = 0.0

    for imgs, masks, _ in train_loader:
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
        for imgs, masks, _ in val_loader:
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
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
        print(f" -> saved checkpoint")
