import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from monai.losses import DiceCELoss
from tqdm import tqdm
import albumentations as A

import argparse

from src.data.dataset import KneeDataset
from src.data.sampling import sample_diverse_rows, stratify_labels_for_split
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

    channels = target.shape[1]
    laplacian = (
        torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .repeat(channels, 1, 1, 1)
        .to(pred.device)
    )
    boundary = (
        F.conv2d(target.float(), laplacian, padding=1, groups=channels)
        .abs()
        .clamp(0, 1)
    )
    return F.binary_cross_entropy_with_logits(pred, boundary)


def dice_from_intersection_denominator(
    intersection: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return (2 * intersection + eps) / (denominator + eps)


def train(
    data_path: str,
    cache_dir: str,
    checkpoint_dir: str,
    manifest_csv: str = None,
    image_col: str = None,
    mask_col: str = None,
    split_col: str = "split",
    metadata_csv: str = None,
    dicom_path_col: str = "dicom_path",
    mask_path_col: str = "mask_path",
    laterality_col: str = "laterality",
    segmented_col: str = "is_segmented",
    image_root: str = None,
    mask_root: str = None,
    sample_size: int = None,
    diversity_columns: list[str] = None,
    numeric_bins: int = 4,
    random_state: int = 42,
    mask_format: str = "nrrd",
    target_labels: list[str] = None,
    classes: int = 1,
    model_name: str = "unet_resnet34",
    architecture: str = "unet",
    version: str = "v1",
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 0.001,
    boundary_loss_weight: float = 0.0,
    early_stopping_patience: int = 20,
):
    """Train a segmentation model and save checkpoints.

    Loads segmented images from metadata, optionally samples a metadata-diverse
    subset, applies an 80/20 train/val split, and trains with DiceCE + optional
    boundary loss.
    Saves the best checkpoint by validation loss and applies early stopping.
    TensorBoard logs are written to ``checkpoint_dir/logs``.

    Args:
        data_path (str): Root directory of the dataset.
        cache_dir (str): Directory for caching preprocessed images.
        checkpoint_dir (str): Directory to save model checkpoints and logs.
        manifest_csv (str, optional): Processed manifest CSV path. If provided,
            it is used instead of the default MRKR metadata CSV.
        image_col (str, optional): Image path column for manifest mode. Overrides
            ``dicom_path_col`` when provided.
        mask_col (str, optional): Mask path column for manifest mode. Overrides
            ``mask_path_col`` when provided.
        split_col (str, optional): Existing split column with train/val/test
            values. If present, train/val rows are used directly.
        metadata_csv (str, optional): Metadata CSV path. Defaults to
            ``data_path/tables/MRKR_lateral_image_metadata.csv``.
        dicom_path_col (str, optional): Column containing DICOM paths.
        mask_path_col (str, optional): Column containing segmentation mask paths.
        laterality_col (str, optional): Column containing ``"L"``/``"R"`` labels.
        segmented_col (str, optional): Boolean column used to keep labelled rows.
            Set to an empty string to skip this filter.
        image_root (str, optional): Root prepended to relative DICOM paths.
            Defaults to ``data_path/images``.
        mask_root (str, optional): Root prepended to relative mask paths.
        sample_size (int, optional): Number of labelled images to sample.
        diversity_columns (list[str], optional): Metadata columns to balance across.
        numeric_bins (int, optional): Quantile bins for numeric diversity columns.
        random_state (int, optional): Seed for sampling and splitting.
        mask_format (str, optional): Mask format passed to ``KneeDataset``.
            Defaults to ``"nrrd"``.
        target_labels (list[str], optional): Labels selected from multilabel NPZ
            masks, in output channel order.
        classes (int, optional): Number of model output channels. Defaults to 1.
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

    if target_labels is not None:
        target_labels = list(target_labels)
        if classes != len(target_labels):
            raise ValueError(
                f"classes={classes} must match len(target_labels)="
                f"{len(target_labels)} for multilabel training."
            )

    # Load metadata and keep rows with available segmentation masks.
    if manifest_csv is not None:
        metadata_csv = manifest_csv
        if image_col is not None:
            dicom_path_col = image_col
        if mask_col is not None:
            mask_path_col = mask_col

    if metadata_csv is None:
        metadata_csv = os.path.join(
            data_path, "tables", "MRKR_lateral_image_metadata.csv"
        )
    image_root = image_root or os.path.join(data_path, "images")

    lateral_df = pd.read_csv(metadata_csv)
    required_cols = [dicom_path_col, mask_path_col, laterality_col]
    missing_cols = [col for col in required_cols if col not in lateral_df.columns]
    if missing_cols:
        raise KeyError(f"Missing metadata columns: {', '.join(missing_cols)}")

    segmented_df = lateral_df.copy()
    if segmented_col:
        if segmented_col in segmented_df.columns:
            segmented_df = segmented_df[segmented_df[segmented_col].astype(bool)]
        elif manifest_csv is None:
            raise KeyError(f"Missing segmented column: {segmented_col}")

    segmented_df = segmented_df[
        segmented_df[dicom_path_col].notna() & segmented_df[mask_path_col].notna()
    ].reset_index(drop=True)
    if segmented_df.empty:
        raise ValueError("No labelled rows found after filtering metadata.")

    if sample_size is not None:
        segmented_df = sample_diverse_rows(
            segmented_df,
            n=sample_size,
            diversity_columns=diversity_columns,
            numeric_bins=numeric_bins,
            random_state=random_state,
        )

    split_strata = None
    if split_col not in segmented_df.columns:
        split_strata = stratify_labels_for_split(
            segmented_df,
            diversity_columns=diversity_columns,
            numeric_bins=numeric_bins,
        )

    def resolve_path(root: str | None, path: str) -> str:
        path = str(path)
        if os.path.isabs(path) or root is None:
            return path
        return os.path.join(root, path)

    # Build image_paths, mask_paths, laterality lists
    image_paths = [
        resolve_path(image_root, p) for p in segmented_df[dicom_path_col].tolist()
    ]
    mask_paths = [
        resolve_path(mask_root, p) for p in segmented_df[mask_path_col].tolist()
    ]
    laterality = segmented_df[laterality_col].tolist()

    # Split train/val indices, using manifest splits when available.
    if split_col in segmented_df.columns:
        split_values = segmented_df[split_col].astype(str).str.lower()
        train_idx = segmented_df.index[split_values == "train"].tolist()
        val_idx = segmented_df.index[split_values == "val"].tolist()
        if not train_idx or not val_idx:
            raise ValueError(
                f"Split column '{split_col}' must contain non-empty train and val rows."
            )
    else:
        train_idx, val_idx = train_test_split(
            range(len(image_paths)),
            test_size=0.2,
            random_state=random_state,
            stratify=split_strata,
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
        mask_format=mask_format,
        target_labels=target_labels,
    )

    val_ds = KneeDataset(
        image_paths=[image_paths[i] for i in val_idx],
        mask_paths=[mask_paths[i] for i in val_idx],
        laterality=[laterality[i] for i in val_idx],
        cache_dir=cache_dir,
        target_size=512,
        mask_format=mask_format,
        target_labels=target_labels,
    )

    # Create DataLoader

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    model = build_model(
        model_name,
        architecture=architecture,
        classes=classes,
    ).to(device=DEVICE)

    # loss/criterion
    criterion = DiceCELoss(sigmoid=True)
    metric_labels = (
        target_labels
        if target_labels is not None
        else [f"class_{i}" for i in range(classes)]
    )

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
        val_intersection = torch.zeros(classes, device=DEVICE)
        val_denominator = torch.zeros(classes, device=DEVICE)
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc="val"):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

                ypred = model(imgs)
                preds = (torch.sigmoid(ypred) > 0.5).float()
                val_intersection += (preds * masks).sum(dim=(0, 2, 3))
                val_denominator += preds.sum(dim=(0, 2, 3)) + masks.sum(
                    dim=(0, 2, 3)
                )

        channel_dice = dice_from_intersection_denominator(
            val_intersection, val_denominator
        )
        val_dice = channel_dice.mean().item()

        scheduler.step(val_dice)

        print(f" val_dice: {val_dice:.4f}")
        writer.add_scalar("Dice/val", val_dice, epoch)
        for label, dice in zip(metric_labels, channel_dice.detach().cpu().tolist()):
            print(f"  val_dice/{label}: {dice:.4f}")
            writer.add_scalar(f"Dice/val_{label}", dice, epoch)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "architecture": architecture,
                    "classes": classes,
                    "mask_format": mask_format,
                    "target_labels": target_labels,
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
        "--data_path",
        default=".",
        help="Root directory of the dataset.",
    )
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
    parser.add_argument(
        "--metadata_csv",
        default=None,
        help="Metadata CSV path. Defaults to data_path/tables/MRKR_lateral_image_metadata.csv.",
    )
    parser.add_argument(
        "--manifest_csv",
        default=None,
        help="Processed manifest CSV path. Overrides the default metadata CSV.",
    )
    parser.add_argument(
        "--image_col",
        default=None,
        help="Image path column in manifest mode. Overrides dicom_path_col.",
    )
    parser.add_argument(
        "--mask_col",
        default=None,
        help="Mask path column in manifest mode. Overrides mask_path_col.",
    )
    parser.add_argument(
        "--split_col",
        default="split",
        help="Optional manifest split column with train/val/test values. Default: split.",
    )
    parser.add_argument(
        "--dicom_path_col",
        default="dicom_path",
        help="Column containing DICOM paths. Default: dicom_path.",
    )
    parser.add_argument(
        "--mask_path_col",
        default="mask_path",
        help="Column containing mask paths. Use your femur mask column here.",
    )
    parser.add_argument(
        "--laterality_col",
        default="laterality",
        help="Column containing L/R laterality. Default: laterality.",
    )
    parser.add_argument(
        "--segmented_col",
        default="is_segmented",
        help="Boolean labelled-row filter column. Set to '' to skip. Default: is_segmented.",
    )
    parser.add_argument(
        "--image_root",
        default=None,
        help="Root prepended to relative DICOM paths. Default: data_path/images.",
    )
    parser.add_argument(
        "--mask_root",
        default=None,
        help="Root prepended to relative mask paths. Default: no root.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional number of labelled images to sample for training.",
    )
    parser.add_argument(
        "--diversity_columns",
        nargs="*",
        default=None,
        help="Metadata columns to balance across when sampling and splitting.",
    )
    parser.add_argument(
        "--numeric_bins",
        type=int,
        default=4,
        help="Quantile bins for numeric diversity columns. Default: 4.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for sampling and train/val split. Default: 42.",
    )
    parser.add_argument(
        "--mask_format",
        default="nrrd",
        choices=["nrrd", "multilabel_npz"],
        help="Mask format passed to KneeDataset. Default: nrrd.",
    )
    parser.add_argument(
        "--target_labels",
        nargs="*",
        default=None,
        help="Labels selected from multilabel NPZ masks, in output channel order.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=1,
        help="Number of model output channels/classes. Default: 1.",
    )

    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
