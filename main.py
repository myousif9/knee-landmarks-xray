import torch
from src.utils import laterality_to_orientation, orientation_to_laterality
from src.segmentation.model import load_model
from src.segmentation.predict import predict
from src.landmarks.pipeline import run_pipeline
from src.landmarks.qc import compute_qc

import os
import pandas as pd
import matplotlib.pyplot as plt

from typing import Literal
import argparse
from tqdm import tqdm


_orientation = Literal["right", "left"]
_pts_method = Literal["medial", "lateral", "posterior_cortex"]


def run(
    model_path,
    dcm_path,
    orientation: _orientation,
    pts_method: _pts_method = "medial",
    device=None,
):
    """Run segmentation and shape analysis on a single DICOM file.

    Args:
        model_path (str): Path to the model checkpoint file.
        dcm_path (str): Path to the DICOM file.
        orientation (_orientation): Laterality of the bone — ``"left"`` or ``"right"``.
        pts_method (_pts_method, optional): Shaft region selection method for PTS.
            Defaults to ``"medial"``.
        device (str | None, optional): Inference device. Auto-detected if None.

    Returns:
        LandmarkResult: PCA axes, boundary conditions, eikonal maps, and PTS result.
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    laterality = orientation_to_laterality(orientation)

    model = load_model(model_path, device)
    mask = predict(model, dcm_path, laterality, device, fill_close=True)
    result = run_pipeline(mask, orientation=orientation, pts_method=pts_method)
    return result


def run_batch(
    model_path,
    csv_path,
    output_dir,
    pts_method: _pts_method = "medial",
    device=None,
    save_plot: bool = False,
):
    """Run the full pipeline on a batch of DICOM files listed in a CSV.

    Loads the model once, iterates over each row, appends PTS results to a
    shared ``results.csv``, and optionally saves per-image plots.

    The CSV must contain ``dicom_path`` and ``laterality`` columns.

    Args:
        model_path (str): Path to the model checkpoint file.
        csv_path (str): Path to the CSV file listing images to process.
        output_dir (str): Directory to write ``results.csv`` and optional plots.
        pts_method (_pts_method, optional): Shaft region selection method for PTS.
            Defaults to ``"medial"``.
        device (str | None, optional): Inference device. Auto-detected if None.
        save_plot (bool, optional): If True, saves PTS and boundary condition plots
            as PNG files for each image. Defaults to False.
    """

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    csv_out = os.path.join(output_dir, "results.csv")
    qc_out = os.path.join(output_dir, "qc.csv")

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    model = load_model(model_path, device)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        stem = os.path.splitext(os.path.basename(row["dicom_path"]))[0]
        laterality = row["laterality"]
        orientation = laterality_to_orientation(laterality)
        mask = predict(model, row["dicom_path"], laterality, device, fill_close=True)
        result = run_pipeline(mask, orientation=orientation, pts_method=pts_method)
        qc = compute_qc(result)

        if save_plot:
            fig, ax = plt.subplots(figsize=(6, 10))
            result.pts.plot(ax=ax)
            fig.savefig(
                os.path.join(output_dir, f"{stem}_pts.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            fig, axes = result.bc.plot_all()
            fig.savefig(
                os.path.join(output_dir, f"{stem}_bc.png"), dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        pd.DataFrame(
            [
                {
                    "dicom": row["dicom_path"],
                    "orientation": orientation,
                    "pts_method": result.pts.method,
                    "pts_angle": result.pts.angle,
                }
            ]
        ).to_csv(csv_out, mode="a", header=not os.path.exists(csv_out), index=False)

        pd.DataFrame(
            [
                {
                    "dicom": row["dicom_path"],
                    **qc.to_dict(),
                }
            ]
        ).to_csv(qc_out, mode="a", header=not os.path.exists(qc_out), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run the knee PTS pipeline on a single DICOM file."
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint file.")
    parser.add_argument("--dicom", required=True, help="Path to DICOM file.")
    parser.add_argument(
        "--orientation",
        choices=["left", "right"],
        required=True,
        help="Laterality of the bone.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. 'cuda', 'mps', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save PTS plot, boundary condition plot, and results CSV. Optional.",
    )
    parser.add_argument(
        "--pts_method",
        choices=["medial", "lateral", "posterior_cortex"],
        default="medial",
        help="Shaft region selection method for PTS computation. Default: medial.",
    )

    args = parser.parse_args()

    result = run(
        args.model,
        args.dicom,
        args.orientation,
        args.pts_method,
        args.device,
    )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.dicom))[0]

        # save PTS plot
        fig, ax = plt.subplots(figsize=(6, 10))
        result.pts.plot(ax=ax)
        fig.savefig(
            os.path.join(args.output_dir, f"{stem}_pts.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # save boundary conditions plot
        fig, axes = result.bc.plot_all()
        fig.savefig(
            os.path.join(args.output_dir, f"{stem}_bc.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        csv_path = os.path.join(args.output_dir, "results.csv")
        pd.DataFrame(
            [
                {
                    "dicom": args.dicom,
                    "orientation": args.orientation,
                    "pts_method": result.pts.method,
                    "pts_angle": result.pts.angle,
                }
            ]
        ).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

    print(f"PTS ({result.pts.method}): {result.pts.angle:.2f} degrees")


if __name__ == "__main__":
    main()
