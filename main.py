import torch
from src.segmentation.predict import load_model, predict
from src.landmarks.pipeline import run_pipeline

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
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    laterality = "R" if orientation == "right" else "L"

    model = load_model(model_path, device)
    mask = predict(model, dcm_path, laterality, device, smooth=True)
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
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    csv_out = os.path.join(output_dir, "results.csv")

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    model = load_model(model_path, device)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        stem = os.path.splitext(os.path.basename(row["dicom_path"]))[0]
        orientation = (
            "right" if str(row["laterality"]).strip().upper() == "R" else "left"
        )
        laterality = row["laterality"]

        mask = predict(model, row["dicom_path"], laterality, device, smooth=True)
        result = run_pipeline(mask, orientation=orientation, pts_method=pts_method)

        if save_plot:
            fig, ax = plt.subplots(figsize=(6, 10))
            result.pts.plot(ax=ax)
            fig.savefig(
                os.path.join(output_dir, f"{stem}_pts.png"),
                dpi=150,
                bbox_inches="tight",
            )

            fig, ax = result.bc.plot_all()
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


def main():
    parser = argparse.ArgumentParser(description="Knee PTS pipeline")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--dicom", required=True, help="Path to DICOM file")
    parser.add_argument("--orientation", choices=["left", "right"], required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None, help="Directory to save outputs")
    parser.add_argument(
        "--pts_method",
        choices=["medial", "lateral", "posterior_cortex"],
        default="medial",
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
