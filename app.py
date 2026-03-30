import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pydicom
from PIL import Image
import gradio as gr

from src.utils import orientation_to_laterality
from src.segmentation.model import load_model
from src.segmentation.predict import run_inference, postprocess
from src.data.preprocessing import (
    clip_img,
    zscore_normalize,
    ResizeTransform,
    burned_text_removal,
)
from src.landmarks.pipeline import run_pipeline
from src.landmarks.qc import compute_qc

from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download("myousif9/lateral-tibia-xray_model", "model.pt")
# MODEL_PATH = "models/unet_resnet34_v6_best.pt"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

_model = None


def _preprocess_dicom(path: str, laterality: str):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    display = img.copy()

    if (
        getattr(ds, "PhotometricInterpretation", "").lower().replace(" ", "")
        == "monochrome1"
    ):
        img = img.max() - img

    flipped = laterality.strip().upper() == "R"
    if flipped:
        img = np.fliplr(img)

    img = burned_text_removal(img)
    img, _, _ = clip_img(img)
    img = zscore_normalize(img)

    resize = ResizeTransform(target_size=512)
    img, metadata = resize.forward(img)

    return img, metadata, flipped, display


def _preprocess_image(path: str, laterality: str):
    img = np.array(Image.open(path).convert("L")).astype(np.float32)

    display = img.copy()

    flipped = laterality.strip().upper() == "R"
    if flipped:
        img = np.fliplr(img)

    img, _, _ = clip_img(img)
    img = zscore_normalize(img)

    resize = ResizeTransform(target_size=512)
    img, metadata = resize.forward(img)

    return img, metadata, flipped, display


def run(file, orientation: str, pts_method: str):
    plt.close("all")
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH, DEVICE)

    if file is None:
        return None, None, None, "No file uploaded."

    laterality = orientation_to_laterality(orientation)
    ext = os.path.splitext(file.name)[-1].lower()

    try:
        if ext == ".dcm":
            img, metadata, flipped, display = _preprocess_dicom(file.name, laterality)
        elif ext in (".png", ".jpg", ".jpeg"):
            img, metadata, flipped, display = _preprocess_image(file.name, laterality)
        else:
            return None, None, None, f"Unsupported file type: {ext}"

        # normalize display to uint8
        display_uint8 = (
            (display - display.min()) / (display.max() - display.min() + 1e-8) * 255
        ).astype(np.uint8)

        tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_raw = run_inference(_model, tensor, DEVICE)
        mask = postprocess(mask_raw, metadata, flipped, apply_fill_close=True)

        result = run_pipeline(mask, orientation=orientation, pts_method=pts_method)
        qc = compute_qc(result)

    except Exception as e:
        return None, None, None, f"Error: {e}"

    # PTS plot
    fig_pts, ax_pts = plt.subplots(figsize=(5, 9))
    result.pts.plot(ax=ax_pts)
    fig_pts.tight_layout()

    # QC plot
    fig_qc, ax_qc = plt.subplots(figsize=(6, 4))
    qc.plot(ax=ax_qc)
    fig_qc.tight_layout()

    status = f"{'PASSED' if qc.passed else 'FAILED'} — QC score: {qc.score:.2f}  |  PTS ({result.pts.method}): {result.pts.angle:.2f}°"

    return fig_pts, fig_qc, display_uint8, status


with gr.Blocks(title="Knee PTS") as demo:
    gr.Markdown("## Posterior Tibial Slope")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload DICOM / PNG / JPEG",
                file_types=[".dcm", ".png", ".jpg", ".jpeg"],
            )
            orientation_input = gr.Dropdown(
                choices=["left", "right"],
                value="left",
                label="Laterality",
            )
            method_input = gr.Dropdown(
                choices=["medial", "lateral", "posterior_cortex"],
                value="medial",
                label="PTS method",
            )
            image_out = gr.Image(label="Input image", interactive=False)
            run_btn = gr.Button("Run", variant="primary")
            status_box = gr.Textbox(label="Result", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("PTS"):
                    pts_plot = gr.Plot()
                with gr.Tab("QC"):
                    qc_plot = gr.Plot()

    run_btn.click(
        fn=run,
        inputs=[file_input, orientation_input, method_input],
        outputs=[pts_plot, qc_plot, image_out, status_box],
    )


if __name__ == "__main__":
    demo.launch()
