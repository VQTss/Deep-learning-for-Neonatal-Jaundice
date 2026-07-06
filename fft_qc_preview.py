#!/usr/bin/env python3
"""Quick FFT Low-pass Denoise QC on 5 random images."""

import os
import sys
import numpy as np
from PIL import Image

IMAGE_DIR = "/home/quocthai/research-working/NJ-v5/datasets/images_wb_region"
ROI_SIZE = 128
D0 = 30.0
OUTPUT_DIR = "/home/quocthai/research-working/NJ-v5/fft_qc_output"

np.random.seed(42)
all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
all_images_minus1 = [f for f in all_images if "-1" in f]
chosen = list(np.random.choice(all_images_minus1, size=5, replace=False))

def center_crop_roi(image: Image.Image, roi_size: int) -> Image.Image:
    w, h = image.size
    rs = roi_size
    pad_left = max(0, (rs - w) // 2)
    pad_top = max(0, (rs - h) // 2)
    pad_right = max(0, rs - w - pad_left)
    pad_bottom = max(0, rs - h - pad_top)
    if pad_left or pad_top or pad_right or pad_bottom:
        new_img = Image.new("RGB", (w + pad_left + pad_right, h + pad_top + pad_bottom), (0, 0, 0))
        new_img.paste(image, (pad_left, pad_top))
        image = new_img
        w, h = image.size
    left = (w - rs) // 2
    top = (h - rs) // 2
    return image.crop((left, top, left + rs, top + rs))

def _gaussian_lowpass(size: int, d0: float) -> np.ndarray:
    cy = cx = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return np.exp(-(r ** 2) / (2.0 * d0 ** 2))

def fft_denoise_numpy(rgb: np.ndarray, d0: float = 30.0) -> np.ndarray:
    size = rgb.shape[0]
    H = _gaussian_lowpass(size, d0)
    clean = np.zeros_like(rgb, dtype=np.float64)
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(rgb[..., c]))
        clean[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return np.clip(clean, 0, 255).astype(np.uint8)

out_raw = os.path.join(OUTPUT_DIR, "roi_raw_minus1")
out_fft = os.path.join(OUTPUT_DIR, f"roi_fft_d0{int(D0)}_minus1")
os.makedirs(out_raw, exist_ok=True)
os.makedirs(out_fft, exist_ok=True)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(5, 2, figsize=(6, 15))
fig.suptitle(f"FFT Gaussian Low-pass Denoise (d0={D0})\nTop: Original ROI | Bottom: FFT Denoised", fontsize=11)

for i, fname in enumerate(chosen):
    img_path = os.path.join(IMAGE_DIR, fname)
    img = Image.open(img_path).convert("RGB")

    roi_raw = center_crop_roi(img, ROI_SIZE)
    roi_np = np.array(roi_raw).astype(np.float64)
    roi_fft = fft_denoise_numpy(roi_np, d0=D0)

    roi_raw.save(os.path.join(out_raw, fname))
    Image.fromarray(roi_fft).save(os.path.join(out_fft, fname))

    axes[i, 0].imshow(roi_raw)
    axes[i, 0].set_title(f"{fname}\nROI raw", fontsize=9)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(Image.fromarray(roi_fft))
    axes[i, 1].set_title(f"{fname}\nFFT d0={D0}", fontsize=9)
    axes[i, 1].axis("off")

plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, f"comparison_d0{int(D0)}_minus1.png")
fig.savefig(combined_path, dpi=130, bbox_inches="tight")
plt.close(fig)

print(f"\n5 chosen images: {chosen}")
print(f"ROI raw saved to  : {out_raw}/")
print(f"ROI FFT saved to  : {out_fft}/")
print(f"Combined figure  : {combined_path}")
