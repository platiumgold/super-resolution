import random
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import classic_algos.bicubic_interpolation as bicubic
import classic_algos.lanczos as lanczos

from skimage.color import rgb2ycbcr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / (mse + 1e-20))

def loss_psnr_graphic(train_loss, psnr_metric):
    plt.figure()
    plt.suptitle("Результаты")

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(psnr_metric, label='Val PSNR')
    plt.title('PSNR')
    plt.xlabel('Epochs (x5)')
    plt.ylabel('dB')
    plt.legend()
    plt.grid(True)

    plt.show()


def model_to_baseline_compare(model_class, model_path, dataset, mode='rgb', model_args=None, scale=2, zoom_size=144,
                     random_crop=True):
    if model_args is None:
        model_args = {}

    full_idx = random.randint(0, len(dataset.file_paths) - 1)
    img_path = dataset.file_paths[full_idx]
    print(f"File: {img_path}")

    with rasterio.open(img_path) as src:
        image_raw = src.read()  # (C, H, W)

    if mode == 'y':
        image_hwc = np.transpose(image_raw, (1, 2, 0)).astype(np.uint8)
        image_y = rgb2ycbcr(image_hwc)[:, :, 0]
        image_norm = image_y.astype(np.float32) / 255.0

        H, W = image_norm.shape
        H = H - (H % scale)
        W = W - (W % scale)
        image_norm = image_norm[:H, :W]

        image_hwc_input = image_norm[:, :, np.newaxis]  # (H, W, 1)
        hr_tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).to(device)
        cmap_viz = 'gray'

    else:  # rgb
        image_norm = image_raw.astype(np.float32) / 255.0
        C, H, W = image_norm.shape

        H = H - (H % scale)
        W = W - (W % scale)
        image_norm = image_norm[:, :H, :W]

        image_hwc_input = np.transpose(image_norm, (1, 2, 0))  # (H, W, C)
        hr_tensor = torch.from_numpy(image_norm).unsqueeze(0).to(device)
        cmap_viz = None

    lr_h = H // scale
    lr_w = W // scale

    lr_hwc = bicubic.SR_bicubic(image_hwc_input, lr_w, lr_h,
                                preserve_range=True, output_dtype=np.float32)

    lr_chw = np.transpose(lr_hwc, (2, 0, 1))
    lr_tensor = torch.from_numpy(lr_chw).unsqueeze(0).float().to(device)  # (1, C, H, W)

    model_vis = model_class(**model_args).to(device)

    try:
        weights = torch.load(model_path, map_location=device)
        model_vis.load_state_dict(weights)
        model_vis.eval()
    except Exception as e:
        print(f"ERROR loading model: {e}")

    with torch.no_grad():
        sr_model_tensor = model_vis(lr_tensor)

    sr_bicubic_hwc = bicubic.SR_bicubic(lr_hwc, H, W,
                                        preserve_range=True, output_dtype=np.float32)
    sr_lanczos_hwc = lanczos.SR_lanczos(lr_hwc, H, W,
                                        preserve_range=True, output_dtype=np.float32)

    sr_bicubic_tensor = torch.from_numpy(np.transpose(sr_bicubic_hwc, (2, 0, 1))).unsqueeze(0).to(device)
    sr_lanczos_tensor = torch.from_numpy(np.transpose(sr_lanczos_hwc, (2, 0, 1))).unsqueeze(0).to(device)

    psnr_model = PSNR(sr_model_tensor, hr_tensor).item()
    psnr_bicubic = PSNR(sr_bicubic_tensor, hr_tensor).item()
    psnr_lanczos = PSNR(sr_lanczos_tensor, hr_tensor).item()

    if random_crop:
        top = random.randint(0, H - zoom_size)
        left = random.randint(0, W - zoom_size)
    else:
        top = max((H - zoom_size) // 2, 0)
        left = max((W - zoom_size) // 2, 0)

    def get_crops(tensor_img):
        full = tensor_img.squeeze().detach().cpu().numpy()
        if full.ndim == 3:  # (C, H, W) -> RGB
            full = np.transpose(full, (1, 2, 0))
            crop = full[top:top + zoom_size, left:left + zoom_size, :]
        else:  # (H, W) -> Y channel
            crop = full[top:top + zoom_size, left:left + zoom_size]
        return full, crop

    hr_full, hr_crop = get_crops(hr_tensor)
    sr_full, sr_crop = get_crops(sr_model_tensor)
    bi_full, bi_crop = get_crops(sr_bicubic_tensor)
    la_full, la_crop = get_crops(sr_lanczos_tensor)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    model_name = f"{mode.upper()} {model_class.__name__}"

    fig.suptitle(
        f'{mode.upper()} SR: {Path(img_path).name}\nPSNR (full) — {model_name}: {psnr_model:.2f} | Bicubic: {psnr_bicubic:.2f} | Lanczos: {psnr_lanczos:.2f}')

    imgs_full = [hr_full, sr_full, bi_full, la_full]
    titles = ['Original HR', f'{model_name} (Ours)', 'Bicubic', 'Lanczos']

    for ax, img, title in zip(axes[0], imgs_full, titles):
        ax.imshow(np.clip(img, 0, 1), cmap=cmap_viz)
        ax.axis('off')
        ax.set_title(title)

    imgs_crop = [hr_crop, sr_crop, bi_crop, la_crop]
    titles_zoom = ['Zoom HR', f'Zoom {model_name}', 'Zoom Bicubic', 'Zoom Lanczos']

    for ax, img, title in zip(axes[1], imgs_crop, titles_zoom):
        ax.imshow(np.clip(img, 0, 1), cmap=cmap_viz)
        ax.axis('off')
        ax.set_title(title)

    rect = plt.Rectangle((left, top), zoom_size, zoom_size, linewidth=2, edgecolor='r', facecolor='none')
    axes[0, 0].add_patch(rect)

    plt.tight_layout()
    plt.show()

    print(f"Zoom Area: Top={top}, Left={left}, Size={zoom_size}")
