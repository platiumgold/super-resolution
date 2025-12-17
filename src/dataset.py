import os
import glob
import random
import numpy as np
import torch
from skimage.color import rgb2ycbcr
from torch.utils.data import Dataset
import rasterio
import classic_algos.bicubic_interpolation as bicubic


class SatelliteSRDataset(Dataset):
    def __init__(self, root_dir, mode, hr_patch_size=144, augment=True):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.hr_patch_size = hr_patch_size
        self.augment = augment
        self.file_paths = glob.glob(os.path.join(root_dir, '**', '*.tif'), recursive=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        with rasterio.open(img_path) as src:
            image = src.read()  # C x H x W

        if self.mode == 'y':
            image_hwc = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            image_y = rgb2ycbcr(image_hwc)[:, :, 0]
            image_norm = image_y.astype(np.float32) / 255.0
            image_norm = image_norm[np.newaxis, :, :]
        else:
            image_norm = image.astype(np.float32) / 255.0

        # random crop
        c, h, w = image_norm.shape
        top = random.randint(0, h - self.hr_patch_size)
        left = random.randint(0, w - self.hr_patch_size)
        hr_patch_np = image_norm[:, top:top + self.hr_patch_size, left:left + self.hr_patch_size]

        if self.augment:
            # Flip Horizontal
            if random.random() < 0.5:
                hr_patch_np = np.flip(hr_patch_np, axis=2)

            # Flip Vertical
            if random.random() < 0.5:
                hr_patch_np = np.flip(hr_patch_np, axis=1)

            # rotations
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                hr_patch_np = np.rot90(hr_patch_np, k, axes=(1, 2))

        hr_patch_np = np.ascontiguousarray(hr_patch_np.astype(np.float32))  # C x H x W, [0,1]
        hr_tensor = torch.from_numpy(hr_patch_np).float()

        # LR from augmented HR patch
        hr_patch_hwc = np.transpose(hr_patch_np, (1, 2, 0))  # H x W x C, [0,1]
        lr_size = self.hr_patch_size // 2
        lr_numpy_hwc = bicubic.SR_bicubic(hr_patch_hwc, lr_size, lr_size, preserve_range=True, output_dtype=np.float32)

        if lr_numpy_hwc.ndim == 2:
            lr_numpy_hwc = lr_numpy_hwc[:, :, np.newaxis]

        lr_numpy = np.transpose(lr_numpy_hwc, (2, 0, 1))  # C x h_lr x w_lr
        lr_tensor = torch.from_numpy(lr_numpy.astype(np.float32))

        return lr_tensor, hr_tensor