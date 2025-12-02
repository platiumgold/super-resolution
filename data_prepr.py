from rasterio.plot import reshape_as_image
import rasterio


def TifToNumpy(img_path):
    with rasterio.open(img_path) as src:
        data = src.read() # (C, H, W) - (3, 256, 256), dtype - uint8
    image_for_plot = reshape_as_image(data)  # (C, H, W) -> (H, W, C)
    return image_for_plot

