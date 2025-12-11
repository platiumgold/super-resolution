import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

SOURCE_DIR = Path.home() / '.data' / 'UCMerced_LandUse' / 'Images'

DEST_DIR = Path.home() / '.data' / 'UCMerced_LandUse_Split'

VAL_SIZE = 0.1
TEST_SIZE = 0.1


def copy_files(files, source_class_path, dest_class_path):
    os.makedirs(dest_class_path, exist_ok=True)
    for file_name in files:
        src = os.path.join(source_class_path, file_name)
        dst = os.path.join(dest_class_path, file_name)
        shutil.copy2(src, dst)


for subset in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DEST_DIR, subset), exist_ok=True)

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for class_name in classes:
    class_path = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]

    train_val_imgs, test_imgs = train_test_split(
        images, test_size=TEST_SIZE, random_state=42, shuffle=True
    )

    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)

    train_imgs, val_imgs = train_test_split(
        train_val_imgs, test_size=relative_val_size, random_state=42, shuffle=True
    )

    copy_files(train_imgs, class_path, os.path.join(DEST_DIR, 'train', class_name))
    copy_files(val_imgs, class_path, os.path.join(DEST_DIR, 'val', class_name))
    copy_files(test_imgs, class_path, os.path.join(DEST_DIR, 'test', class_name))

print(f"Данные сохранены в папку: {DEST_DIR}")
print(f"  {DEST_DIR}/train/ ")
print(f"  {DEST_DIR}/val/ ")
print(f"  {DEST_DIR}/test/ ")

