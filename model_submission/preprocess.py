import os
import numpy as np
import pandas as pd
import cv2
import torch

RESIZE = 256
CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# column name variants backend CSV might use
_IMG_COLS = ["image_path", "filepath", "image", "path", "file_name"]
_LAT_COLS = ["Latitude", "latitude", "lat"]
_LON_COLS = ["Longitude", "longitude", "lon"]


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of {candidates} in columns {list(df.columns)}")


def _load_and_preprocess(path):
    """Resize so the short side is 256, center-crop to 224x224, ImageNet-normalize."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # smallest-side resize to RESIZE
    h, w = img.shape[:2]
    scale = RESIZE / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA)

    # center crop
    h, w = img.shape[:2]
    y0 = (h - CROP) // 2
    x0 = (w - CROP) // 2
    img = img[y0 : y0 + CROP, x0 : x0 + CROP]

    # normalize
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    base_dir = os.path.dirname(csv_path)

    img_col = _find_col(df, _IMG_COLS)
    lat_col = _find_col(df, _LAT_COLS)
    lon_col = _find_col(df, _LON_COLS)

    images = []
    for p in df[img_col]:
        full_path = p if os.path.isabs(p) else os.path.join(base_dir, p)
        images.append(_load_and_preprocess(full_path))

    X = torch.tensor(np.stack(images), dtype=torch.float32)
    y = torch.tensor(df[[lat_col, lon_col]].values, dtype=torch.float32)
    return X, y
