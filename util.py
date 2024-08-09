from pyproj import Transformer
from pathlib import Path
from tqdm import tqdm

import presto
import xarray
import torch
import json
import os


def get_LABELS(label_path):
    unique_labels = set()
    try:
        with open(label_path, 'r') as file:
            labels_data = json.load(file)
            for labels_list in labels_data.values():
                unique_labels.update(labels_list)
    except FileNotFoundError:
        print(f"Error: File not found at {label_path}")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON")

    labels_list = list(unique_labels)
    labels_list.sort()
    LABELS = labels_list

    return LABELS

label_path = 'test_data/processed_data/20170613T101032_labels/all_labels_20170613T101032.json'

LABELS = get_LABELS(label_path)


# 加载标签数据
def load_labels(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)


def process_images(filenames):
    arrays, masks, latlons, image_names, labels_list, dynamic_worlds = [], [], [], [], [], []
    S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    directory_path = Path('test_data/processed_data/20170613T101032_tiff')
    label_data = load_labels('test_data/processed_data/20170613T101032_labels/all_labels_20170613T101032.json')

    for filename in tqdm(filenames):
        tif_file = xarray.open_rasterio(directory_path / filename.strip())
        crs = tif_file.crs.split("=")[-1]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        height, width = tif_file.shape[1], tif_file.shape[2]
        for x_idx in range(width):
            for y_idx in range(height):
                x, y = tif_file.x[x_idx], tif_file.y[y_idx]
                lon, lat = transformer.transform(x, y)
                latlons.append(torch.tensor([lat, lon]))

                s2_data_for_pixel = torch.from_numpy(tif_file.values[:, y_idx, x_idx].astype(int)).float()
                s2_data_with_time_dimension = s2_data_for_pixel.unsqueeze(0)
                x, mask, dynamic_world = presto.construct_single_presto_input(
                    s2=s2_data_with_time_dimension, s2_bands=S2_BANDS
                )
                arrays.append(x)
                masks.append(mask)
                dynamic_worlds.append(dynamic_world)

                # 获取文件的标签并创建one-hot编码
                file_key = Path(filename).stem
                file_labels = label_data.get(file_key, [])
                one_hot_labels = torch.zeros(len(LABELS), dtype=torch.float32)
                for label in file_labels:
                    if label in LABELS:
                        one_hot_labels[LABELS.index(label)] = 1
                labels_list.append(one_hot_labels)
                image_names.append(filename)

    return (
        torch.stack(arrays, axis=0),
        torch.stack(masks, axis=0),
        torch.stack(dynamic_worlds, axis=0),
        torch.stack(latlons, axis=0),
        torch.stack(labels_list, dim=0),
        image_names
    )


