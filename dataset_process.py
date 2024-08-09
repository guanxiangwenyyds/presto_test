import os
import json
import rasterio

import glob
from tqdm import tqdm


def find_labels_json(input_folder):
    """
    Read labels
    :param input_folder: input folder

    :return: list of labels json files
    """

    json_files = glob.glob(os.path.join(input_folder, '*.json'))

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
            if "labels" in data:
                return data["labels"]

    return []


def resample_merge_and_save_labels(input_folder, output_tiff_path, target_pixels=10):
    """

    Resample the merged TIFF file and save the labels to a json file

    param input_folder: path to the folder containing the TIFF file
    param output_tiff_path: path to the folder where the merged TIFF file will be saved
    param output_json_path: path to the folder where the labels will be saved
    param target_pixels: number of pixels to resample the labels for

    return: None

    """

    tiff_files = sorted(glob.glob(os.path.join(input_folder, '*.tif')))
    resampled_data = []

    for tiff in tiff_files:
        with rasterio.open(tiff) as src:
            step_size_x = src.width // target_pixels
            step_size_y = src.height // target_pixels
            data = src.read(1)[::step_size_y, ::step_size_x][:target_pixels, :target_pixels]
            resampled_data.append(data)

    if not resampled_data:
        print(f"No TIFF files found in {input_folder}.")
        return None, None

    with rasterio.open(tiff_files[0]) as src:
        out_meta = src.meta.copy()
        out_meta.update(
            {'driver': 'GTiff', 'height': target_pixels, 'width': target_pixels, 'count': len(resampled_data)})

    with rasterio.open(output_tiff_path, 'w', **out_meta) as dest:
        for i, data in enumerate(resampled_data, start=1):
            dest.write(data, i)

    return find_labels_json(input_folder)


parent_folder = 'test_data/BigEarthNet'
output_tiff_dir = 'test_data/processed_data/20170613T101032_tiff'
output_label_path = 'test_data/processed_data/20170613T101032_labels/all_labels_20170613T101032.json'

os.makedirs(output_tiff_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

all_labels = {}

folder_prefix = "S2A_MSIL2A_20170613T101032"

folders_to_process = [folder for folder in os.listdir(parent_folder) if
                      folder.startswith(folder_prefix) and os.path.isdir(os.path.join(parent_folder, folder))]

for folder in tqdm(folders_to_process, desc="Processing folders"):
    folder_path = os.path.join(parent_folder, folder)
    output_tiff_path = os.path.join(output_tiff_dir, f"{folder}.tif")
    labels = resample_merge_and_save_labels(folder_path, output_tiff_path)
    if labels:
        all_labels[folder] = labels

with open(output_label_path, 'w') as file:
    json.dump(all_labels, file, indent=4)


print('Check Shape of transformed data: ')

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

file_test = 'test_data/processed_data/20170613T101032_tiff/S2A_MSIL2A_20170613T101032_26_57.tif'

with rasterio.open(file_test) as src_test:
    print("Number of Band:", src_test.count)
    for i in range(1, src_test.count + 1):
        print(f"Shape of Band {i} ：", src_test.read(i).shape)


unique_labels = set()
# Read labels from json
try:
    with open(output_label_path, 'r') as file:
        labels_data = json.load(file)
        for labels_list in labels_data.values():
            unique_labels.update(labels_list)
except FileNotFoundError:
    print(f"Error: File not found at {output_label_path}")
except json.JSONDecodeError:
    print("Error: Failed to decode JSON")

labels_list = list(unique_labels)
labels_list.sort()
LABELS = labels_list
print("Labels found:", labels_list)
print("Length of LABELS:", len(labels_list))