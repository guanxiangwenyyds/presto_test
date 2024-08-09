from util import process_images

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import warnings
import presto
import os

warnings.filterwarnings('ignore')


tiff_directory = 'test_data/processed_data/20170613T101032_tiff'
all_files = [file for file in os.listdir(tiff_directory) if file.endswith('.tif')]

train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

print(f"{len(train_files)} train files and {len(test_files)} test files")

train_data = process_images(train_files)
test_data = process_images(test_files)


output_dir = 'output/temp'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save train set
train_data_path = os.path.join(output_dir, 'train_data.pt')
torch.save(train_data, train_data_path)
print(f"Train data saved successfully to {train_data_path}")

# Save test set
test_data_path = os.path.join(output_dir, 'test_data.pt')
torch.save(test_data, test_data_path)
print(f"Test data saved successfully to {test_data_path}.")

# load pretrained model
pretrained_model = presto.Presto.load_pretrained()

month = torch.tensor([6] * train_data[0].shape[0]).long()
batch_size = 64
dl = DataLoader(
    TensorDataset(
        train_data[0].float(),  # x
        train_data[1].bool(),  # mask
        train_data[2].long(),  # dynamic world
        train_data[3].float(),  # latlons
        month
    ),
    batch_size=batch_size,
    shuffle=False,
)

features_list = []
for (x, mask, dw, latlons, month) in tqdm(dl):
    with torch.no_grad():
        encodings = (
            pretrained_model.encoder(
                x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
            )
            .cpu()
            .numpy()
        )
        features_list.append(encodings)
features_np = np.concatenate(features_list)
print("Feature Extraction -- Done, shape of feature:", features_np.shape)

features_file = 'output/temp/features.npy'
np.save(features_file, features_np)
print(f"Features saved to {features_file}")


# features_test
month = torch.tensor([6] * test_data[0].shape[0]).long()
dl = DataLoader(
    TensorDataset(
        test_data[0].float(),  # x
        test_data[1].bool(),  # mask
        test_data[2].long(),  # dynamic world
        test_data[3].float(),  # latlons
        month
    ),
    batch_size=batch_size,
    shuffle=False,
)

features_list_test = []
for (x, mask, dw, latlons, month) in tqdm(dl):
    with torch.no_grad():
        encodings = (
            pretrained_model.encoder(
                x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
            )
            .cpu()
            .numpy()
        )
        features_list_test.append(encodings)
features_test = np.concatenate(features_list_test)
print("Feature_test Extraction -- Done, shape of feature_test:", features_test.shape)

features_test_file = 'output/temp/features_test.npy'
np.save(features_test_file, features_test)
print(f"Features saved to {features_test_file}")

