import os
import umap
import torch
import json
import numpy as np
import warnings
from util import get_LABELS
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("loading data... ...")
# Load the features array from disk
features_file = 'output/temp/features.npy'  #
features = np.load(features_file)
print(f"Features loaded successfully.")

train_data = torch.load('output/temp/train_data.pt')
print("Train data loaded successfully.")

labels_UMAP = train_data[4].numpy()


label_path = 'test_data/processed_data/20170613T101032_labels/all_labels_20170613T101032.json'
LABELS = get_LABELS(label_path)

print("Doing UMAP, waiting ... ...")
# 3D Reduction
reducer_3d = umap.UMAP(
    n_neighbors=5,
    spread=1.0,
    min_dist=0.5,
    n_components=3,
    random_state=42
)

embedding_3d = reducer_3d.fit_transform(features)
print("3D Reduction Done!!")

label_names = LABELS
sample_size = 2000

n_plots = len(label_names)
cols = 3
rows = int(np.ceil(n_plots / cols))
fig = plt.figure(figsize=(cols * 8, rows * 8))

print("Processing individual labels...")
for idx, label_name in tqdm(enumerate(label_names), total=len(label_names), desc="Processing Labels"):
    label_data = labels_UMAP[:, idx]
    indices = np.where(label_data == 1)[0]
    non_indices = np.where(label_data == 0)[0]

    chosen_indices = np.random.choice(indices, sample_size, replace=False) if len(indices) >= sample_size else indices
    non_chosen_indices = np.random.choice(non_indices, sample_size, replace=False) if len(non_indices) >= sample_size else non_indices

    ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
    ax.scatter(embedding_3d[non_chosen_indices, 0], embedding_3d[non_chosen_indices, 1], embedding_3d[non_chosen_indices, 2], s=5, color='blue', alpha=0.5, label='Not ' + label_name)
    ax.scatter(embedding_3d[chosen_indices, 0], embedding_3d[chosen_indices, 1], embedding_3d[chosen_indices, 2], s=5, color='red', alpha=0.8, label=label_name)
    ax.set_title(label_name)
    ax.legend()
    ax.axis('on')

plt.tight_layout()
plt.show()