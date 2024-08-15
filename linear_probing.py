import os
import umap
import torch
import json
import presto
import numpy as np
import warnings
import torch.optim as optim
import torch.nn as nn

from util import get_LABELS
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, fbeta_score


warnings.filterwarnings('ignore')


print("loading data... ...")
# Load the features array from disk
features_file = 'output/temp/features.npy'
features = np.load(features_file)
print(f"Features loaded successfully")

features_test_file = 'output/temp/features_test.npy'
features_test = np.load(features_test_file)
print(f"Features_test loaded successfully")

train_data = torch.load('output/temp/train_data.pt')
print("Train data loaded successfully.")

test_data = torch.load('output/temp/test_data.pt')
print("Test data loaded successfully.")

label_path = 'test_data/processed_data/20170613T101032_labels/all_labels_20170613T101032.json'
LABELS = get_LABELS(label_path)


X_train = torch.tensor(features, dtype=torch.float32)
y_train = torch.tensor(train_data[4].numpy(), dtype=torch.float32)

X_test = torch.tensor(features_test, dtype=torch.float32)
y_test = torch.tensor(test_data[4].numpy(), dtype=torch.float32)


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
num_classes = y_train.shape[1]
model_lp = LinearProbe(input_dim, num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_lp.parameters(), lr=0.002)

n_epochs = 100
for epoch in range(n_epochs):
    model_lp.train()
    optimizer.zero_grad()
    outputs = model_lp(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

model_lp.eval()
with torch.no_grad():
    outputs = model_lp(X_test)
    preds = torch.sigmoid(outputs).numpy() > 0.5

    subset_accuracy = accuracy_score(y_test, preds)
    print("Subset Accuracy:", subset_accuracy)


    f1_micro = f1_score(y_test, preds, average='micro')
    f1_macro = f1_score(y_test, preds, average='macro')
    f1_weighted = f1_score(y_test, preds, average='weighted')

    precision_micro = precision_score(y_test, preds, average='micro')
    precision_macro = precision_score(y_test, preds, average='macro')
    precision_weighted = precision_score(y_test, preds, average='weighted')

    recall_micro = recall_score(y_test, preds, average='micro')
    recall_macro = recall_score(y_test, preds, average='macro')
    recall_weighted = recall_score(y_test, preds, average='weighted')

    f2_micro = fbeta_score(y_test, preds, beta=2, average='micro')
    f2_macro = fbeta_score(y_test, preds, beta=2, average='macro')
    f2_weighted = fbeta_score(y_test, preds, beta=2, average='weighted')

    print("F1 Score (Micro):", f1_micro)
    print("F1 Score (Macro):", f1_macro)
    print("F1 Score (Weighted):", f1_weighted)

    print("Precision (Micro):", precision_micro)
    print("Precision (Macro):", precision_macro)
    print("Precision (Weighted):", precision_weighted)

    print("Recall (Micro):", recall_micro)
    print("Recall (Macro):", recall_macro)
    print("Recall (Weighted):", recall_weighted)

    print("F2 Score (Micro):", f2_micro)
    print("F2 Score (Macro):", f2_macro)
    print("F2 Score (Weighted):", f2_weighted)

    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=LABELS))
