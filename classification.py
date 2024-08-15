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
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, fbeta_score


warnings.filterwarnings('ignore')


print("loading data... ...")
# Load the features array from disk
features_file = 'output/temp/features.npy'  #
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


X_train = torch.tensor(features, dtype=torch.float32).numpy()
y_train = torch.tensor(train_data[4].numpy(), dtype=torch.float32).numpy()

rf_classifier = RandomForestClassifier(
    n_estimators=20,
    criterion='gini',
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features='sqrt',
    max_leaf_nodes=10000,
    bootstrap=False,
    n_jobs=-1
)
print("Training random forest classifier... ...")
rf_classifier.fit(X_train, y_train)
print("Random forest classifier -- Done！！")

X_test = torch.tensor(features_test, dtype=torch.float32).numpy()
y_test = torch.tensor(test_data[4].numpy(), dtype=torch.float32).numpy()

y_pred = rf_classifier.predict(X_test)

f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

f2_micro = fbeta_score(y_test, y_pred, beta=2, average='micro')
f2_macro = fbeta_score(y_test, y_pred, beta=2, average='macro')
f2_weighted = fbeta_score(y_test, y_pred, beta=2, average='weighted')

print("F1 Score (Micro):", f1_micro)
print("F1 Score (Macro):", f1_macro)
print("F1 Score (Weighted):", f1_weighted)
print("F2 Score (Micro):", f2_micro)
print("F2 Score (Macro):", f2_macro)
print("F2 Score (Weighted):", f2_weighted)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS))