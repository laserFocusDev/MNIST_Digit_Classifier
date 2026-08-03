import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# =====================================================
# Device Configuration
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# Load MNIST IDX Files
# =====================================================

def load_images(file):
    with open(file, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return images


def load_labels(file):
    with open(file, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# =====================================================
# Dataset Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_mnist")

train_images = load_images(
    os.path.join(DATA_DIR, "train-images.idx3-ubyte")
)

train_labels = load_labels(
    os.path.join(DATA_DIR, "train-labels.idx1-ubyte")
)

test_images = load_images(
    os.path.join(DATA_DIR, "t10k-images.idx3-ubyte")
)

test_labels = load_labels(
    os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte")
)

# =====================================================
# Preprocessing
# =====================================================

train_images = train_images / 255.0
test_images = test_images / 255.0

X_train = torch.tensor(train_images, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

X_test = torch.tensor(test_images, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# =====================================================
# Train / Validation Split
# =====================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# Flatten images for MLP

X_train = X_train.view(-1, 28 * 28)
X_val = X_val.view(-1, 28 * 28)
X_test = X_test.view(-1, 28 * 28)

print("Training Set :", X_train.shape)
print("Validation Set:", X_val.shape)
print("Test Set     :", X_test.shape)

# Move validation and test data to device once
X_val = X_val.to(device)
y_val = y_val.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

# =====================================================
# Neural Network
# =====================================================

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# =====================================================
# Model
# =====================================================

model = MNISTClassifier().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

batch_size = 64
epochs = 20

# =====================================================
# Training
# =====================================================

for epoch in range(epochs):

    model.train()

    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    running_loss = 0.0

    for i in range(0, len(X_train), batch_size):

        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)

        loss = loss_fn(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / ((len(X_train) + batch_size - 1) // batch_size)

    # ---------------- Validation ----------------

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val)

        _, val_predictions = torch.max(val_outputs, 1)

        val_accuracy = (
            (val_predictions == y_val)
            .float()
            .mean()
            .item()
        )

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy*100:.2f}%"
    )

# =====================================================
# Testing
# =====================================================
model.eval()

with torch.no_grad():

    outputs = model(X_test)

    _, predicted = torch.max(outputs, 1)

# Move predictions back to CPU for sklearn

y_true = y_test.cpu().numpy()
y_pred = predicted.cpu().numpy()

accuracy = accuracy_score(y_true, y_pred)

print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report")
print(classification_report(y_true, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

# =====================================================
# Save Checkpoint
# =====================================================

os.makedirs("models", exist_ok=True)

torch.save(
    {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    },
    os.path.join("models", "mnist_checkpoint.pth"),
)

print("Model checkpoint saved successfully!")