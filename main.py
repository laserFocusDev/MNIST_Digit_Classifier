import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct

def load_images(file):
    with open(file,'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return images
    
def load_labels(file):
    with open(file,'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels
    
train_images = load_images(r"C:\Users\Sara Sapre\OneDrive\Desktop\MNIST Digit Classifier\data_mnist\train-images.idx3-ubyte")
train_labels = load_labels(r"C:\Users\Sara Sapre\OneDrive\Desktop\MNIST Digit Classifier\data_mnist\train-labels.idx1-ubyte")

test_images = load_images(r"C:\Users\Sara Sapre\OneDrive\Desktop\MNIST Digit Classifier\data_mnist\t10k-images.idx3-ubyte")
test_labels = load_labels(r"C:\Users\Sara Sapre\OneDrive\Desktop\MNIST Digit Classifier\data_mnist\t10k-labels.idx1-ubyte")

train_images = train_images / 255.0
test_images = test_images / 255.0

X_train = torch.tensor(train_images, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

X_test = torch.tensor(test_images, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

X_train = X_train.view(-1, 28*28)
X_test = X_test.view(-1, 28*28)

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self,x):
        return self.model(x)

model = MNISTClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 64

for epoch in range(20):

    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    for i in range(0, len(X_train), batch_size):

        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch+1, "Loss:", loss.item())

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
print("Test Accuracy:", accuracy.item()*100, "%")
torch.save(model.state_dict(), "mnist_model.pth")
