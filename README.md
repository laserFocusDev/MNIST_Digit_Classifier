### MNIST Digit Classifier

Handwritten digit recognition from raw IDX binary files using a custom PyTorch MLP trained to ~97% test accuracy.

Problem

Most MNIST implementations rely on high-level dataset loaders, hiding how raw data is structured and processed. This project focuses on building the full pipeline from scratch — from parsing binary files to training a neural network — to understand every step of the process.

Approach

1. Data Loading

Parsed raw IDX binary files manually using NumPy (no torchvision.datasets)
Extracted metadata (magic number, dimensions) and reconstructed image tensors

2. Preprocessing

Normalized pixel values to [0,1]
Flattened 28×28 images into 784-dimensional vectors
Converted labels into integer class indices

3. Model (Baseline)

Fully connected neural network: 784 → 128 → 64 → 10
ReLU activations
CrossEntropyLoss (applies softmax internally)

4. Training

Optimizer: Adam (lr = 0.001)
Batch size: 64
Epochs: 20
Manual batching and shuffling implemented

5. Evaluation

Computed overall test accuracy on 10,000 samples
Verified convergence using training loss
Results
Metric	Value
Test Accuracy	~97%
Final Training Loss	~0.1
Model Type	MLP (Baseline)

Dataset: MNIST (70,000 images: 60K train / 10K test)

Project Structure
```
.
├── data_mnist/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── main.py
├── requirements.txt
├── README.md
└── LICENSE

```
## How to Run
pip install -r requirements.txt
python main.py

## Limitations
Uses MLP on flattened pixels → ignores spatial structure
No validation split → generalization not fully analyzed
No hyperparameter tuning
No robustness testing (noise, distortions)

## Future Work
Replace MLP with CNN → expected accuracy >99%
Add confusion matrix and misclassification analysis
Introduce validation split and hyperparameter tuning
Build a simple UI for real-time digit prediction

## Stack
Python, PyTorch, NumPy

## Key Contribution
Implemented full MNIST pipeline from raw binary IDX files without high-level dataset libraries
Built training and evaluation loop from scratch to understand end-to-end workflow
