# 🧠 MNIST Digit Classifier

An interactive handwritten digit recognition web application built using **PyTorch** and **Flask**. Users can draw a digit (0–9) on a canvas, and the trained neural network predicts the digit in real time.

---

## 📌 Features

- Handwritten digit recognition using a trained PyTorch model
- Interactive HTML5 drawing canvas
- Real-time digit prediction
- Flask backend for inference
- Manual loading of the original MNIST IDX dataset
- Automatic CPU/GPU device detection
- Training, validation, and testing pipeline
- Model checkpoint saving
- Classification report and confusion matrix evaluation

---

## 🖼️ Demo

### Homepage
<img width="627" height="746" alt="image" src="https://github.com/user-attachments/assets/1a9720db-5c76-45fe-a7c7-cd13e61f6780" />

---

## 🛠️ Tech Stack

### Machine Learning
- PyTorch
- NumPy
- Scikit-learn

### Backend
- Flask

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap 5

### Image Processing
- Pillow (PIL)

---

## 📂 Project Structure

```text
MNIST_Digit_Classifier/
│
├── app.py                  # Flask web application
├── train.py                # Model training script
├── requirements.txt
├── README.md
│
├── models/
│   └── mnist_checkpoint.pth
│
├── data_mnist/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

---

## 🧠 Model Architecture

The classifier uses a Multilayer Perceptron (MLP):

```text
Input (784)
      │
      ▼
Linear (784 → 128)
      │
ReLU
      │
▼
Linear (128 → 64)
      │
ReLU
      │
▼
Linear (64 → 10)
```

Loss Function:

- CrossEntropyLoss

Optimizer:

- Adam

Learning Rate:

- 0.001

Batch Size:

- 64

Epochs:

- 20

---

## 📊 Model Performance

| Metric | Result |
|---------|--------|
| Test Accuracy | ~97.8% |
| Validation Accuracy | ~98% |

The model is additionally evaluated using:

- Classification Report
- Confusion Matrix
- Precision
- Recall
- F1-score

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/laserFocusDev/MNIST_Digit_Classifier.git

cd MNIST_Digit_Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

The trained checkpoint will be saved in:

```text
models/mnist_checkpoint.pth
```

---

## 🌐 Run the Web Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

Draw a digit on the canvas and click **Predict** to see the model's prediction.

---

## 📚 Dataset

This project uses the original **MNIST handwritten digit dataset** consisting of:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 grayscale
- 10 digit classes (0–9)

---

## 🔮 Future Improvements

- Convolutional Neural Network (CNN)
- Confidence score visualization
- Improved image preprocessing
- Better UI/UX
- Online deployment
- Mobile support

---

## 👩‍💻 Author

**Sara Sapre**

GitHub: https://github.com/laserFocusDev
