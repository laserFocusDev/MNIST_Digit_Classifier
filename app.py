import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn

from PIL import Image
import io
import base64
import numpy as np

# ---------------------------------------------------
# Flask App
# ---------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------
# Device
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# Model Architecture
# ---------------------------------------------------

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

# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------

model = MNISTClassifier().to(device)

checkpoint = torch.load(
    os.path.join("models", "mnist_checkpoint.pth"),
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully!")

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    image_data = data["image"]

    # Remove header:
    # data:image/png;base64,...
    image_data = image_data.split(",")[1]

    image = Image.open(
        io.BytesIO(base64.b64decode(image_data))
    )

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy
    image = np.array(image)

    # MNIST digits are white on black
    image = 255 - image

    # Normalize
    image = image / 255.0

    # Flatten
    image = image.reshape(1, 784)

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        outputs = model(tensor)

        prediction = torch.argmax(outputs, dim=1).item()

    return jsonify({
        "prediction": prediction
    })

# ---------------------------------------------------
# Run App
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)