from flask import Flask, request, jsonify
import torchxrayvision as xrv
from torchxrayvision.models import DenseNet
import skimage.io
import numpy as np
import torch
import torchvision

app = Flask(__name__)
model = DenseNet(weights="densenet121-res224-all").eval()
if torch.cuda.is_available():
    model.cuda()

def preprocess_xray(arr):
    img = xrv.datasets.normalize(arr, 255)
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(2)[None, ...]
    if img.ndim == 2:
        img = img[None, ...]
    tr = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = tr(img)
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "please upload an image with field name file"}), 400

    f = request.files["file"]
    try:
        arr = skimage.io.imread(f, plugin="pil")
    except Exception:
        return jsonify({"error": "unable to read image"}), 400

    tensor = preprocess_xray(arr)

    with torch.no_grad():
        out = model(tensor).cpu().numpy()[0]
    names = xrv.datasets.default_pathologies

    results = [
        {"pathology": p, "probability": float(f"{prob:.4f}")}
        for p, prob in zip(names, out) if p.strip()
    ]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)