from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import torch
import pandas as pd
import numpy as np

from utils.model_utils import CombinedClassifier
from utils.infer_utils import transform, compute_shape_descriptors, estimate_severity

# Initialize app
app = Flask(__name__)

# Config paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load fabric type mappings from CSV
fabric_df = pd.read_csv('fabric_dataset/fabric_types.csv')
fabric_classes = sorted(fabric_df['Fabric_Type'].unique())  # e.g., ['cotton', 'linen', ...]
fabric_map = {i: name for i, name in enumerate(fabric_classes)}

# Define defect classes
defect_classes = ['hole', 'knot', 'stain']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedClassifier(num_defect_classes=len(defect_classes), num_fabric_classes=len(fabric_map))
model.load_state_dict(torch.load("models/combined_fabric_defect_model.pt", map_location=device))
model.to(device)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load and process image
    image = Image.open(file_path).convert("RGB")
    img_np = np.array(image)
    input_tensor = transform(img_np).unsqueeze(0).to(device)

    with torch.no_grad():
        defect_logits, fabric_logits = model(input_tensor)
        _, pred_defect = torch.max(defect_logits, 1)
        _, pred_fabric = torch.max(fabric_logits, 1)

    pred_defect_name = defect_classes[pred_defect.item()]
    pred_fabric_name = fabric_map[pred_fabric.item()]

    # Simulate defect area and bounding box
    w, h = image.size
    x1, y1, x2, y2 = int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)
    area, rel_area, shape = compute_shape_descriptors(x1, y1, x2, y2, w, h)
    severity, emoji = estimate_severity(rel_area)

    # Draw rectangle on image
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    draw.text((x1, y1 - 20), f"{pred_defect_name} | {severity} {emoji}", fill="white")

    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    image.save(result_path)

    return jsonify({
        "fabric_type": pred_fabric_name,
        "defect_type": pred_defect_name,
        "defect_count": 1 if pred_defect_name != "None" else 0,
        "shape": shape,
        "area": int(area),
        "severity": severity,
        "emoji": emoji,
        "original_image": f"/static/uploads/{filename}",
        "analyzed_image": f"/static/results/{result_filename}"
    })

if __name__ == '__main__':
    app.run(debug=True)
