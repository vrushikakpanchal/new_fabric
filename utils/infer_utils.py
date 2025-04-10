import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import streamlit as st

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def compute_shape_descriptors(x1, y1, x2, y2, image_w, image_h):
    w, h = x2 - x1, y2 - y1
    area = w * h
    rel_area = area / (image_w * image_h)
    aspect_ratio = w / h if h != 0 else 0
    shape = "Tall & Narrow" if aspect_ratio < 0.5 else "Wide & Short" if aspect_ratio > 2 else "Square-ish"
    return area, rel_area, shape

def estimate_severity(rel_area):
    if rel_area < 0.01:
        return 'Minor', '🟢'
    elif rel_area < 0.03:
        return 'Moderate', '🟡'
    else:
        return 'Severe', '🔴'

def predict_and_display(model, img_np, class_names, fabric_map):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        defect_logits, fabric_logits = model(input_tensor)
        _, pred_defect = torch.max(defect_logits, 1)
        _, pred_fabric = torch.max(fabric_logits, 1)

    pred_defect_name = class_names[pred_defect.item()]
    pred_fabric_name = fabric_map[pred_fabric.item()]

    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = w * 0.3, h * 0.3, w * 0.7, h * 0.7

    area, rel_area, shape = compute_shape_descriptors(x1, y1, x2, y2, w, h)
    severity, emoji = estimate_severity(rel_area)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    label = f"{pred_defect_name} | {severity} {emoji}\n{shape}, Area: {area:.0f}px"
    ax.text(x1, y1 - 10, label, fontsize=10, color='white', backgroundcolor='black')

    plt.title(f"Fabric Type: {pred_fabric_name} | Defect: {pred_defect_name}", fontsize=14)
    plt.axis("off")
    st.pyplot(plt.gcf())

