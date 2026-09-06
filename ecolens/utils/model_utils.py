"""
Model loading and inference utilities.

Swap `load_model()` and the transform to match your actual trained model.
This file assumes an image classifier with 2+ classes; adjust CLASS_NAMES
to match your training labels exactly (order matters).
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES = ["Healthy Forest", "Deforested Area"]  # extend if you add more classes


@st.cache_resource
def load_model(weights_path: str = None):
    """
    Loads and caches the classifier so it isn't reloaded on every rerun.
    Replace this with your actual architecture + weights.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            st.warning(f"Model weights not found at '{weights_path}' — using untrained weights for now.")

    model.eval()
    return model


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def classify_image(image, model, transform):
    """Returns (pred_class, confidence_pct) or raises on failure."""
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    pred_class = CLASS_NAMES[predicted.item()]
    conf_score = confidence.item() * 100
    return pred_class, conf_score
