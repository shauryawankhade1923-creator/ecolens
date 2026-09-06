import os
import io
import json
import time
import math
import base64
import hashlib
import requests
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.nn.functional as F

from gtts import gTTS
import speech_recognition as sr

import folium
from streamlit_folium import st_folium

from PIL import Image
from torchvision import models, transforms

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib import colors

from species_db import SPECIES_DATABASE, SPECIES_NAME_MAP


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EcoLens 2.0",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model_weights.pth"

FOREST_CLASSES = [
    "Healthy Forest",
    "Deforested Area",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #777;
        padding: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_forest_result" not in st.session_state:
    st.session_state.last_forest_result = None

if "last_forest_image" not in st.session_state:
    st.session_state.last_forest_image = None

if "last_veg_result" not in st.session_state:
    st.session_state.last_veg_result = None

if "last_forest_gemini_diag" not in st.session_state:
    st.session_state.last_forest_gemini_diag = None

if "last_temporal_result" not in st.session_state:
    st.session_state.last_temporal_result = None

if "map_lat" not in st.session_state:
    st.session_state.map_lat = -3.4653

if "map_lon" not in st.session_state:
    st.session_state.map_lon = -62.2159

if "map_place_name" not in st.session_state:
    st.session_state.map_place_name = "Amazon Rainforest, Brazil"

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 15

if "last_map_image" not in st.session_state:
    st.session_state.last_map_image = None

if "last_map_result" not in st.session_state:
    st.session_state.last_map_result = None

if "last_species" not in st.session_state:
    st.session_state.last_species = None

if "last_species_type" not in st.session_state:
    st.session_state.last_species_type = None

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

if "voice_chat_history" not in st.session_state:
    st.session_state.voice_chat_history = []

if "voice_accent" not in st.session_state:
    st.session_state.voice_accent = "co.in"

if "voice_language" not in st.session_state:
    st.session_state.voice_language = "en-IN"

if "auto_speech" not in st.session_state:
    st.session_state.auto_speech = True

if "species_database" not in st.session_state:
    st.session_state.species_database = SPECIES_DATABASE.copy()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# LOAD YOUR EXACT FOREST MODEL
# ============================================================

@st.cache_resource
def load_forest_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} was not found."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise RuntimeError(
            "The uploaded file does not contain a valid state_dict."
        )

    # Remove DataParallel prefix if present
    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    state_dict = cleaned

    # --------------------------------------------------------
    # VERIFIED ARCHITECTURE FROM YOUR UPLOADED WEIGHTS
    #
    # ResNet18
    #
    # fc.0.weight = [128, 512]
    # fc.3.weight = [2, 128]
    # --------------------------------------------------------

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


try:
    forest_model = load_forest_model()
    MODEL_LOADED = True
    MODEL_ERROR = None

except Exception as e:
    forest_model = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def image_hash(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
    )

    return hashlib.sha256(
        buffer.getvalue()
    ).hexdigest()[:12]


def now():

    return datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )


# ============================================================
# SATELLITE TILE FETCHER & REMOTE SENSING ENGINE
# ============================================================

def get_satellite_patch(lat, lon, zoom=15):
    """
    Fetches high-resolution satellite imagery tiles for specified coordinates (WGS84)
    and stitches them into a 512x512 composite patch.
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    tiles = []
    headers = {"User-Agent": "EcoLens/2.0 (Environmental AI Platform)"}

    for dy in range(-1, 1):
        row = []
        for dx in range(-1, 1):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile+dy}/{xtile+dx}"
            try:
                r = requests.get(url, headers=headers, timeout=6)
                if r.status_code == 200:
                    row.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
                else:
                    row.append(Image.new("RGB", (256, 256), color=(30, 45, 30)))
            except Exception:
                row.append(Image.new("RGB", (256, 256), color=(30, 45, 30)))
        tiles.append(row)

    patch = Image.new("RGB", (512, 512))
    patch.paste(tiles[0][0], (0, 0))
    patch.paste(tiles[0][1], (256, 0))
    patch.paste(tiles[1][0], (0, 256))
    patch.paste(tiles[1][1], (256, 256))
    return patch


def get_place_name_from_coords(lat, lon):
    """
    Reverse geocodes coordinates to a human-readable place name (country, region, park).
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    headers = {"User-Agent": "EcoLens/2.0 (Environmental AI Platform)"}
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=4)
        if r.status_code == 200:
            data = r.json()
            address = data.get("address", {})
            parts = [address.get(k) for k in ["natural", "national_park", "forest", "county", "state", "country"] if address.get(k)]
            if parts:
                return ", ".join(parts)
            return data.get("display_name", f"{lat:.4f}°, {lon:.4f}°")
    except Exception:
        pass
    return f"{lat:.4f}°, {lon:.4f}°"


def search_place_coordinates(query):
    """
    Geocodes a search string (place or forest name) to latitude and longitude coordinates.
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "EcoLens/2.0 (Environmental AI Platform)"}
    params = {"q": query, "format": "json", "limit": 1}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=5)
        if r.status_code == 200 and r.json():
            res = r.json()[0]
            return float(res["lat"]), float(res["lon"]), res.get("display_name", query)
    except Exception:
        pass
    return None


def compute_vegetation_indices(image):
    """
    Computes remote sensing vegetation indices (GLI, VARI) from RGB imagery.
    Extracts canopy cover %, bare ground %, and generates multi-layer visual maps.
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Visible Atmospherically Resistant Index (VARI)
    denom_vari = g + r - b
    vari = np.where(np.abs(denom_vari) > 1e-4, (g - r) / (denom_vari + 1e-6), 0.0)

    # Green Leaf Index (GLI)
    denom_gli = 2.0 * g + r + b
    gli = np.where(denom_gli > 1e-4, (2.0 * g - r - b) / (denom_gli + 1e-6), 0.0)

    # Excess Green Index (ExG)
    exg = 2.0 * g - r - b

    # Biophysical Canopy Segmentation:
    # Requires true chlorophyll absorption (G > R * 1.02 and G > B * 1.05 with positive GLI/ExG)
    canopy_mask = (gli > 0.05) & (exg > 0.04) & (g > r * 1.02) & (g > b * 1.05)

    total_pixels = canopy_mask.size
    canopy_pixels = int(np.sum(canopy_mask))
    canopy_cover_percent = (canopy_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    bare_ground_percent = max(0.0, 100.0 - canopy_cover_percent)
    mean_gli = float(np.mean(gli))

    # Estimated carbon stock metrics (Standard benchmark: ~145 t CO2 / ha in dense forest)
    est_carbon_loss_per_ha = round((100.0 - canopy_cover_percent) * 1.45, 1)

    # 1. Canopy Segmentation Mask (Emerald Green = Canopy, Crimson = Bare / Non-canopy)
    orig_np = np.array(img_rgb)
    mask_overlay = orig_np.copy()
    mask_overlay[canopy_mask] = (mask_overlay[canopy_mask] * 0.35 + np.array([34, 197, 94]) * 0.65).astype(np.uint8)
    mask_overlay[~canopy_mask] = (mask_overlay[~canopy_mask] * 0.35 + np.array([239, 68, 68]) * 0.65).astype(np.uint8)
    canopy_mask_pil = Image.fromarray(mask_overlay)

    # 2. Vegetation Health Gradient Map
    # Map GLI from [-0.15, 0.35] to [0, 1]
    gli_norm = np.clip((gli + 0.15) / 0.50, 0.0, 1.0)
    health_r = np.where(gli_norm < 0.5, 240, (240 - (gli_norm - 0.5) * 2 * 200)).astype(np.uint8)
    health_g = np.where(gli_norm < 0.5, (gli_norm * 2 * 220), 220).astype(np.uint8)
    health_b = np.full_like(health_r, 40, dtype=np.uint8)

    health_rgb = np.stack([health_r, health_g, health_b], axis=-1)
    blended_health = (orig_np * 0.30 + health_rgb * 0.70).astype(np.uint8)
    veg_health_pil = Image.fromarray(blended_health)

    return {
        "canopy_cover_percent": round(canopy_cover_percent, 1),
        "bare_ground_percent": round(bare_ground_percent, 1),
        "mean_gli": round(mean_gli, 3),
        "est_carbon_loss_per_ha": est_carbon_loss_per_ha,
        "canopy_mask": canopy_mask,
        "canopy_mask_img": canopy_mask_pil,
        "veg_health_img": veg_health_pil,
    }


def compute_temporal_change(img_before, img_after):
    """
    Computes Before vs After temporal change detection between historical & recent imagery.
    Quantifies canopy loss delta, deforestation rate, and generates difference heatmaps.
    """
    img_before_rgb = img_before.convert("RGB")
    img_after_rgb = img_after.convert("RGB").resize(img_before_rgb.size, Image.Resampling.BILINEAR)

    veg_before = compute_vegetation_indices(img_before_rgb)
    veg_after = compute_vegetation_indices(img_after_rgb)

    mask_before = veg_before["canopy_mask"]
    mask_after = veg_after["canopy_mask"]

    # Lost canopy: was canopy before, but NOT canopy now
    lost_mask = mask_before & (~mask_after)
    # Regrowth canopy: was not canopy before, but IS canopy now
    gained_mask = (~mask_before) & mask_after

    total_pixels = mask_before.size
    loss_percent = (np.sum(lost_mask) / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    gain_percent = (np.sum(gained_mask) / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    net_change = round(gain_percent - loss_percent, 1)

    # Create Difference Overlay on Recent Image
    after_np = np.array(img_after_rgb)
    change_overlay = after_np.copy()

    # Highlight Deforested Areas in Bright Fluorescent Red
    change_overlay[lost_mask] = (change_overlay[lost_mask] * 0.2 + np.array([255, 40, 40]) * 0.8).astype(np.uint8)
    # Highlight Afforested/Regrown Areas in Vibrant Cyan/Green
    change_overlay[gained_mask] = (change_overlay[gained_mask] * 0.2 + np.array([0, 255, 160]) * 0.8).astype(np.uint8)

    change_pil = Image.fromarray(change_overlay)

    return {
        "canopy_before": veg_before["canopy_cover_percent"],
        "canopy_after": veg_after["canopy_cover_percent"],
        "loss_percent": round(loss_percent, 1),
        "gain_percent": round(gain_percent, 1),
        "net_change": net_change,
        "change_overlay_img": change_pil,
        "veg_before": veg_before,
        "veg_after": veg_after,
    }


# ============================================================
# FOREST AI (WITH HYBRID OPTICAL RECTIFICATION)
# ============================================================

def forest_prediction(image, veg_result=None):
    if forest_model is None:
        raise RuntimeError(
            MODEL_ERROR
        )

    if veg_result is None:
        veg_result = compute_vegetation_indices(image)

    tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    start = time.perf_counter()

    with torch.no_grad():
        output = forest_model(
            tensor
        )
        probabilities = F.softmax(
            output,
            dim=1,
        )
        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    elapsed = (
        time.perf_counter() - start
    )

    raw_predicted_index = prediction.item()
    raw_predicted_class = FOREST_CLASSES[raw_predicted_index]
    raw_confidence = confidence.item() * 100.0

    canopy_cover = veg_result["canopy_cover_percent"]
    bare_ground = veg_result["bare_ground_percent"]
    mean_gli = veg_result["mean_gli"]

    # -------------------------------------------------------------
    # OPTICAL CANOPY RECTIFICATION (Eliminating False Positives on Dry Land)
    # -------------------------------------------------------------
    # Dry land / desert / arid soil / cleared ground has:
    # 1. Low green canopy cover (< 35%) OR
    # 2. Dominant bare soil (> 65%) with low/negative GLI (< 0.04)
    # A true "Healthy Forest" requires significant green vegetative canopy.

    if raw_predicted_class == "Healthy Forest" and (canopy_cover < 35.0 or (mean_gli < 0.035 and canopy_cover < 50.0)):
        # Deep learning ResNet false positive on dry/bare land -> Corrected to Deforested Area
        predicted_class = "Deforested Area"
        # Calibrated confidence reflects the verified lack of canopy
        calibrated_conf = round(min(98.5, max(75.0, bare_ground * 0.95)), 1)
        confidence_value = calibrated_conf
        probabilities_dict = {
            "Healthy Forest": round(100.0 - confidence_value, 1),
            "Deforested Area": confidence_value,
        }
        risk = "Critical" if confidence_value >= 85 else "High"
        severity = confidence_value
        recommendation = (
            f"⚠️ Arid / Dry / Cleared Land Detected: Optical remote sensing verified only {canopy_cover:.1f}% vegetative canopy "
            f"(mean GLI: {mean_gli:.3f}, bare ground: {bare_ground:.1f}%). The neural network false positive was corrected to Deforested / Degraded Area."
        )

    elif raw_predicted_class == "Deforested Area" and canopy_cover > 75.0 and mean_gli > 0.12:
        # Dense green forest mistakenly flagged -> Corrected to Healthy Forest
        predicted_class = "Healthy Forest"
        calibrated_conf = round(min(98.5, max(75.0, canopy_cover)), 1)
        confidence_value = calibrated_conf
        probabilities_dict = {
            "Healthy Forest": confidence_value,
            "Deforested Area": round(100.0 - confidence_value, 1),
        }
        risk = "Low"
        severity = max(0.0, 100.0 - confidence_value)
        recommendation = (
            f"🌿 Dense Forest Confirmed: Optical remote sensing verified {canopy_cover:.1f}% green canopy cover "
            f"(mean GLI: {mean_gli:.3f}). Classified as Healthy Forest."
        )

    else:
        # Standard classification alignment
        predicted_class = raw_predicted_class
        confidence_value = raw_confidence
        probabilities_dict = {
            FOREST_CLASSES[i]: probabilities[0, i].item() * 100.0
            for i in range(len(FOREST_CLASSES))
        }

        if predicted_class == "Healthy Forest":
            risk = "Low"
            recommendation = (
                f"The model and optical indices indicate healthy forest cover ({canopy_cover:.1f}% canopy coverage). "
                "Continue periodic monitoring."
            )
            severity = max(0.0, 100.0 - confidence_value)
        else:
            if confidence_value >= 90:
                risk = "Critical"
            elif confidence_value >= 75:
                risk = "High"
            else:
                risk = "Moderate"

            recommendation = (
                f"Potential forest degradation or deforestation detected ({bare_ground:.1f}% bare/cleared ground). "
                "Verify using historical satellite imagery, GIS analysis, and field observations."
            )
            severity = confidence_value

    return {
        "prediction": predicted_class,
        "confidence": confidence_value,
        "probabilities": probabilities_dict,
        "risk": risk,
        "severity": severity,
        "recommendation": recommendation,
        "inference_time": elapsed,
        "timestamp": now(),
        "image_hash": image_hash(image),
        "veg_result": veg_result,
    }


# ============================================================
# GRAD-CAM
# ============================================================

def gradcam(image, target_class):

    if forest_model is None:
        return None

    activations = []
    gradients = []

    target_layer = forest_model.layer4[-1]

    def forward_hook(
        module,
        inputs,
        output,
    ):
        activations.append(output)

    def backward_hook(
        module,
        grad_input,
        grad_output,
    ):
        gradients.append(
            grad_output[0]
        )

    fh = target_layer.register_forward_hook(
        forward_hook
    )

    bh = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        tensor = transform(
            image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )

        forest_model.zero_grad()

        with torch.enable_grad():

            output = forest_model(
                tensor
            )

            score = output[
                0,
                target_class
            ]

            score.backward()

        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activation
        ).sum(
            dim=1
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        heat = (
            cam * 255
        ).astype(
            np.uint8
        )

        original = np.array(
            image.convert("RGB")
        )

        # Red intensity heatmap
        overlay = original.copy()

        overlay[:, :, 0] = np.maximum(
            overlay[:, :, 0],
            heat,
        )

        result = Image.fromarray(
            overlay
        )

        return result

    except Exception:
        return None

    finally:

        fh.remove()
        bh.remove()


# ============================================================
# BIODIVERSITY INFORMATION DATABASE
# ============================================================
# SPECIES_DATABASE and SPECIES_NAME_MAP are now imported from species_db.py


# ============================================================
# SPECIES INFORMATION DISPLAY
# ============================================================

def show_species_information(species_name):

    data = st.session_state.species_database.get(species_name)

    if data is None:
        # Fallback to query iNaturalist API for dynamic species information if not in local DB
        try:
            res = requests.get(
                f"https://api.inaturalist.org/v1/taxa?q={species_name}&per_page=1",
                timeout=3,
            )
            if res.status_code == 200 and res.json().get("results"):
                tax = res.json()["results"][0]
                data = {
                    "scientific_name": tax.get("name", species_name),
                    "type": tax.get("iconic_taxon_name", "Organism"),
                    "family": tax.get("rank", "Taxon"),
                    "habitat": "Terrestrial / Natural ecosystems",
                    "ecological_role": "Contributes to ecosystem biodiversity and food web structure.",
                    "importance": "Key biological species documented in global biodiversity index.",
                    "conservation": "Recorded in iNaturalist Biodiversity Network",
                    "diet": "Natural Ecosystem Forager",
                    "threats": "Habitat alteration and environmental change.",
                    "facts": f"{species_name} ({tax.get('name', '')}) is documented in international biodiversity archives.",
                }
        except Exception:
            pass

    if data is None:
        st.info(
            """
            Species information is not available in the offline EcoLens database yet.
            Add this species to the database or connect EcoLens to a biodiversity API.
            """
        )
        return

    st.subheader(f"📚 {species_name}")
    
    sci_name = data.get("scientific_name") or data.get("scientificName") or species_name
    spec_type = data.get("type") or "Organism"
    family = data.get("family") or "N/A"
    habitat = data.get("habitat") or "N/A"
    region = data.get("region") or "N/A"
    diet = data.get("diet") or "N/A"
    conservation = data.get("conservation") or "N/A"
    ecological_role = data.get("ecological_role") or data.get("ecologicalRole") or "N/A"
    importance = data.get("importance") or "N/A"
    threats = data.get("threats") or "N/A"
    facts = data.get("facts") or data.get("fact") or data.get("interesting_fact") or f"{species_name} is documented in international biodiversity archives."

    st.write(f"**Scientific Name:** *{sci_name}*")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write(f"**Type:** {spec_type}")
        st.write(f"**Family:** {family}")
        st.write(f"**🌍 Found In:** {region}")
        st.write(f"**Habitat:** {habitat}")
        st.write(f"**Diet:** {diet}")

    with c2:
        st.write(f"**Conservation:** {conservation}")
        
        is_conserved = data.get("conserved", False)
        conserved_str = "Protected / Conserved ✅" if is_conserved else "No Active Protection ❌"
        st.write(f"**Conserved Status:** {conserved_str}")
        
        est_pop = data.get("estimated_population") or data.get("estimatedPopulation") or "N/A"
        if est_pop != "N/A" and not any(w in str(est_pop).lower() for w in ["approx", "estimated", "~", "roughly"]):
            est_pop_display = f"Estimated ~{est_pop}"
        else:
            est_pop_display = est_pop
        st.write(f"**Estimated Population:** {est_pop_display}")

    with c3:
        st.write(f"**Ecological Role:** {ecological_role}")
        st.write(f"**Importance:** {importance}")
        st.write(f"**Threats:** {threats}")

    st.info(f"💡 **Interesting Fact:** {facts}")


# ============================================================
# API INTEGRATION ENGINE (GEMINI & PLANTNET)
# ============================================================

import re as _re

def _parse_gemini_json(text_content):
    """
    Robustly parse JSON from Gemini API responses.
    Handles markdown fences, trailing commas, truncated output, and LLM quirks.
    """
    text_content = text_content.strip()
    
    # Strip markdown code fences
    if text_content.startswith("```"):
        lines = text_content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text_content = "\n".join(lines).strip()
    
    # Remove control characters except newlines and tabs
    text_content = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text_content)
    
    # Try direct parse first
    try:
        return json.loads(text_content)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON object from surrounding text
    match = _re.search(r'\{', text_content)
    if match:
        json_str = text_content[match.start():]
        
        # Fix trailing commas before } or ]
        json_str = _re.sub(r',\s*([}\]])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try to repair truncated JSON by closing open strings and brackets
        repaired = json_str
        # If the JSON is truncated mid-value, try to close it
        # Count open braces/brackets
        in_string = False
        escape_next = False
        open_braces = 0
        open_brackets = 0
        last_good = 0
        for i, ch in enumerate(repaired):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                open_braces += 1
            elif ch == '}':
                open_braces -= 1
            elif ch == '[':
                open_brackets += 1
            elif ch == ']':
                open_brackets -= 1
            if open_braces >= 0 and open_brackets >= 0:
                last_good = i
        
        # If we ended inside a string, close it
        if in_string:
            repaired += '"'
        # Remove any trailing comma
        repaired = repaired.rstrip().rstrip(',')
        # Close remaining brackets and braces
        repaired += ']' * open_brackets + '}' * open_braces
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        
        # Final attempt: truncate at the last complete key-value pair
        # Find the last complete "key": "value" or "key": bool/number
        last_comma = json_str.rfind(',')
        if last_comma > 0:
            truncated = json_str[:last_comma].rstrip()
            # Close remaining structure
            truncated += '}'
            truncated = _re.sub(r',\s*}', '}', truncated)
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass
    
    raise RuntimeError(f"Could not parse Gemini response as JSON. Raw: {text_content[:300]}")

# Supported Gemini API models in fallback priority order
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
]

def _call_gemini_api(payload, api_key, timeout=25):
    """
    Calls Google Gemini REST API with automatic multi-model fallback cascade.
    If a model hits rate limit / quota (HTTP 429) or timeout, it automatically
    tries the next available model in the priority list.
    """
    headers = {"Content-Type": "application/json"}
    last_err = None

    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                last_err = f"Gemini quota / rate limit reached on {model} (HTTP 429)"
                time.sleep(0.6)
                continue
            else:
                last_err = f"Gemini API returned status {res.status_code} on {model}"
                continue
        except requests.exceptions.Timeout:
            last_err = f"Timeout on {model}"
            continue
        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(last_err or "Gemini API unavailable across all fallback models.")


def query_gemini_vision(image, species_type, api_key):
    """
    Call Gemini REST API with the image to identify the species and return detailed JSON.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    prompt = (
        f"Identify the {species_type} in this image. Return a JSON object with these keys. "
        "Keep each value SHORT (under 15 words). Do NOT use special characters or quotes inside values.\n"
        '{"species_name": "...", "scientific_name": "...", '
        '"type": "Plant / Tree or Plant / Aquatic or Animal / Mammal or Animal / Bird or Animal / Amphibian or Animal / Aquatic", '
        '"family": "...", "habitat": "...", "region": "country or continent", "diet": "...", '
        '"conservation": "IUCN status", "conserved": true/false, '
        '"estimated_population": "number or N/A", '
        '"ecological_role": "...", "importance": "...", '
        '"threats": "...", "facts": "one short fact"}'
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 1024,
            "temperature": 0.2
        }
    }
    
    res_json = _call_gemini_api(payload, api_key, timeout=30)
    candidates = res_json.get('candidates', [])
    if candidates:
        text_content = candidates[0]['content']['parts'][0]['text']
        return _parse_gemini_json(text_content)
    raise RuntimeError("Received empty response candidate from Gemini API.")


def query_gemini_details_by_name(species_name, api_key):
    """
    Query Gemini REST API to get the detailed profile for a species by name.
    """
    prompt = (
        f"Provide a species profile for '{species_name}' as a JSON object. "
        "Keep each value SHORT (under 15 words). No special characters or quotes inside values.\n"
        '{"species_name": "...", "scientific_name": "...", '
        '"type": "Plant / Tree or Plant / Aquatic or Animal / Mammal or Animal / Bird or Animal / Amphibian or Animal / Aquatic", '
        '"family": "...", "habitat": "...", "region": "country or continent", "diet": "...", '
        '"conservation": "IUCN status", "conserved": true/false, '
        '"estimated_population": "number or N/A", '
        '"ecological_role": "...", "importance": "...", '
        '"threats": "...", "facts": "one short fact"}'
    )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 1024,
            "temperature": 0.2
        }
    }
    
    res_json = _call_gemini_api(payload, api_key, timeout=20)
    candidates = res_json.get('candidates', [])
    if candidates:
        text_content = candidates[0]['content']['parts'][0]['text']
        return _parse_gemini_json(text_content)
    raise RuntimeError("Received empty response candidate from Gemini API.")


def query_plantnet_api(image, api_key):
    """
    Query Pl@ntNet API to identify the plant.
    Returns the top match's details or None.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={api_key}"
    
    files = [
        ('images', ('image.jpg', img_bytes, 'image/jpeg'))
    ]
    data = {
        'organs': ['flower']
    }
    
    try:
        res = requests.post(url, files=files, data=data, timeout=10)
        if res.status_code == 200:
            res_json = res.json()
            results = res_json.get("results", [])
            if results:
                top_match = results[0]
                species_info = top_match.get("species", {})
                scientific_name = species_info.get("scientificNameWithoutAuthor", "")
                common_names = species_info.get("commonNames", [])
                common_name = common_names[0] if common_names else scientific_name
                return {
                    "scientific_name": scientific_name,
                    "species_name": common_name,
                    "score": top_match.get("score", 0.0)
                }
        return None
    except Exception as e:
        st.warning(f"Pl@ntNet API warning: {e}")
        return None


def clean_for_speech(text):
    """
    Cleans markdown formatting, links, and emojis for natural Text-To-Speech pronunciation.
    Normalizes numbers (e.g. ~55,000 -> approximately 55000) so they are read as whole numbers.
    Ensures clear, distinct pauses between bullet points, list items, and before/after brackets.
    """
    import re
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Normalize approximations: ~55,000 -> approximately 55,000
    text = re.sub(r'[~≈]\s*', 'approximately ', text)
    
    # Normalize number ranges: 10,000 - 25,000 -> 10,000 to 25,000
    text = re.sub(r'(\d+)\s*[-–—]\s*(\d+)', r'\1 to \2', text)
    
    # Remove commas between digits so "55,000" becomes "55000" (spoken as "fifty-five thousand", not "fifty five zero zero zero")
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    
    # Add natural audible pauses before and after parentheses and brackets
    text = re.sub(r'\s*[\(\[\{]\s*', ', ', text)
    text = re.sub(r'\s*[\)\]\}]\s*', ', ', text)
    
    # Replace slashes in taxonomies/classifications with pauses (e.g. "Animal / Mammal" -> "Animal, Mammal")
    text = re.sub(r'\s*/\s*', ', ', text)
    
    # Remove remaining markdown formatting characters
    text = re.sub(r'[*#_`>]', '', text)
    
    # Process line by line to preserve list structure and insert audible pauses
    lines = text.split('\n')
    cleaned_items = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        # Remove bullet marks and leading numbers
        line = re.sub(r'^[•\-\*\d\.\)]+\s*', '', line).strip()
        if not line:
            continue
        # Strip emojis and non-speech symbols while keeping letters, numbers, spaces, and punctuation
        line = re.sub(r'[^\w\s.,!?:;\'-]', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        # Clean up repeated or stray commas (excluding commas already removed from numbers)
        line = re.sub(r'\s*,\s*', ', ', line)
        line = re.sub(r'^,\s*', '', line)
        line = re.sub(r',\s*$', '', line)
        if not line:
            continue
        # Add punctuation pause at end of point if not already present
        if line[-1] not in '.!?:;':
            line += '.'
        cleaned_items.append(line)
        
    # Join distinct points with sentence pauses
    result = ' ... '.join(cleaned_items)
    result = re.sub(r',\s*\.', '.', result)
    return result.strip()


def generate_voice_audio(text, lang='en', tld='com'):
    """
    Generates high-fidelity spoken audio bytes (MP3) from text using gTTS.
    Speaks the full response without arbitrary cut-offs or incomplete sentences.
    """
    clean = clean_for_speech(text)
    if not clean:
        clean = "I am EcoLens AI Assistant."
    
    # If text is exceptionally long (>6000 chars), truncate cleanly at the last completed sentence
    if len(clean) > 6000:
        last_period = max(clean[:6000].rfind('.'), clean[:6000].rfind('!'), clean[:6000].rfind('?'))
        if last_period > 1000:
            clean = clean[:last_period+1]
        else:
            clean = clean[:6000]

    try:
        tts = gTTS(text=clean, lang=lang, tld=tld, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.getvalue()
    except Exception:
        return None


def transcribe_audio_bytes(audio_bytes):
    """
    Transcribes audio bytes to text using SpeechRecognition or returns an informative error.
    """
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except Exception:
        return None


def render_speech_controls(text, uid, accent_tld='co.in', auto_play=False, voice_lang_override=None):
    """
    Renders an interactive voice speech player toolbar with Play, Pause, Continue, and Stop buttons.
    Uses a clean, single-engine browser Web Speech API with zero double-audio or crossover playback.
    Supports multilingual TTS: English variants, Hindi (hi-IN), and Marathi (mr-IN).
    """
    clean_txt = clean_for_speech(text)
    if not clean_txt:
        return
    
    # Safely escape text for JS string embedding
    js_text = json.dumps(clean_txt)
    
    # Map accent TLD to BCP 47 language code for Web Speech API
    lang_map = {
        "com": "en-US",
        "co.uk": "en-GB",
        "co.in": "en-IN",
        "com.au": "en-AU",
        "ca": "en-CA",
        "hi": "hi-IN",
        "mr": "mr-IN",
    }
    # Use explicit override if provided, else map from TLD
    voice_lang = voice_lang_override or lang_map.get(accent_tld, "en-IN")

    player_html = f"""
    <div style="display: inline-flex; align-items: center; gap: 8px; margin-top: 6px; background: #0f172a; border: 1px solid #334155; border-radius: 20px; padding: 5px 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <span style="font-size: 13px; font-weight: 600; color: #38bdf8; margin-right: 4px; display: flex; align-items: center; gap: 4px;">
            🗣️ <span id="status_{uid}">Voice</span>
        </span>
        <button id="btn_play_{uid}" onclick="play_{uid}()" title="Play Voice Speech" style="background: #10b981; border: none; color: white; border-radius: 12px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s;">▶️ Play</button>
        <button id="btn_pause_{uid}" onclick="pause_{uid}()" title="Pause Voice Speech" style="background: #f59e0b; border: none; color: white; border-radius: 12px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s;">⏸️ Pause</button>
        <button id="btn_resume_{uid}" onclick="resume_{uid}()" title="Continue Voice Speech" style="background: #3b82f6; border: none; color: white; border-radius: 12px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s;">▶️ Continue</button>
        <button id="btn_stop_{uid}" onclick="stop_{uid}()" title="Stop Voice Speech" style="background: #ef4444; border: none; color: white; border-radius: 12px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s;">🛑 Stop</button>
    </div>
    <script>
        const text_{uid} = {js_text};
        const lang_{uid} = "{voice_lang}";
        const status_{uid} = document.getElementById('status_{uid}');
        
        let isSpeaking_{uid} = false;

        function setStatus(text, color) {{
            if (status_{uid}) {{
                status_{uid}.innerText = text;
                status_{uid}.style.color = color;
            }}
        }}

        function play_{uid}() {{
            if (!('speechSynthesis' in window)) {{
                alert('Web Speech API is supported in Chrome, Edge, Safari, and modern browsers.');
                return;
            }}
            
            // Cancel any ongoing speech first to ensure single voice output
            window.speechSynthesis.cancel();
            
            const utter = new SpeechSynthesisUtterance(text_{uid});
            utter.lang = lang_{uid};
            utter.rate = 0.90;
            utter.pitch = 1.0;
            
            // Select voice matching language if available
            const voices = window.speechSynthesis.getVoices();
            for (let v of voices) {{
                if (v.lang === lang_{uid} || v.lang.startsWith(lang_{uid}.slice(0, 2))) {{
                    utter.voice = v;
                    break;
                }}
            }}

            utter.onstart = function() {{
                isSpeaking_{uid} = true;
                setStatus('Speaking', '#10b981');
            }};
            
            utter.onend = function() {{
                isSpeaking_{uid} = false;
                setStatus('Finished', '#94a3b8');
            }};
            
            utter.onerror = function(e) {{
                isSpeaking_{uid} = false;
                // If user stopped or cancelled, do not do anything or restart
                if (e.error === 'canceled' || e.error === 'interrupted') {{
                    setStatus('Stopped', '#ef4444');
                    return;
                }}
                setStatus('Ready', '#94a3b8');
            }};

            window.speechSynthesis.speak(utter);
        }}

        function pause_{uid}() {{
            if ('speechSynthesis' in window && window.speechSynthesis.speaking) {{
                window.speechSynthesis.pause();
                setStatus('Paused', '#f59e0b');
            }}
        }}

        function resume_{uid}() {{
            if ('speechSynthesis' in window) {{
                if (window.speechSynthesis.paused) {{
                    window.speechSynthesis.resume();
                    setStatus('Speaking', '#10b981');
                }} else if (!window.speechSynthesis.speaking) {{
                    play_{uid}();
                }}
            }}
        }}

        function stop_{uid}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
            }}
            isSpeaking_{uid} = false;
            setStatus('Stopped', '#ef4444');
        }}

        // Auto play if requested
        if ({str(auto_play).lower()}) {{
            setTimeout(function() {{
                play_{uid}();
            }}, 300);
        }}
    </script>
    """
    components.html(player_html, height=48)


def answer_gemini_question(question, species_data=None, api_key=None, chat_history=None, species_name=None):
    """
    Call Gemini REST API to answer ecological and species questions with multi-turn conversation memory.
    Ensures complete, thoroughly concluded answers with zero premature truncations and strict topical isolation.
    """
    q_low = question.lower()
    
    # Topic detection for strict context data isolation
    is_status = any(k in q_low for k in ["status", "conservation", "iucn", "population", "how many", "threat", "endangered"])
    is_diet = any(k in q_low for k in ["diet", "eat", "food", "prey", "feed", "forage", "nutrition"])
    is_habitat = any(k in q_low for k in ["habitat", "where", "live", "range", "distribution", "biome"])
    is_role = any(k in q_low for k in ["role", "importance", "ecological", "function", "significance"])
    is_facts = any(k in q_low for k in ["fact", "interesting", "trivia", "unique"])
    is_forest = any(k in q_low for k in ["forest", "deforest", "canopy", "gli", "tree", "carbon"])
    
    sp_name = species_name or (species_data.get('species_name') if species_data else '') or ''
    sci = species_data.get('scientific_name', '') if species_data else ''
    
    # Dedicated Topic-Specific Prompts to completely avoid cross-contamination
    if is_status:
        cons = species_data.get("conservation", "IUCN Red List Registered") if species_data else "IUCN Red List Registered"
        reg = species_data.get("habitat", "Global native biomes") if species_data else "Global native biomes"
        pop = species_data.get("estimated_population", "Monitored across international conservation reserves") if species_data else "Monitored across international conservation reserves"
        threats = species_data.get("threats", "Habitat fragmentation, deforestation, and human encroachment") if species_data else "Habitat fragmentation, deforestation, and human encroachment"
        
        prompt = (
            f"You are EcoLens Senior Ecological AI Assistant.\n"
            f"User Inquiry: {question}\n\n"
            f"Subject: {sp_name} (*{sci}*)\n"
            f"IUCN Conservation Status: {cons}\n"
            f"Native Geographic Region & Biome: {reg}\n"
            f"Estimated Wild Population: {pop}\n"
            f"Primary Environmental Threats: {threats}\n\n"
            f"STRICT OUTPUT INSTRUCTION:\n"
            f"Provide a comprehensive status response covering:\n"
            f"1. Official IUCN Conservation Status\n"
            f"2. Native Geographic Region & Biome\n"
            f"3. Estimated Wild Population count (always use statistical qualifiers like 'Estimated at approximately ~')\n"
            f"4. Primary Environmental & Anthropogenic Threats\n"
            f"CRITICAL: Do NOT mention diet, food, eating habits, or foraging behavior under any circumstances."
        )
    elif is_diet:
        diet = species_data.get("diet", "Specialized forage and food sources") if species_data else "Specialized forage and food sources"
        spec_type = species_data.get("type", "Species") if species_data else "Species"
        
        prompt = (
            f"You are EcoLens Senior Ecological AI Assistant.\n"
            f"User Inquiry: {question}\n\n"
            f"Subject: {sp_name} (*{sci}*)\n"
            f"Dietary Sources: {diet}\n"
            f"Type: {spec_type}\n\n"
            f"STRICT OUTPUT INSTRUCTION:\n"
            f"Detail ONLY the diet, food sources, prey/plants consumed, and foraging behavior of {sp_name}. Do NOT mention conservation status, population numbers, or geographic regions."
        )
    elif is_habitat:
        habitat = species_data.get("habitat", "Native forest ecosystems") if species_data else "Native forest ecosystems"
        
        prompt = (
            f"You are EcoLens Senior Ecological AI Assistant.\n"
            f"User Inquiry: {question}\n\n"
            f"Subject: {sp_name} (*{sci}*)\n"
            f"Native Biome & Distribution: {habitat}\n\n"
            f"STRICT OUTPUT INSTRUCTION:\n"
            f"Detail ONLY the natural habitat, elevation, native biomes, and geographic range of {sp_name}. Do NOT discuss diet or conservation population counts."
        )
    elif is_role:
        role = species_data.get("ecological_role", "Ecological balance") if species_data else "Ecological balance"
        
        prompt = (
            f"You are EcoLens Senior Ecological AI Assistant.\n"
            f"User Inquiry: {question}\n\n"
            f"Subject: {sp_name} (*{sci}*)\n"
            f"Ecological Function: {role}\n\n"
            f"STRICT OUTPUT INSTRUCTION:\n"
            f"Detail ONLY the ecological role, trophic function, and ecosystem importance of {sp_name}."
        )
    elif is_facts:
        fact = (species_data.get("facts") or species_data.get("fact", "Unique biological adaptation")) if species_data else "Unique biological adaptation"
        
        prompt = (
            f"You are EcoLens Senior Ecological AI Assistant.\n"
            f"User Inquiry: {question}\n\n"
            f"Subject: {sp_name} (*{sci}*)\n"
            f"Unique Fact: {fact}\n\n"
            f"STRICT OUTPUT INSTRUCTION:\n"
            f"Provide fascinating, unique biological facts and adaptations about {sp_name}."
        )
    elif is_forest:
        prompt = (
            f"You are EcoLens Senior Ecological AI Assistant.\n"
            f"User Inquiry: {question}\n\n"
            f"Subject: {sp_name}\n\n"
            f"STRICT OUTPUT INSTRUCTION:\n"
            f"Explain how forest canopy loss, deforestation, and climate change impact {sp_name} and its native biome."
        )
    else:
        # General conversational prompt with minimal history
        hist_text = ""
        if chat_history:
            for turn in chat_history[-2:]:
                r = "User" if turn["role"] == "user" else "EcoLens"
                hist_text += f"{r}: {turn['text']}\n"
        prompt = (
            f"You are EcoLens Senior Ecological AI Assistant.\n"
            f"Active Species: {sp_name}\n"
            f"Recent Context:\n{hist_text}\n"
            f"User Inquiry: {question}\n\n"
            f"Answer thoroughly and accurately for the user's inquiry."
        )
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 0.2
        }
    }
    
    try:
        res_data = _call_gemini_api(payload, api_key, timeout=30)
        candidates = res_data.get('candidates', [])
        if candidates:
            parts = candidates[0].get('content', {}).get('parts', [])
            ans_text = "".join(p.get('text', '') for p in parts).strip()
            if ans_text:
                return ans_text
        return "EcoLens analyzed your inquiry but received an empty response. Please try rephrasing."
    except Exception as e:
        return f"Note: Gemini temporarily busy ({e}). Answering from local ecological knowledge base."


def query_gemini_forest_diagnostics(image, prediction, confidence, canopy_percent, api_key):
    """
    Multimodal AI Ecological, Carbon & Deforestation Diagnostic via Gemini.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    res_json = _call_gemini_api(payload, api_key, timeout=30)
    candidates = res_json.get('candidates', [])
    if candidates:
        text_content = candidates[0]['content']['parts'][0]['text']
        return _parse_gemini_json(text_content)
    raise RuntimeError("Received empty response candidate from Gemini API.")


# ============================================================
# AI SPECIES IDENTIFICATION ENGINE
# ============================================================

# SPECIES_NAME_MAP is now imported from species_db.py


def _match_to_database(api_name, species_type):
    """
    Match an API-returned species name to a SPECIES_DATABASE key.
    Uses the SPECIES_NAME_MAP for fuzzy matching.
    Returns (matched_db_key, confidence_boost) or (None, 0).
    """
    if not api_name:
        return None, 0

    name_lower = api_name.lower().strip()

    # 1. Direct map lookup
    if name_lower in SPECIES_NAME_MAP:
        return SPECIES_NAME_MAP[name_lower], 0

    # 2. Check if any map key is contained in the name
    for key, db_name in SPECIES_NAME_MAP.items():
        if key in name_lower or name_lower in key:
            return db_name, 0

    # 3. Check against DB keys directly
    for db_key in st.session_state.species_database:
        if db_key.lower() in name_lower or name_lower in db_key.lower():
            return db_key, 0

    # 4. Check against scientific names in DB
    for db_key, data in st.session_state.species_database.items():
        sci = data.get("scientific_name", "").lower()
        if sci and (sci in name_lower or name_lower in sci):
            return db_key, 0

    return None, 0


def _query_inaturalist_vision(image, species_type):
    """
    Query the iNaturalist API for species identification using image search.
    Falls back to text-based taxa search using ImageNet predictions.
    Returns list of (species_name, confidence) tuples.
    """
    results = []
    target = species_type.lower()

    # Strategy 1: Use iNaturalist taxa autocomplete with ImageNet prediction
    # First get an ImageNet prediction to guide the search
    imagenet_label = ""
    imagenet_conf = 0.0

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.eval()
        tfm = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
        img_rgb = image.convert("RGB")
        tensor = tfm(img_rgb).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)

            # Get top 5 predictions
            top5_probs, top5_idxs = torch.topk(probs, 5, dim=1)
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            categories = weights.meta["categories"]

            for i in range(5):
                label = categories[top5_idxs[0, i].item()].lower()
                conf = float(top5_probs[0, i].item() * 100)

                # Try to match this ImageNet label to our database
                matched, _ = _match_to_database(label, target)
                if matched:
                    # Verify the match is the right type
                    db_data = st.session_state.species_database.get(matched, {})
                    if target in db_data.get("type", "").lower():
                        results.append((matched, conf))

                if i == 0:
                    imagenet_label = label
                    imagenet_conf = conf
    except Exception:
        pass

    # Strategy 2: Query iNaturalist API for refined identification
    try:
        # Search iNaturalist with the ImageNet label as a query
        search_term = imagenet_label.replace("_", " ") if imagenet_label else species_type
        taxon_filter = "Plantae" if target == "plant" else "Animalia"

        res = requests.get(
            "https://api.inaturalist.org/v1/taxa",
            params={
                "q": search_term,
                "per_page": 10,
                "rank": "species,genus",
                "is_active": "true",
            },
            timeout=5,
        )

        if res.status_code == 200:
            taxa = res.json().get("results", [])
            for taxon in taxa:
                taxon_name = taxon.get("preferred_common_name", "") or taxon.get("name", "")
                sci_name = taxon.get("name", "")
                iconic = (taxon.get("iconic_taxon_name", "") or "").lower()

                # Filter by kingdom
                if target == "plant" and iconic not in ("plantae", "fungi", ""):
                    continue
                if target == "animal" and iconic not in ("animalia", "aves", "mammalia",
                                                          "reptilia", "amphibia", "insecta",
                                                          "arachnida", "mollusca", "actinopterygii", ""):
                    continue

                # Try matching to our database
                matched, _ = _match_to_database(taxon_name, target)
                if not matched:
                    matched, _ = _match_to_database(sci_name, target)

                if matched and matched not in [r[0] for r in results]:
                    db_data = st.session_state.species_database.get(matched, {})
                    if target in db_data.get("type", "").lower():
                        # Give API results a reasonable confidence
                        api_conf = max(70.0, imagenet_conf * 0.85) if imagenet_conf > 0 else 75.0
                        results.append((matched, api_conf))

                # Also return the raw iNaturalist name if not in DB
                if not matched and taxon_name:
                    # Create a dynamic entry name
                    display_name = taxon_name.title() if taxon_name else sci_name
                    if display_name and display_name not in [r[0] for r in results]:
                        results.append((display_name, 65.0))

    except Exception:
        pass  # Network error — fall through to offline fallback

    return results


def identify_species_demo(image, species_type):
    """
    AI Biodiversity Identification Engine.
    Uses Google Gemini Vision API / Pl@ntNet API if keys are present.
    Falls back to offline iNaturalist/ImageNet matching logic when no key is present.
    """
    target = species_type.lower()
    
    # Get all DB candidates of the correct type from session state database
    db_candidates = [
        name for name, data in st.session_state.species_database.items()
        if target in data.get("type", "").lower()
    ]

    # Check for active Gemini API Key
    gemini_api_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    
    if gemini_api_key:
        try:
            data = None
            
            plantnet_key = st.session_state.get("plantnet_api_key") or st.secrets.get("PLANTNET_API_KEY") or os.environ.get("PLANTNET_API_KEY")
            if target == "plant" and plantnet_key:
                st.info("🌱 Querying Pl@ntNet API for botanical identification...")
                pnet_res = query_plantnet_api(image, plantnet_key)
                if pnet_res:
                    st.info(f"🌿 Pl@ntNet matched scientific name: {pnet_res['scientific_name']}. Querying Gemini for profile details...")
                    data = query_gemini_details_by_name(pnet_res['scientific_name'], gemini_api_key)
            
            # If not identified yet (or animal search), query Gemini Vision directly
            if not data:
                st.info("👁️ Running Gemini Vision analysis...")
                data = query_gemini_vision(image, species_type, gemini_api_key)
            
            if data and ("species_name" in data or "scientific_name" in data):
                species_name = data.get("species_name") or data.get("scientific_name")
                data["species_name"] = species_name
                data.setdefault("scientific_name", species_name)
                data.setdefault("type", species_type.title())
                data.setdefault("family", "N/A")
                data.setdefault("habitat", "Natural ecosystems")
                data.setdefault("diet", "N/A")
                data.setdefault("conservation", "Recorded in Biodiversity Archive")
                data.setdefault("conserved", False)
                data.setdefault("estimated_population", "N/A")
                data.setdefault("ecological_role", "Contributes to ecosystem biodiversity.")
                data.setdefault("importance", "Ecologically significant species.")
                data.setdefault("threats", "Habitat alteration and environmental change.")
                data.setdefault("facts", data.get("fact") or f"{species_name} is documented in international biodiversity archives.")
                
                # Cache the results dynamically inside session state database
                st.session_state.species_database[species_name] = data
                
                candidates = [species_name] + [c for c in db_candidates if c != species_name]
                
                return {
                    "status": "identified",
                    "species": species_name,
                    "confidence": 98.5,
                    "candidates": candidates,
                    "message": f"Successfully identified {species_name} via Gemini API.",
                }
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower() or "rate limit" in err_msg.lower():
                st.warning("⚠️ Gemini free-tier rate limit reached. Seamlessly switched to offline MobileNet & 81-Species Database.")
            else:
                st.info(f"💡 Cloud AI fallback: {err_msg[:100]}... Using offline model.")
            # Fall through to offline logic

    # --- OFFLINE FALLBACK LOGIC ---
    # 1. Query the AI identification pipeline
    api_results = _query_inaturalist_vision(image, species_type)

    # 2. If we got results, use the top one
    if api_results:
        top_species, top_conf = api_results[0]
        confidence = round(min(98.5, max(60.0, top_conf)), 1)

        # Collect all unique candidates from API results
        all_candidates = [r[0] for r in api_results]
        # Add remaining DB candidates not already in the list
        for c in db_candidates:
            if c not in all_candidates:
                all_candidates.append(c)

        return {
            "status": "identified",
            "species": top_species,
            "confidence": confidence,
            "candidates": all_candidates,
            "message": f"Identified {top_species} ({confidence}% confidence).",
        }

    # 3. Fallback: Use ImageNet label mapping directly
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.eval()
        tfm = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
        img_rgb = image.convert("RGB")
        tensor = tfm(img_rgb).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            predicted_label = weights.meta["categories"][top_idx.item()].lower()
            top_conf = float(top_prob.item() * 100)

        matched, _ = _match_to_database(predicted_label, target)
        if matched:
            return {
                "status": "identified",
                "species": matched,
                "confidence": round(min(95.0, max(55.0, top_conf)), 1),
                "candidates": db_candidates,
                "message": f"Identified {matched} (offline mode).",
            }
    except Exception:
        pass

    # 4. Final fallback: return the first DB candidate with low confidence
    fallback = db_candidates[0] if db_candidates else list(st.session_state.species_database.keys())[0]
    return {
        "status": "identified",
        "species": fallback,
        "confidence": 50.0,
        "candidates": db_candidates,
        "message": f"Best guess: {fallback}. Try a clearer photo for better results.",
    }



# ============================================================
# VOICE ASSISTANT KNOWLEDGE ENGINE
# ============================================================

def answer_local_question(
    question,
    species_name=None,
    chat_history=None,
):
    """
    Answers ecological, forest, and species questions using Gemini or offline knowledge base.
    """
    q = question.lower().strip()

    # 1. Retrieve API key and data safely
    gemini_api_key = None
    try:
        gemini_api_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")

    data = None
    try:
        data = st.session_state.species_database.get(species_name) if species_name else None
    except Exception:
        data = SPECIES_DATABASE.get(species_name) if species_name else None

    # Auto-resolve species if not explicitly provided
    if not data:
        for sname in st.session_state.species_database.keys():
            if sname.lower() in q:
                species_name = sname
                data = st.session_state.species_database[sname]
                break
        if not data and SPECIES_NAME_MAP:
            for alias, mapped in SPECIES_NAME_MAP.items():
                if alias.lower() in q:
                    species_name = mapped
                    data = st.session_state.species_database.get(mapped)
                    break

    # 2. DETERMINISTIC OFFLINE HANDLERS FIRST — for known single-topic queries
    #    These are guaranteed to return the correct topic without any LLM hallucination.
    if data:
        sci_name = data.get("scientific_name") or species_name
        habitat = data.get("habitat") or "Natural forest and grassland ecosystems"
        diet = data.get("diet") or "N/A"
        conservation = data.get("conservation") or "Documented in IUCN Red List"
        threats = data.get("threats") or "Habitat fragmentation, land-use change, and environmental pressures"
        ecological_role = data.get("ecological_role") or "maintaining vital trophic and ecological balance"
        facts = data.get("facts") or data.get("fact") or f"{species_name} is documented in international biodiversity archives."
        family = data.get("family") or "Documented in biological taxonomy"
        spec_type = data.get("type") or "Species"
        region = data.get("region") or "Native ecosystems worldwide"

        # --- Status / Conservation (checked FIRST — most common chip query) ---
        if (
            "status" in q
            or "conservation" in q
            or "endangered" in q
            or "iucn" in q
            or "population" in q
            or "number" in q
            or "how many" in q
        ):
            pop = data.get("estimated_population") or "Monitored across international conservation reserves"
            if pop == "N/A" or not pop:
                pop_str = "Approximately abundant across native range / Monitored in protected habitats"
            elif any(w in str(pop).lower() for w in ["approx", "estimated", "~", "roughly", "around"]):
                pop_str = str(pop)
            else:
                pop_str = f"Estimated at approximately ~{pop} individuals in the wild"

            reg = data.get("region") or data.get("habitat") or "Global natural biomes"
            cons = data.get("conservation") or "Recorded in IUCN database"
            threats_info = data.get("threats") or "Habitat alteration, land conversion, and climate stress"
            return (
                f"📊 **Conservation Status of {species_name} (*{sci_name}*):**\n\n"
                f"• **Official IUCN Status:** **{cons}**\n\n"
                f"• **Native Region:** {reg}\n\n"
                f"• **Natural Habitat:** {habitat}\n\n"
                f"• **Estimated Wild Population:** **{pop_str}**\n\n"
                f"• **Primary Threats:** {threats_info}"
            )

        # --- Diet (word-boundary check for "eat" to avoid matching "threats") ---
        import re
        if "diet" in q or re.search(r'\beat\b', q) or "food" in q or "prey" in q or "feed" in q or "forage" in q or "nutrition" in q:
            return (
                f"🍽️ **Diet & Nutrition of {species_name} (*{sci_name}*):**\n\n"
                f"• **Food Sources:** {diet}\n\n"
                f"• **Foraging & Feeding Dynamics:** As a {spec_type.lower()}, it relies on foraging and nutrient intake adapted to its native physiology and local food web."
            )

        # --- Habitat / Distribution ---
        if "habitat" in q or "where" in q or "live" in q or "range" in q or "distribution" in q or "region" in q:
            return (
                f"🗺️ **Habitat & Distribution of {species_name} (*{sci_name}*):**\n\n"
                f"• **Found In:** {region}\n\n"
                f"• **Natural Biome:** {habitat}\n\n"
                f"• **Geographic Adaptation:** Specifically adapted to the elevation, canopy cover, and moisture gradients of this ecosystem."
            )

        # --- Role / Importance ---
        if "role" in q or "importance" in q or "ecological" in q or "function" in q or "significance" in q:
            return (
                f"🌿 **Ecological Role of {species_name} (*{sci_name}*):**\n\n"
                f"• **Core Ecosystem Role:** {ecological_role}\n\n"
                f"• **Trophic Significance:** Functions as an integral biological component supporting biodiversity stability."
            )

        # --- Facts / Trivia ---
        if "fact" in q or "interesting" in q or "tell me" in q or "unique" in q:
            return (
                f"💡 **Fascinating Biological Fact:**\n\n"
                f"• {facts}\n\n"
                f"• **Taxonomy:** *{sci_name}* ({spec_type}, Family: {family})."
            )

        # --- Forest / Deforestation Impact ---
        if "forest" in q or "deforest" in q or "tree" in q or "canopy" in q or "climate" in q:
            return (
                f"🌲 **Forest & Canopy Impact on {species_name}:**\n\n"
                f"• **Canopy Vulnerability:** Deforestation and habitat fragmentation directly degrade the {habitat} required by {species_name}.\n\n"
                f"• **Conservation Urgency:** Preserving continuous tree canopy and remote sensing tracking help maintain stable populations."
            )

        # --- Scientific Name / Taxonomy ---
        if "scientific" in q or "scientific name" in q or "taxonomy" in q or "family" in q:
            return (
                f"🔬 **Taxonomy of {species_name}:**\n\n"
                f"• **Scientific Name:** ***{sci_name}***\n\n"
                f"• **Classification:** {spec_type}\n\n"
                f"• **Family:** {family}\n\n"
                f"• **IUCN Status:** {conservation}"
            )

        # --- Threats ---
        if "threat" in q or "danger" in q or "risk" in q or "extinction" in q:
            return (
                f"⚠️ **Environmental Threats to {species_name}:**\n\n"
                f"• **Primary Pressures:** {threats}\n\n"
                f"• **Current Status:** Categorized under **{conservation}**."
            )

    # 3. GEMINI FALLBACK — only for general/free-form questions that don't match any known topic
    if gemini_api_key:
        try:
            gemini_ans = answer_gemini_question(question, data, gemini_api_key, chat_history=chat_history, species_name=species_name)
            if gemini_ans and not gemini_ans.startswith("Error communicating") and not gemini_ans.startswith("Note: Gemini query timed out"):
                return gemini_ans
        except Exception:
            pass

    # 4. General environmental offline knowledge base
    if "status" in q or "conservation" in q or "iucn" in q or "population" in q or "endangered" in q:
        return (
            "📊 **Global Conservation & IUCN Status Overview:**\n\n"
            "The International Union for Conservation of Nature (IUCN) classifies species survival risk into standard global categories:\n\n"
            "• **Critically Endangered (CR):** Extreme risk of extinction in the wild.\n"
            "• **Endangered (EN):** High risk of extinction in the wild (e.g. Asian Elephant, Bengal Tiger, Snow Leopard).\n"
            "• **Vulnerable (VU):** High risk of endangerment in the medium-term (e.g. Teak, Olive Ridley Sea Turtle).\n"
            "• **Least Concern (LC):** Widespread and abundant populations (e.g. Neem, Banyan Tree).\n\n"
            "💡 *Tip: Select a specific species from the 'Switch Subject Context' dropdown above to see its exact wild population count, native region, and conservation status!*"
        )

    if "deforest" in q or "forest loss" in q or "cutting trees" in q:
        return (
            f"🌲 **Deforestation & Canopy Loss Overview:**\n\n"
            f"Deforestation represents the large-scale removal and clearing of forested land for agriculture, logging, or urbanization.\n\n"
            f"• **Carbon Emissions:** Tropical deforestation releases an estimated ~145 tonnes of CO2 per hectare into the atmosphere.\n"
            f"• **Biodiversity Impact:** Forest clearing directly destroys critical habitats for over 80% of terrestrial plant and animal species.\n"
            f"• **EcoLens Analytics:** Using deep learning (ResNet-18) combined with Green Leaf Index (GLI) remote sensing, EcoLens detects canopy loss in real time from satellite imagery."
        )
    if "canopy" in q or "gli" in q or "remote sensing" in q or "vegetation" in q:
        return (
            f"🛰️ **Remote Sensing & Canopy Health (GLI / VARI):**\n\n"
            f"EcoLens calculates multispectral remote sensing indices from optical imagery to measure vegetation density and canopy vigor:\n\n"
            f"• **Green Leaf Index (GLI):** `(2*G - R - B) / (2*G + R + B)` isolates chlorophyll absorption, distinguishing healthy intact crowns (GLI > 0.10) from bare ground.\n"
            f"• **Visible Atmospherically Resistant Index (VARI):** Minimizes atmospheric distortions for reliable satellite canopy tracking.\n"
            f"• **Canopy Segmentation:** Provides automated pixel-level canopy coverage percentages and estimated carbon loss."
        )
    if "carbon" in q or "co2" in q or "climate" in q or "sequestration" in q:
        return (
            f"💨 **Forest Carbon Sequestration & Climate Balance:**\n\n"
            f"Forests act as Earth's primary terrestrial carbon sinks, storing gigatonnes of atmospheric carbon in biomass and root soils.\n\n"
            f"• **Sequestration Rate:** Mature primary forests absorb between 2.5 to 11 tonnes of CO2 per hectare annually.\n"
            f"• **Degradation Impact:** When canopies are cleared or burned, stored carbon is rapidly released as greenhouse gases.\n"
            f"• **Preservation Benefit:** Halting deforestation is recognized by climate scientists as one of the most cost-effective nature-based solutions."
        )
    if "help" in q or "capabilities" in q or "who are you" in q:
        return (
            f"👋 **Welcome! I am the EcoLens Ecological AI Voice Assistant.**\n\n"
            f"Here is how I can assist you:\n"
            f"• **Species Intelligence:** Ask about any scanned plant or animal (diet, scientific name, habitat, IUCN status, threats, and ecological role).\n"
            f"• **Forest Monitoring:** Inquire about satellite canopy health, Grad-CAM neural attention, and deforestation risk scores.\n"
            f"• **Conservation Insights:** Explore environmental science, carbon sequestration calculations, and biodiversity preservation strategies."
        )

    if species_name and data:
        return (
            f"🌿 **Species Overview: {species_name} (*{data.get('scientific_name', species_name)}*)**\n\n"
            f"• **Classification:** {data.get('type', 'Species')} (Family: {data.get('family', 'N/A')})\n"
            f"• **Native Habitat:** {data.get('habitat', 'Natural forest and grassland biomes')}\n"
            f"• **Conservation Status:** **{data.get('conservation', 'Recorded in database')}**\n"
            f"• **Diet & Role:** Consumes {data.get('diet', 'N/A')} and functions in {data.get('ecological_role', 'ecological stability')}.\n\n"
            f"Feel free to ask specific questions regarding its habitat distribution, conservation threats, or biological adaptations!"
        )

    return (
        f"🌿 **EcoLens Environmental Intelligence Ready!**\n\n"
        f"You can ask any detailed ecological or conservation question, or explore our database of plants, animals, and satellite forest canopy diagnostics. "
        f"For extended multimodal analysis, ensure your Google Gemini API Key is configured in the sidebar."
    )


# ============================================================
# PDF REPORT
# ============================================================

def create_forest_pdf(
    image,
    result,
    veg_result=None,
    gemini_diag=None,
    coordinates=None,
    place_name=None,
):

    temp_image = "ecolens_temp.jpg"

    image.save(
        temp_image
    )

    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=15,
    )

    story = []

    story.append(
        Paragraph(
            "🌲 EcoLens Environmental & Canopy Intelligence Report",
            title,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    canopy_str = f"{veg_result['canopy_cover_percent']}%" if veg_result else "N/A"
    carbon_str = f"~{veg_result['est_carbon_loss_per_ha']} t CO2/ha" if veg_result else "N/A"
    inf_str = f"{result.get('inference_time', 0.0):.3f}s"
    if coordinates:
        coord_str = f"{coordinates[0]:.4f}°, {coordinates[1]:.4f}°"
        if place_name:
            coord_str = f"{place_name[:35]} ({coord_str})"
    else:
        coord_str = "Uploaded / Camera Capture"

    data_metrics = [
        ["Date & Time:", result["timestamp"], "Classification:", result["prediction"]],
        ["Model Confidence:", f"{result['confidence']:.2f}%", "Deforestation Risk:", result["risk"]],
        ["Severity Score:", f"{result['severity']:.1f}/100", "Canopy Coverage:", canopy_str],
        ["Est. Carbon Loss:", carbon_str, "Location / Coords:", coord_str],
    ]

    t = Table(data_metrics, colWidths=[120, 150, 120, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#1e293b")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(
        RLImage(
            temp_image,
            width=280,
            height=200,
        )
    )

    story.append(
        Spacer(1, 15)
    )

    if gemini_diag:
        story.append(Paragraph("<b>AI Ecological & Carbon Assessment</b>", styles["Heading2"]))
        biome = gemini_diag.get("forest_biome", "N/A")
        drivers = gemini_diag.get("probable_drivers", "N/A")
        threat = gemini_diag.get("biodiversity_threat_level", "N/A")
        plan = gemini_diag.get("remediation_plan", "N/A")
        story.append(Paragraph(f"<b>Forest Biome:</b> {biome}", styles["Normal"]))
        story.append(Paragraph(f"<b>Probable Drivers:</b> {drivers}", styles["Normal"]))
        story.append(Paragraph(f"<b>Biodiversity Threat Level:</b> {threat}", styles["Normal"]))
        story.append(Paragraph(f"<b>Restoration Strategy:</b> {plan}", styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "<b>Recommended Action</b>",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            result["recommendation"],
            styles["Normal"],
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            """
            <i>Scientific note: This automated report combines deep learning scene classification (ResNet-18)
            with optical remote sensing Green Leaf Index (GLI) analytics. Carbon estimates are benchmarks based on
            IPCC biomass baselines and should be verified with field and LiDAR measurements.</i>
            """,
            styles["Italic"],
        )
    )

    doc.build(
        story
    )

    if os.path.exists(
        temp_image
    ):

        os.remove(
            temp_image
        )

    return pdf_buffer.getvalue()


def generate_forest_pdf(result, image, veg_result=None, gemini_diag=None):
    return create_forest_pdf(image, result, veg_result, gemini_diag)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌲 EcoLens 2.0"
)

st.sidebar.caption(
    "AI Biodiversity & Forest Intelligence"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Configuration")

gemini_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    value=st.session_state.get("gemini_api_key", ""),
    type="password",
    help="Enter your Gemini API key to enable real-time dynamic species identification, population estimates, and voice assistant.",
)
if gemini_key_input:
    st.session_state["gemini_api_key"] = gemini_key_input

default_plantnet_key = ""
try:
    default_plantnet_key = st.secrets.get("PLANTNET_API_KEY", "") or os.environ.get("PLANTNET_API_KEY", "")
except Exception:
    pass

plantnet_key_input = st.sidebar.text_input(
    "Pl@ntNet API Key",
    value=st.session_state.get("plantnet_api_key", default_plantnet_key),
    type="password",
    help="Enter your Pl@ntNet API key to enable precision botanical identification for plants.",
)
if plantnet_key_input:
    st.session_state["plantnet_api_key"] = plantnet_key_input

gemini_active = bool(st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY"))
plantnet_active = bool(st.session_state.get("plantnet_api_key"))

if gemini_active:
    st.sidebar.success("✅ Google Gemini Active")
else:
    st.sidebar.warning("⚠️ Gemini Key Not Set")

if plantnet_active:
    st.sidebar.success("✅ Pl@ntNet Active")
else:
    st.sidebar.info("ℹ️ Pl@ntNet Key Not Set")

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Display Theme")
theme_choice = st.sidebar.selectbox(
    "Select Theme",
    ["🌲 Biosphere Dark", "🌿 Emerald Forest", "🌑 Midnight OLED", "☀️ Daylight Eco"],
    index=0
)

if theme_choice == "🌿 Emerald Forest":
    st.markdown("""<style>
    .stApp { background-color: #031c13 !important; color: #e6fcf0 !important; }
    [data-testid="stSidebar"] { background-color: #0c3828 !important; }
    </style>""", unsafe_allow_html=True)
elif theme_choice == "🌑 Midnight OLED":
    st.markdown("""<style>
    .stApp { background-color: #000000 !important; color: #f3f4f6 !important; }
    [data-testid="stSidebar"] { background-color: #111827 !important; }
    </style>""", unsafe_allow_html=True)
elif theme_choice == "☀️ Daylight Eco":
    st.markdown("""<style>
    .stApp { background-color: #f8fafc !important; color: #0f172a !important; }
    [data-testid="stSidebar"] { background-color: #f1f5f9 !important; }
    .card { background-color: #ffffff !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; }
    </style>""", unsafe_allow_html=True)

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Choose Module",
    [
        "🏠 Dashboard",
        "🌲 Forest AI",
        "🌿 Plant Scanner",
        "🐅 Animal Scanner",
        "🎙️ AI Voice Assistant",
        "📚 Species Database",
        "📊 Biodiversity History",
        "⚙️ System",
    ],
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🌲 EcoLens 2.0</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-Powered Forest, Plant, Animal & Biodiversity Intelligence"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if mode == "🏠 Dashboard":

    st.header(
        "🌍 Environmental Intelligence Dashboard"
    )

    st.write(
        """
        EcoLens combines forest-condition analysis,
        biodiversity scanning and an AI assistant into
        one environmental monitoring platform.
        """
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:
        st.metric(
            "🌲 Forest AI",
            "ONLINE" if MODEL_LOADED else "ERROR",
        )

    with c2:
        st.metric(
            "🌿 Plant Scanner",
            "READY",
        )

    with c3:
        st.metric(
            "🐅 Animal Scanner",
            "READY",
        )

    with c4:
        st.metric(
            "🎙️ Voice Assistant",
            "READY",
        )

    st.markdown(
        "---"
    )

    st.subheader(
        "🚀 EcoLens Capabilities"
    )

    capabilities = pd.DataFrame(
        {
            "Module": [
                "🌲 Forest AI",
                "🗺️ Satellite Map AI",
                "⏳ Change Detection",
                "🌿 Plant Scanner",
                "🐅 Animal Scanner",
                "🎙️ Voice Assistant",
                "🧠 Multi-Layer XAI",
                "📄 Environmental Reports",
            ],

            "Capability": [
                "ResNet-18 Healthy vs Deforested classification with optical remote sensing validation",
                "Interactive global satellite map — pin any region on Earth to fetch & analyze live tiles",
                "Temporal Before vs After canopy change detection & deforestation difference maps",
                "Camera & API-driven dynamic botanical identification (Pl@ntNet + Gemini)",
                "Camera & API-driven dynamic wildlife identification with IUCN conservation status",
                "Context-aware AI voice assistant answering species & ecological inquiries",
                "Grad-CAM CNN attention, Canopy Segmentation Mask, and Vegetation Health Density Gradient",
                "Comprehensive multi-metric PDF reports with GPS geotagging and carbon stock estimates",
            ],
        }
    )

    st.dataframe(
        capabilities,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOREST AI
# ============================================================

elif mode == "🌲 Forest AI":

    st.header(
        "🌲 Forest AI Environmental Intelligence Engine"
    )

    st.write(
        """
        Comprehensive forest monitoring combining ResNet-18 deep learning classification,
        optical remote sensing vegetation indices (GLI/VARI), multi-layer Explainable AI (XAI),
        and multimodal Gemini ecological impact assessments.
        """
    )

    forest_submode = st.radio(
        "Select Forest AI Analysis Workflow:",
        [
            "🛰️ Single Image Intelligence",
            "🗺️ Interactive Satellite Map (Pin & Analyze)",
            "⏳ Temporal Change Detection (Before vs After)",
        ],
        horizontal=True,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # WORKFLOW 1: SINGLE IMAGE INTELLIGENCE
    # --------------------------------------------------------
    if forest_submode == "🛰️ Single Image Intelligence":

        confidence_threshold = st.slider(
            "Minimum Confidence Threshold (%)",
            50,
            95,
            70,
            5,
            help="Threshold below which the system flags classification as low confidence.",
        )

        source = st.radio(
            "Image Source",
            [
                "📁 Upload Image",
                "📷 Capture via Camera",
            ],
            horizontal=True,
        )

        if source == "📁 Upload Image":
            uploaded = st.file_uploader(
                "Upload forest / satellite / aerial image",
                type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
            )
        else:
            uploaded = st.camera_input("Capture forest scene")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

            col_preview, col_action = st.columns([1, 1])

            with col_preview:
                st.image(
                    image,
                    caption="Input Forest Region",
                    use_container_width=True,
                )

            with col_action:
                st.markdown("### 🔍 Environmental Analysis")
                st.write("Run deep learning classification, canopy loss quantification, and AI diagnostics.")
                
                run_btn = st.button("🚀 Run Full Forest AI Inspection", type="primary", use_container_width=True)

            if run_btn:
                if not MODEL_LOADED:
                    st.error(f"Forest AI Model Error: {MODEL_ERROR}")
                else:
                    with st.spinner("🌲 Processing ResNet-18 Classification & Remote Sensing Indices..."):
                        # 1. Remote Sensing & Canopy Indices
                        veg_result = compute_vegetation_indices(image)
                        st.session_state.last_veg_result = veg_result

                        # 2. Deep Learning Classification with Optical Rectification
                        result = forest_prediction(image, veg_result)
                        st.session_state.last_forest_result = result
                        st.session_state.last_forest_image = image

                        # 3. Log History
                        st.session_state.history.append(
                            {
                                "Date": result["timestamp"],
                                "Module": "Forest AI",
                                "Result": result["prediction"],
                                "Confidence": result["confidence"],
                            }
                        )

                    # 4. Optional Multimodal Gemini Ecological Diagnostic
                    gemini_api_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
                    if gemini_api_key:
                        with st.spinner("🤖 Querying Gemini Multimodal Ecological & Carbon Diagnostic..."):
                            try:
                                diag = query_gemini_forest_diagnostics(
                                    image,
                                    result["prediction"],
                                    result["confidence"],
                                    veg_result["canopy_cover_percent"],
                                    gemini_api_key,
                                )
                                st.session_state.last_forest_gemini_diag = diag
                            except Exception as diag_err:
                                st.warning(f"Note on AI Ecological Diagnostic: {diag_err}")
                                st.session_state.last_forest_gemini_diag = None
                    else:
                        st.session_state.last_forest_gemini_diag = None

            result = st.session_state.get("last_forest_result")
            veg_result = st.session_state.get("last_veg_result")
            gemini_diag = st.session_state.get("last_forest_gemini_diag")

            if result and veg_result:
                st.markdown("---")
                st.subheader("📊 Forest Intelligence Dashboard")

                if result["prediction"] == "Healthy Forest":
                    st.success(
                        f"🌿 **Healthy Forest Cover Detected** — {result['confidence']:.2f}% confidence "
                        f"(Canopy Coverage: {veg_result['canopy_cover_percent']}%)"
                    )
                else:
                    st.error(
                        f"⚠️ **Potential Deforestation / Forest Degradation Detected** — {result['confidence']:.2f}% confidence "
                        f"(Estimated Canopy Loss / Bare Ground: {veg_result['bare_ground_percent']}%)"
                    )

                # 4-Column Key Metric Cards
                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.metric(
                        "Classification",
                        result["prediction"],
                        f"{result['confidence']:.1f}% conf",
                    )

                with m2:
                    st.metric(
                        "Deforestation Risk",
                        result["risk"],
                        f"Severity: {result['severity']:.1f}/100",
                    )

                with m3:
                    st.metric(
                        "Canopy Coverage",
                        f"{veg_result['canopy_cover_percent']}%",
                        f"Bare Earth: {veg_result['bare_ground_percent']}%",
                    )

                with m4:
                    st.metric(
                        "Est. Carbon Loss",
                        f"~{veg_result['est_carbon_loss_per_ha']} t/ha",
                        f"Mean GLI: {veg_result['mean_gli']}",
                    )

                # Multi-View Explainable AI Tabs
                st.markdown("### 🧠 Multi-View Explainable AI & Remote Sensing")
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🛰️ Original Scene",
                    "🧠 Grad-CAM Attention Heatmap",
                    "🌿 Canopy Segmentation Mask",
                    "📊 Vegetation Health Density",
                ])

                with tab1:
                    st.image(image, caption="Original Input Imagery", use_container_width=True)

                with tab2:
                    with st.spinner("Rendering Grad-CAM Heatmap..."):
                        heatmap = gradcam(image, FOREST_CLASSES.index(result["prediction"]))
                    if heatmap:
                        st.image(heatmap, caption="Grad-CAM Deep Learning Focus (ResNet-18)", use_container_width=True)
                    else:
                        st.info("Grad-CAM overlay not available for this run.")

                with tab3:
                    st.image(
                        veg_result["canopy_mask_img"],
                        caption="Canopy Segmentation (Green = Intact Canopy, Red = Bare / Deforested Ground)",
                        use_container_width=True,
                    )
                    st.caption(
                        f"🟢 Canopy Area: **{veg_result['canopy_cover_percent']}%** | 🔴 Bare Ground / Loss: **{veg_result['bare_ground_percent']}%**"
                    )

                with tab4:
                    st.image(
                        veg_result["veg_health_img"],
                        caption="Vegetation Health Density Gradient (Green = High Density, Yellow = Sparse, Red = Non-Vegetated)",
                        use_container_width=True,
                    )

                # Gemini AI Ecological Diagnostic Section
                if gemini_diag:
                    st.markdown("### 🤖 Gemini AI Ecological & Carbon Impact Assessment")
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.write(f"🌲 **Forest Biome:** {gemini_diag.get('forest_biome', 'N/A')}")
                        st.write(f"🔍 **Canopy Integrity:** {gemini_diag.get('canopy_integrity', 'N/A')}")
                        st.write(f"🚜 **Probable Drivers:** {gemini_diag.get('probable_drivers', 'N/A')}")
                    with dc2:
                        st.write(f"⚠️ **Biodiversity Threat:** {gemini_diag.get('biodiversity_threat_level', 'N/A')}")
                        st.write(f"💨 **Carbon Impact:** {gemini_diag.get('carbon_risk_assessment', 'N/A')}")
                        st.write(f"🌊 **Key Hazards:** {gemini_diag.get('key_ecological_risks', 'N/A')}")

                    st.info(f"💡 **Remediation & Restoration Action Plan:** {gemini_diag.get('remediation_plan', 'N/A')}")
                elif not gemini_api_key:
                    st.info("💡 *Enter a Google Gemini API Key in the sidebar to unlock automated ecological biome profiling, driver attribution, and tailored restoration plans.*")

                # Probability Distribution Chart
                st.markdown("### 📈 Class Probabilities")
                probability_df = pd.DataFrame(
                    {
                        "Class": list(result["probabilities"].keys()),
                        "Probability (%)": list(result["probabilities"].values()),
                    }
                )
                st.bar_chart(probability_df, x="Class", y="Probability (%)")

                # PDF Export
                st.markdown("---")
                pdf = create_forest_pdf(image, result, veg_result, gemini_diag)
                st.download_button(
                    "📄 Download Environmental & Canopy Intelligence Report (PDF)",
                    data=pdf,
                    file_name="EcoLens_Forest_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    # --------------------------------------------------------
    # WORKFLOW 2: INTERACTIVE SATELLITE MAP (PIN & ANALYZE)
    # --------------------------------------------------------
    elif forest_submode == "🗺️ Interactive Satellite Map (Pin & Analyze)":
        st.subheader("🗺️ Global Satellite Map: Pin & Analyze Any Forest Region")
        st.write(
            """
            Drop a pin anywhere on Earth or pick a global ecological hotspot.
            EcoLens will extract the live high-resolution satellite imagery patch and run AI forest canopy diagnostics.
            """
        )

        PRESET_LOCATIONS = {
            "🌴 Amazon Rainforest (Brazil)": (-3.4653, -62.2159, 15),
            "🌳 Sundarbans Mangrove (India/Bangladesh)": (21.9497, 89.1833, 14),
            "🌲 Congo Basin (DRC)": (0.7054, 24.3164, 15),
            "🌿 Borneo Rainforest (Indonesia)": (-0.5897, 114.5424, 15),
            "🌲 Western Ghats (India)": (14.1670, 75.0500, 15),
            "🌲 Olympic National Forest (USA)": (47.8021, -123.6044, 15),
            "🌲 Black Forest (Germany)": (48.1500, 8.2000, 15),
            "🏜️ Thar Desert / Arid Terrain (India)": (26.9124, 70.9000, 15),
            "📍 Custom GPS Coordinates": (None, None, 15),
        }

        # 1. Search Location or Forest by Name
        col_search, col_s_btn = st.columns([4, 1])
        with col_search:
            search_query = st.text_input(
                "🔍 Search any place, city, or forest by name:",
                placeholder="e.g. Jim Corbett, Yellowstone, Valdivian Rainforest, Daintree Australia, Boreal Taiga...",
                key="map_search_box",
            )
        with col_s_btn:
            st.write("")
            st.write("")
            search_trigger = st.button("📍 Find Place", use_container_width=True)

        if search_trigger and search_query.strip():
            with st.spinner(f"🔍 Locating '{search_query}'..."):
                found = search_place_coordinates(search_query.strip())
                if found:
                    st.session_state.map_lat = found[0]
                    st.session_state.map_lon = found[1]
                    st.session_state.map_place_name = found[2]
                    st.success(f"📍 Found: **{found[2]}**")
                    st.rerun()
                else:
                    st.error(f"Could not find coordinates for '{search_query}'. Try another search term.")

        if "last_preset_choice" not in st.session_state:
            st.session_state.last_preset_choice = "📍 Custom Pin / Free Exploration"

        col_preset, col_zoom = st.columns([2, 1])

        preset_options = ["📍 Custom Pin / Free Exploration"] + [k for k in PRESET_LOCATIONS.keys() if k != "📍 Custom GPS Coordinates"]

        # Determine current dropdown index
        try:
            current_preset_idx = preset_options.index(st.session_state.last_preset_choice)
        except ValueError:
            current_preset_idx = 0

        with col_preset:
            selected_preset = st.selectbox(
                "🌍 Quick-Jump to Forest Hotspot (or click map freely):",
                preset_options,
                index=current_preset_idx,
                key="forest_map_preset_select",
            )

        with col_zoom:
            zoom_level = st.slider(
                "🔍 Satellite Zoom Level",
                13,
                17,
                st.session_state.map_zoom,
                1,
                help="Higher zoom (16-17) shows individual tree canopies; lower zoom (13-14) shows regional forest areas.",
                key="satellite_zoom_slider",
            )
            st.session_state.map_zoom = zoom_level

        # If user changed the preset dropdown explicitly
        if selected_preset != st.session_state.last_preset_choice and selected_preset != "📍 Custom Pin / Free Exploration":
            st.session_state.last_preset_choice = selected_preset
            preset_lat, preset_lon, _ = PRESET_LOCATIONS[selected_preset]
            st.session_state.map_lat = preset_lat
            st.session_state.map_lon = preset_lon
            st.session_state.map_place_name = selected_preset
            st.rerun()

        # Coordinate Inputs with instant sync
        c_lat, c_lon, c_sync = st.columns([2, 2, 1])
        with c_lat:
            in_lat = st.number_input(
                "Latitude (WGS84)",
                value=float(st.session_state.map_lat),
                format="%.5f",
                step=0.005,
                key="map_num_lat",
            )
        with c_lon:
            in_lon = st.number_input(
                "Longitude (WGS84)",
                value=float(st.session_state.map_lon),
                format="%.5f",
                step=0.005,
                key="map_num_lon",
            )
        with c_sync:
            st.write("")
            st.write("")
            if st.button("📍 Set Pin", use_container_width=True):
                st.session_state.map_lat = in_lat
                st.session_state.map_lon = in_lon
                st.session_state.last_preset_choice = "📍 Custom Pin / Free Exploration"
                st.session_state.map_place_name = get_place_name_from_coords(in_lat, in_lon)
                st.rerun()

        # If user typed into number inputs directly
        if round(in_lat, 5) != round(st.session_state.map_lat, 5) or round(in_lon, 5) != round(st.session_state.map_lon, 5):
            st.session_state.map_lat = in_lat
            st.session_state.map_lon = in_lon

        # Build Interactive Folium Satellite Map
        esri_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        esri_attr = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and GIS User Community"

        esri_labels = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        labels_attr = "Labels &copy; Esri &mdash; Boundaries and Places"

        m = folium.Map(
            location=[st.session_state.map_lat, st.session_state.map_lon],
            zoom_start=st.session_state.map_zoom,
            tiles=None,
            control_scale=True,
        )

        # Base Layer 1: High-Res Satellite Imagery
        folium.TileLayer(
            tiles=esri_tiles,
            attr=esri_attr,
            name="🛰️ Satellite Imagery",
            overlay=False,
            control=True,
        ).add_to(m)

        # Base Layer 2: OpenStreetMap Standard
        folium.TileLayer(
            tiles="OpenStreetMap",
            name="🗺️ Streets & Place Names",
            overlay=False,
            control=True,
        ).add_to(m)

        # Overlay Layer: Place Names & Boundaries (Default ON)
        folium.TileLayer(
            tiles=esri_labels,
            attr=labels_attr,
            name="🏷️ Place Names & Borders (Overlay)",
            overlay=True,
            control=True,
            show=True,
        ).add_to(m)

        # Add Ecological Hotspot markers with place labels
        for spot_name, (s_lat, s_lon, _) in PRESET_LOCATIONS.items():
            if s_lat is not None and s_lon is not None:
                folium.CircleMarker(
                    location=[s_lat, s_lon],
                    radius=7,
                    color="#10b981",
                    fill=True,
                    fill_color="#10b981",
                    fill_opacity=0.85,
                    popup=folium.Popup(f"<b>{spot_name}</b><br>Coords: {s_lat:.4f}°, {s_lon:.4f}°", max_width=250),
                    tooltip=spot_name,
                ).add_to(m)

        # Add Active Target Pin Marker at exact coordinates
        folium.Marker(
            [st.session_state.map_lat, st.session_state.map_lon],
            popup=f"<b>📍 Pinned Target Location:</b><br>{st.session_state.map_place_name}<br>Lat: {st.session_state.map_lat:.5f}°, Lon: {st.session_state.map_lon:.5f}°",
            tooltip=f"📍 Pinned: {st.session_state.map_place_name} ({st.session_state.map_lat:.4f}°, {st.session_state.map_lon:.4f}°)",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        ).add_to(m)

        folium.LatLngPopup().add_to(m)
        folium.LayerControl(position="topright").add_to(m)

        st.info(
            f"📍 **Pinned Target:** **{st.session_state.map_place_name}** | "
            f"Latitude: `{st.session_state.map_lat:.5f}°` | Longitude: `{st.session_state.map_lon:.5f}°`"
        )
        st.caption("👇 **Click anywhere on the map to set a new pin at that exact spot.**")

        map_output = st_folium(
            m,
            width="100%",
            height=480,
            returned_objects=["last_clicked"],
            key=f"folium_map_{round(st.session_state.map_lat, 3)}_{round(st.session_state.map_lon, 3)}_{st.session_state.map_zoom}",
        )

        # Detect User Map Click to Move Pin Anywhere
        if map_output and map_output.get("last_clicked"):
            click_lat = map_output["last_clicked"]["lat"]
            click_lon = map_output["last_clicked"]["lng"]
            if (round(click_lat, 4) != round(st.session_state.map_lat, 4) or 
                round(click_lon, 4) != round(st.session_state.map_lon, 4)):
                st.session_state.map_lat = click_lat
                st.session_state.map_lon = click_lon
                st.session_state.last_preset_choice = "📍 Custom Pin / Free Exploration"
                resolved_name = get_place_name_from_coords(click_lat, click_lon)
                st.session_state.map_place_name = resolved_name
                st.rerun()

        st.markdown("---")
        fetch_btn = st.button(
            f"🚀 Analyze Pinned Location: {st.session_state.map_place_name} ({st.session_state.map_lat:.4f}°, {st.session_state.map_lon:.4f}°)",
            type="primary",
            use_container_width=True,
        )

        if fetch_btn:
            with st.spinner(f"🛰️ Retrieving satellite imagery for {st.session_state.map_place_name}..."):
                try:
                    sat_image = get_satellite_patch(st.session_state.map_lat, st.session_state.map_lon, zoom=zoom_level)
                    st.session_state.last_map_image = sat_image
                    st.session_state.last_forest_image = sat_image

                    # 1. Remote Sensing & Canopy Indices
                    veg_result = compute_vegetation_indices(sat_image)
                    st.session_state.last_veg_result = veg_result

                    # 2. Deep Learning Classification
                    result = forest_prediction(sat_image, veg_result)
                    st.session_state.last_map_result = result
                    st.session_state.last_forest_result = result

                    # 3. History Log
                    st.session_state.history.append({
                        "Date": result["timestamp"],
                        "Module": f"Satellite Map ({st.session_state.map_place_name[:25]})",
                        "Result": result["prediction"],
                        "Confidence": result["confidence"],
                    })

                    # 4. Optional Gemini Multimodal Diagnostic
                    gemini_api_key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
                    if gemini_api_key:
                        with st.spinner("🤖 Generating Gemini Multimodal Ecological Assessment..."):
                            try:
                                diag = query_gemini_forest_diagnostics(
                                    sat_image,
                                    result["prediction"],
                                    result["confidence"],
                                    veg_result["canopy_cover_percent"],
                                    gemini_api_key,
                                )
                                st.session_state.last_forest_gemini_diag = diag
                            except Exception as e:
                                st.warning(f"Gemini Diagnostic note: {e}")
                                st.session_state.last_forest_gemini_diag = None
                    else:
                        st.session_state.last_forest_gemini_diag = None

                except Exception as e:
                    st.error(f"Failed to fetch satellite imagery: {e}")

        map_img = st.session_state.get("last_map_image")
        map_res = st.session_state.get("last_map_result")
        veg_result = st.session_state.get("last_veg_result")
        gemini_diag = st.session_state.get("last_forest_gemini_diag")

        if map_img and map_res and veg_result:
            st.markdown("---")
            st.subheader(f"📊 Satellite Intelligence: {st.session_state.map_place_name} ({st.session_state.map_lat:.4f}°, {st.session_state.map_lon:.4f}°)")

            if map_res["prediction"] == "Healthy Forest":
                st.success(
                    f"🌿 **Healthy Forest Cover Confirmed** — {map_res['confidence']:.2f}% confidence "
                    f"(Canopy Coverage: {veg_result['canopy_cover_percent']}%)"
                )
            else:
                st.error(
                    f"⚠️ **Potential Deforestation / Degradation Detected** — {map_res['confidence']:.2f}% confidence "
                    f"(Estimated Bare / Cleared Land: {veg_result['bare_ground_percent']}%)"
                )

            # 4 Key Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Classification", map_res["prediction"], f"{map_res['confidence']:.1f}% conf")
            with m2:
                st.metric("Deforestation Risk", map_res["risk"], f"Severity: {map_res['severity']:.1f}/100")
            with m3:
                st.metric("Canopy Coverage", f"{veg_result['canopy_cover_percent']}%", f"Bare Earth: {veg_result['bare_ground_percent']}%")
            with m4:
                st.metric("Est. Carbon Loss", f"~{veg_result['est_carbon_loss_per_ha']} t/ha", f"Mean GLI: {veg_result['mean_gli']}")

            # Multi-View Explainable AI Tabs
            st.markdown("### 🧠 Multi-View Explainable AI & Remote Sensing")
            tab1, tab2, tab3, tab4 = st.tabs([
                "🛰️ Extracted Satellite Patch",
                "🧠 Grad-CAM Attention Heatmap",
                "🌿 Canopy Segmentation Mask",
                "📊 Vegetation Health Density",
            ])

            with tab1:
                st.image(map_img, caption=f"ESRI High-Resolution Satellite Patch — {st.session_state.map_place_name} ({st.session_state.map_lat:.4f}°, {st.session_state.map_lon:.4f}°)", use_container_width=True)

            with tab2:
                with st.spinner("Rendering Grad-CAM Heatmap..."):
                    heatmap = gradcam(map_img, FOREST_CLASSES.index(map_res["prediction"]))
                if heatmap:
                    st.image(heatmap, caption="Grad-CAM Deep Learning Focus (ResNet-18)", use_container_width=True)
                else:
                    st.info("Grad-CAM overlay not available for this run.")

            with tab3:
                st.image(
                    veg_result["canopy_mask_img"],
                    caption="Canopy Segmentation (Green = Intact Canopy, Red = Bare / Deforested Ground)",
                    use_container_width=True,
                )
                st.caption(
                    f"🟢 Canopy Area: **{veg_result['canopy_cover_percent']}%** | 🔴 Bare Ground / Loss: **{veg_result['bare_ground_percent']}%**"
                )

            with tab4:
                st.image(
                    veg_result["veg_health_img"],
                    caption="Vegetation Health Density Gradient (Green = High Density, Yellow = Sparse, Red = Non-Vegetated)",
                    use_container_width=True,
                )

            # Gemini Ecological Assessment
            if gemini_diag:
                st.markdown("### 🤖 Gemini AI Ecological & Carbon Impact Assessment")
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.write(f"🌲 **Forest Biome:** {gemini_diag.get('forest_biome', 'N/A')}")
                    st.write(f"🔍 **Canopy Integrity:** {gemini_diag.get('canopy_integrity', 'N/A')}")
                    st.write(f"🚜 **Probable Drivers:** {gemini_diag.get('probable_drivers', 'N/A')}")
                with dc2:
                    st.write(f"⚠️ **Biodiversity Threat:** {gemini_diag.get('biodiversity_threat_level', 'N/A')}")
                    st.write(f"💨 **Carbon Impact:** {gemini_diag.get('carbon_risk_assessment', 'N/A')}")
                    st.write(f"🌊 **Key Hazards:** {gemini_diag.get('key_ecological_risks', 'N/A')}")

                st.info(f"💡 **Remediation & Restoration Action Plan:** {gemini_diag.get('remediation_plan', 'N/A')}")
            elif not gemini_api_key:
                st.info("💡 *Enter a Google Gemini API Key in the sidebar to unlock automated ecological biome profiling, driver attribution, and tailored restoration plans.*")

            # PDF Download with GPS coordinates and Place Name
            st.markdown("---")
            coords = (st.session_state.map_lat, st.session_state.map_lon)
            pdf = create_forest_pdf(
                map_img,
                map_res,
                veg_result,
                gemini_diag,
                coordinates=coords,
                place_name=st.session_state.map_place_name,
            )
            safe_name = "".join(c for c in st.session_state.map_place_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
            st.download_button(
                "📄 Download Geotagged Satellite Environmental Report (PDF)",
                data=pdf,
                file_name=f"EcoLens_Report_{safe_name[:25]}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # --------------------------------------------------------
    # WORKFLOW 3: TEMPORAL CHANGE DETECTION (BEFORE VS AFTER)
    # --------------------------------------------------------
    else:
        st.subheader("⏳ Temporal Deforestation & Change Detection")
        st.write(
            """
            Upload two images of the same geographic patch taken at different times (e.g. historical vs recent satellite/drone imagery)
            to compute canopy loss rate, reforestation progress, and visual difference maps.
            """
        )

        tc1, tc2 = st.columns(2)

        with tc1:
            st.markdown("#### 📅 1. Historical Baseline (Before)")
            up_before = st.file_uploader(
                "Upload historical forest image",
                type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
                key="temp_before_upload",
            )

        with tc2:
            st.markdown("#### 📅 2. Recent Observation (After)")
            up_after = st.file_uploader(
                "Upload recent forest image",
                type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
                key="temp_after_upload",
            )

        if up_before and up_after:
            img_before = Image.open(up_before).convert("RGB")
            img_after = Image.open(up_after).convert("RGB")

            col_b_preview, col_a_preview = st.columns(2)
            with col_b_preview:
                st.image(img_before, caption="Historical Baseline", use_container_width=True)
            with col_a_preview:
                st.image(img_after, caption="Recent Observation", use_container_width=True)

            if st.button("🔍 Run Temporal Change Detection", type="primary", use_container_width=True):
                with st.spinner("Computing optical canopy delta & difference overlay..."):
                    temporal_res = compute_temporal_change(img_before, img_after)
                    st.session_state.last_temporal_result = temporal_res

            temp_res = st.session_state.get("last_temporal_result")

            if temp_res:
                st.markdown("---")
                st.subheader("📊 Temporal Change Analytics")

                tm1, tm2, tm3, tm4 = st.columns(4)

                with tm1:
                    st.metric("Historical Canopy", f"{temp_res['canopy_before']}%")
                with tm2:
                    st.metric("Recent Canopy", f"{temp_res['canopy_after']}%")
                with tm3:
                    st.metric(
                        "Canopy Loss Area",
                        f"{temp_res['loss_percent']}%",
                        delta=f"-{temp_res['loss_percent']}%",
                        delta_color="inverse",
                    )
                with tm4:
                    st.metric(
                        "Net Canopy Change (Δ)",
                        f"{temp_res['net_change']}%",
                        delta=f"{temp_res['net_change']}%",
                    )

                if temp_res['loss_percent'] > 5.0:
                    st.error(
                        f"🚨 **Significant Deforestation Detected:** {temp_res['loss_percent']}% of the original canopy was cleared between the two observation dates."
                    )
                elif temp_res['gain_percent'] > 5.0:
                    st.success(
                        f"🌱 **Positive Reforestation Detected:** {temp_res['gain_percent']}% new canopy growth observed."
                    )
                else:
                    st.info("⚖️ **Stable Canopy Cover:** Minimal structural change observed between both observation periods.")

                st.markdown("### 🗺️ Deforestation & Regrowth Difference Map")
                st.image(
                    temp_res["change_overlay_img"],
                    caption="🔴 Crimson = Deforested Canopy Loss | 🟢 Cyan/Green = Regenerated Canopy",
                    use_container_width=True,
                )


# ============================================================
# PLANT SCANNER
# ============================================================

elif mode == "🌿 Plant Scanner":

    st.header(
        "🌿 AI Plant Scanner"
    )

    st.write(
        """
        Use the camera to scan a plant, leaf, flower,
        fruit or tree.
        """
    )

    image_file = st.camera_input(
        "📷 Scan Plant"
    )

    if image_file is None:

        uploaded = st.file_uploader(
            "Or upload a plant image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
        )

        image_file = uploaded

    if image_file:

        image = Image.open(
            image_file
        ).convert(
            "RGB"
        )

        st.image(
            image,
            caption="Scanned Plant",
            use_container_width=True,
        )

        if st.button(
            "🔍 Identify Plant",
            type="primary",
        ):
            result = identify_species_demo(
                image,
                "plant",
            )
            st.session_state.last_plant_result = result

        plant_res = st.session_state.get("last_plant_result")

        if plant_res and plant_res.get("status") == "identified":
            # Use AI-sorted candidates (detected species first, then others)
            species_candidates = plant_res.get("candidates", [])
            if not species_candidates:
                species_candidates = [
                    name for name, data in st.session_state.species_database.items()
                    if "plant" in data.get("type", "").lower()
                ]

            detected = plant_res["species"]
            if detected not in species_candidates:
                species_candidates.insert(0, detected)
            elif species_candidates[0] != detected:
                species_candidates.remove(detected)
                species_candidates.insert(0, detected)

            conf = plant_res.get('confidence', 50)
            st.success(
                f"🌿 AI Detection: **{detected}** ({conf}% confidence)"
            )
            if conf < 70:
                st.warning(
                    "⚠️ Low confidence — please refine the selection below or try a clearer photo."
                )

            selected_plant = st.selectbox(
                "🌿 Select or Refine Plant Species:",
                species_candidates,
                index=0,
                key="plant_species_select",
            )

            st.session_state.last_species = selected_plant
            st.session_state.last_species_type = "Plant"

            show_species_information(selected_plant)


# ============================================================
# ANIMAL SCANNER
# ============================================================

elif mode == "🐅 Animal Scanner":

    st.header(
        "🐅 AI Animal Scanner"
    )

    st.write(
        """
        Capture wildlife using the camera and identify
        the species using a dedicated wildlife model/API.
        """
    )

    image_file = st.camera_input(
        "📷 Scan Animal"
    )

    if image_file is None:

        uploaded = st.file_uploader(
            "Or upload animal image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
        )

        image_file = uploaded

    if image_file:

        image = Image.open(
            image_file
        ).convert(
            "RGB"
        )

        st.image(
            image,
            caption="Scanned Animal",
            use_container_width=True,
        )

        if st.button(
            "🔍 Identify Animal",
            type="primary",
        ):
            result = identify_species_demo(
                image,
                "animal",
            )
            st.session_state.last_animal_result = result

        anim_res = st.session_state.get("last_animal_result")

        if anim_res and anim_res.get("status") == "identified":
            # Use AI-sorted candidates (detected species first, then others)
            animal_candidates = anim_res.get("candidates", [])
            if not animal_candidates:
                animal_candidates = [
                    name for name, data in st.session_state.species_database.items()
                    if "animal" in data.get("type", "").lower()
                ]

            detected_anim = anim_res["species"]
            if detected_anim not in animal_candidates:
                animal_candidates.insert(0, detected_anim)
            elif animal_candidates[0] != detected_anim:
                animal_candidates.remove(detected_anim)
                animal_candidates.insert(0, detected_anim)

            conf = anim_res.get('confidence', 50)
            st.success(
                f"🐅 AI Detection: **{detected_anim}** ({conf}% confidence)"
            )
            if conf < 70:
                st.warning(
                    "⚠️ Low confidence — please refine the selection below or try a clearer photo."
                )

            selected_animal = st.selectbox(
                "🐅 Select or Refine Animal Species:",
                animal_candidates,
                index=0,
                key="animal_species_select",
            )

            st.session_state.last_species = selected_animal
            st.session_state.last_species_type = "Animal"

            show_species_information(selected_animal)


# ============================================================
# VOICE ASSISTANT
# ============================================================

elif mode == "🎙️ AI Voice Assistant":

    st.header("🎙️ EcoLens AI Voice Assistant")
    st.caption("Real-time voice intelligence & conversational ecological assistant powered by Google Gemini.")

    # 1. Subject Context Card
    col_ctx1, col_ctx2 = st.columns([3, 2])
    with col_ctx1:
        if st.session_state.last_species:
            st.success(f"🌿 **Active Scanned Subject:** **{st.session_state.last_species}** ({st.session_state.get('last_species_type', 'Species')})")
        else:
            st.info("🌍 **Global Environmental Mode:** No specific species scanned yet. You can ask general forest, climate, or biodiversity questions!")

    with col_ctx2:
        all_db_species = list(st.session_state.species_database.keys())
        switch_options = ["(None / Free Exploration)"] + all_db_species
        current_idx = switch_options.index(st.session_state.last_species) if st.session_state.last_species in switch_options else 0
        
        switch_sub = st.selectbox(
            "🎯 Switch Subject Context:",
            switch_options,
            index=current_idx,
            key="voice_subject_switcher",
        )
        if switch_sub != "(None / Free Exploration)" and switch_sub != st.session_state.last_species:
            st.session_state.last_species = switch_sub
            db_type = st.session_state.species_database.get(switch_sub, {}).get("type", "Species")
            st.session_state.last_species_type = db_type
            st.rerun()
        elif switch_sub == "(None / Free Exploration)" and st.session_state.last_species is not None:
            st.session_state.last_species = None
            st.session_state.last_species_type = None
            st.rerun()

    # 2. Voice Audio Settings & Controls
    with st.expander("⚙️ Voice Engine & Audio Controls", expanded=False):
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            language_options = {
                "🇮🇳 English (India)": {"accent": "co.in", "lang": "en-IN"},
                "🇮🇳 Hindi (हिन्दी)": {"accent": "hi", "lang": "hi-IN"},
                "🇮🇳 Marathi (मराठी)": {"accent": "mr", "lang": "mr-IN"},
                "🇺🇸 English (US)": {"accent": "com", "lang": "en-US"},
                "🇬🇧 English (UK)": {"accent": "co.uk", "lang": "en-GB"},
                "🇦🇺 English (Australia)": {"accent": "com.au", "lang": "en-AU"},
                "🇨🇦 English (Canada)": {"accent": "ca", "lang": "en-CA"},
            }
            selected_lang_label = st.selectbox(
                "🌐 Language & Voice:",
                list(language_options.keys()),
                index=0,
                key="voice_language_select",
                help="Sets both speech recognition (mic) and text-to-speech (voice output) language.",
            )
            selected_cfg = language_options[selected_lang_label]
            st.session_state.voice_accent = selected_cfg["accent"]
            st.session_state.voice_language = selected_cfg["lang"]
        with vc2:
            st.session_state.auto_speech = st.checkbox(
                "🔊 Auto-Generate Voice Audio (TTS)",
                value=st.session_state.auto_speech,
                help="Automatically generates high-fidelity spoken audio responses for questions.",
            )
        with vc3:
            st.write("")
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.voice_chat_history = []
                st.rerun()

    # Resolve current mic language for the JS speech recognition component
    current_mic_lang = st.session_state.voice_language  # e.g. "en-IN", "hi-IN", "mr-IN"

    # 3. Live Microphone Widget
    live_mic_component = f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 14px; padding: 14px; text-align: center; color: white; border: 1px solid #334155; margin-bottom: 10px;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 8px;">
            <button id="liveMicBtn" onclick="toggleSpeech()" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border: none; color: white; width: 56px; height: 56px; border-radius: 50%; font-size: 24px; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 16px rgba(16, 185, 129, 0.4); transition: all 0.25s ease;">
                🎙️
            </button>
        </div>
        <div id="micStatus" style="font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">Tap microphone to speak live</div>
        <div id="micTranscript" style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; font-size: 14px; color: #38bdf8; min-height: 36px; font-style: italic;">
            Waiting for live voice input...
        </div>
    </div>

    <script>
        let recognition = null;
        let isListening = false;

        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = '{current_mic_lang}';

            recognition.onstart = function() {{
                isListening = true;
                const btn = document.getElementById('liveMicBtn');
                btn.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
                btn.style.boxShadow = '0 0 22px rgba(239, 68, 68, 0.8)';
                btn.style.transform = 'scale(1.06)';
                document.getElementById('micStatus').innerHTML = '🔴 Listening live... Speak now!';
                document.getElementById('micStatus').style.color = '#ef4444';
            }};

            recognition.onresult = function(event) {{
                let final = '', interim = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {{
                    if (event.results[i].isFinal) {{
                        final += event.results[i][0].transcript;
                    }} else {{
                        interim += event.results[i][0].transcript;
                    }}
                }}
                const text = final || interim;
                if (text) {{
                    document.getElementById('micTranscript').innerHTML = '🗣️ \"' + text + '\"';
                }}
                if (final) {{
                    if (navigator.clipboard) {{
                        navigator.clipboard.writeText(final);
                    }}
                    document.getElementById('micStatus').innerHTML = '✅ Captured (copied to clipboard — paste into chat or press Enter)';
                    document.getElementById('micStatus').style.color = '#10b981';
                }}
            }};

            recognition.onerror = function(event) {{
                document.getElementById('micStatus').innerHTML = '⚠️ Note: ' + event.error;
                document.getElementById('micStatus').style.color = '#f59e0b';
                resetBtn();
            }};

            recognition.onend = function() {{
                resetBtn();
            }};
        }} else {{
            document.getElementById('micStatus').innerHTML = 'Live Web Speech is available in Chrome, Edge, and Safari.';
        }}

        function resetBtn() {{
            isListening = false;
            const btn = document.getElementById('liveMicBtn');
            btn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
            btn.style.boxShadow = '0 0 16px rgba(16, 185, 129, 0.4)';
            btn.style.transform = 'scale(1)';
        }}

        function toggleSpeech() {{
            if (!recognition) {{
                alert('Live Web Speech recognition is supported in Chrome, Edge, or Safari.');
                return;
            }}
            if (isListening) {{
                recognition.stop();
            }} else {{
                try {{
                    recognition.start();
                }} catch(e) {{
                    recognition.stop();
                }}
            }}
        }}
    </script>
    """
    components.html(live_mic_component, height=160)

    # Quick Inquiries Chips
    active_sp = st.session_state.get("last_species")
    if not active_sp or active_sp == "(None / Free Exploration)":
        active_sp = "Bengal Tiger"
        st.session_state.last_species = "Bengal Tiger"
        st.session_state.last_species_type = "Animal"

    st.markdown("##### 💡 Quick Questions")
    chip_cols = st.columns(6)
    suggested_queries = [
        ("⚠️ Status", f"What is the conservation status, native region, estimated wild population, and primary threats for {active_sp}?"),
        ("🍽️ Diet", f"What is the diet, food sources, and feeding behavior of {active_sp}?"),
        ("🗺️ Habitat", f"Where is the natural habitat, native biome, and geographic range of {active_sp}?"),
        ("🌿 Role", f"What is the ecological role and function of {active_sp} in its ecosystem?"),
        ("💡 Facts", f"Tell me a unique, fascinating biological fact about {active_sp}."),
        ("🌲 Forest", f"How is {active_sp} or its ecosystem affected by deforestation, canopy loss, and climate change?"),
    ]
    
    selected_chip_query = None
    for i, (chip_label, chip_prompt) in enumerate(suggested_queries):
        with chip_cols[i]:
            if st.button(chip_label, key=f"chip_btn_{i}_{active_sp}", use_container_width=True):
                selected_chip_query = chip_prompt

    chat_prompt = st.chat_input("💬 Ask EcoLens any question about species, forests, or ecology...")

    active_question = selected_chip_query or chat_prompt

    if active_question:
        st.session_state.voice_chat_history.append({
            "role": "user",
            "text": active_question,
            "timestamp": datetime.now().strftime("%H:%M"),
        })

        with st.spinner("🤖 EcoLens is thinking..."):
            ai_answer = answer_local_question(
                active_question,
                species_name=active_sp,
                chat_history=st.session_state.voice_chat_history,
            )

        st.session_state.voice_chat_history.append({
            "role": "assistant",
            "text": ai_answer,
            "timestamp": datetime.now().strftime("%H:%M"),
        })
        st.rerun()

    # Conversational Dialogue
    st.markdown("---")
    st.markdown("### 💬 Dialogue Stream")

    if not st.session_state.voice_chat_history:
        st.info(
            "👋 **EcoLens AI Voice Assistant Ready!**\n\n"
            "- Tap the microphone above or type in the chat box below.\n"
            "- Use the Quick Question buttons for immediate answers with audible speech controls!"
        )
    else:
        for idx, msg in enumerate(st.session_state.voice_chat_history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["text"])
                    st.caption(f"🕒 {msg.get('timestamp', '')}")
            else:
                with st.chat_message("assistant", avatar="🌲"):
                    st.markdown(msg["text"])
                    st.caption(f"🕒 {msg.get('timestamp', '')}")
                    
                    is_latest = (idx == len(st.session_state.voice_chat_history) - 1)
                    render_speech_controls(
                        msg["text"],
                        uid=f"chat_speech_{idx}",
                        accent_tld=st.session_state.voice_accent,
                        auto_play=(is_latest and st.session_state.auto_speech),
                        voice_lang_override=st.session_state.voice_language,
                    )


# ============================================================
# SPECIES DATABASE
# ============================================================

elif mode == "📚 Species Database":

    st.header(
        "📚 EcoLens Species Knowledge Base"
    )

    search = st.text_input(
        "🔎 Search species"
    )

    species_names = list(
        st.session_state.species_database.keys()
    )

    if search:

        species_names = [
            x
            for x in species_names
            if search.lower()
            in x.lower()
        ]

    if not species_names:

        st.warning(
            "No species found."
        )

    else:

        selected = st.selectbox(
            "Select species",
            species_names,
        )

        show_species_information(
            selected
        )


# ============================================================
# HISTORY
# ============================================================

elif mode == "📊 Biodiversity History":

    st.header(
        "📊 EcoLens Analysis History"
    )

    if not st.session_state.history:

        st.info(
            "No analyses performed yet."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        csv = df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "📥 Download History",
            data=csv,
            file_name="EcoLens_History.csv",
            mime="text/csv",
        )


# ============================================================
# SYSTEM
# ============================================================

elif mode == "⚙️ System":

    st.header(
        "⚙️ EcoLens System Information"
    )

    c1, c2 = st.columns(
        2
    )

    with c1:

        st.subheader(
            "🤖 Forest AI"
        )

        st.write(
            "**Architecture:** ResNet18"
        )

        st.write(
            "**Classifier:** 512 → 128 → 2"
        )

        st.write(
            f"**Weights:** {MODEL_PATH}"
        )

        st.write(
            f"**Device:** {DEVICE}"
        )

        st.write(
            "**Classes:** Healthy Forest / "
            "Deforested Area"
        )

        if MODEL_LOADED:

            st.success(
                "Model Loaded Successfully ✅"
            )

        else:

            st.error(
                MODEL_ERROR
            )

    with c2:

        st.subheader(
            "🌍 Platform Modules"
        )

        st.write(
            "✅ Forest classification"
        )

        st.write(
            "✅ Camera scanning"
        )

        st.write(
            "✅ Plant scanner interface"
        )

        st.write(
            "✅ Animal scanner interface"
        )

        st.write(
            "✅ Species knowledge base"
        )

        st.write(
            "✅ Voice assistant interface"
        )

        st.write(
            "✅ Grad-CAM explainability"
        )

        st.write(
            "✅ PDF reports"
        )

    st.markdown(
        "---"
    )

    st.warning(
        """
        SCIENTIFIC LIMITATION

        The uploaded model is a forest-condition
        classification model. Its confidence indicates
        classification confidence; it does not indicate
        the percentage of geographic area that is
        deforested.

        Plant and animal identification require dedicated
        biodiversity models or APIs.

        Grad-CAM is an interpretability visualization and
        should not be treated as an exact deforestation
        boundary.

        Environmental decisions should be validated using
        appropriate satellite, GIS, historical and/or
        field data.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌲 <b>EcoLens 2.0</b><br>
        AI Forest & Biodiversity Intelligence Platform<br><br>
        Forest Monitoring • Plant Identification •
        Wildlife Monitoring • Explainable AI •
        Voice Assistance
    </div>
    """,
    unsafe_allow_html=True,
)