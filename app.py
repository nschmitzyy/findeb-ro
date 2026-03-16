import streamlit as st
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
from supabase import create_client, Client
import datetime
import os

# --- KONFIGURATION ---
SUPABASE_URL = "https://xbvffylpvjsdmjvwjaej.supabase.co"
SUPABASE_KEY = "DEIN_KEY" # Den Key aus Sicherheitsgründen hier weggelassen

# --- SEITENKONFIGURATION & THEME ---
st.set_page_config(page_title="Fundbüro - KI-Erkennung", page_icon="🔍", layout="wide")

# Dark Mode Logik via Sidebar
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

with st.sidebar:
    st.title("Einstellungen")
    st.button(f"Umschalten auf {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode", on_click=toggle_theme)

# Custom CSS für den Dark Mode (Streamlit hat ein eingebautes Theme, aber das hier erzwingt Anpassungen)
if st.session_state.theme == 'dark':
    st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: white; }
        .stButton>button { border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)

# --- FUNKTIONEN ---

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_keras_model():
    try:
        model = load_model("keras_model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None, None

def prepare_and_classify(image, model, class_names):
    try:
        image = image.convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        if ': ' in class_name: class_name = class_name.split(': ')[1]
        return class_name, float(prediction[0][index]), index
    except Exception as e:
        return None, None, None

def save_to_supabase(supabase, image, class_name, confidence_score, description, location, finder_name, tags):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"fundstuecke/{timestamp}_{class_name}.png"
        
        supabase.storage.from_("fundbuero-bilder").upload(file_name, img_byte_arr)
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/fundbuero-bilder/{file_name}"

        data = {
            "class_name": class_name,
            "confidence_score": confidence_score,
            "description": description,
            "location": location,
            "finder_name": finder_name,
            "tags": tags, # Speichere Tags als String oder Array
            "image_url": image_url,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "gemeldet"
        }
        result = supabase.table("fundstuecke").insert(data).execute()
        return True, result
    except Exception as e:
        st.error(f"Speicherfehler: {e}")
        return False, None

def get_fundstuecke(supabase, filter_class=None, search_term=None):
    try:
        query = supabase.table("fundstuecke").select("*").order("created_at", desc=True)
        if filter_class and filter_class != "Alle":
            query = query.eq("class_name", filter_class)
        
        result = query.execute()
        data = result.data
        
        # Manuelle Filterung für Beschreibung UND Tags (da Supabase OR-Filterung komplexer ist)
        if search_term:
            s = search_term.lower()
            data = [d for d in data if s in (d.get('description') or "").lower() or s in (d.get('tags') or "").lower()]
            
        return data
    except Exception as e:
        st.error(f"Abruffehler: {e}")
        return []

# --- HAUPT APP ---
def main():
    st.title("🔍 KI-Fundbüro")
    supabase = init_supabase()
    model, class_names = load_keras_model()

    if model is None: return

    available_classes = ["Alle"] + [line.split(': ')[1].strip() if ': ' in line else line.strip() for line in class_names]

    tab1, tab2 = st.tabs(["📤 Fundstück melden",
