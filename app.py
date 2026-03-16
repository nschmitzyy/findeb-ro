import streamlit as st
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
from supabase import create_client, Client
import datetime

# --- KONFIGURATION ---
SUPABASE_URL = "https://xbvffylpvjsdmjvwjaej.supabase.co"
SUPABASE_KEY = "DEIN_KEY" 

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Fundbüro - Deep Black", page_icon="🔍", layout="wide")

# --- DARK MODE LOGIK & CSS ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark' # Standardmäßig Dark

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# Erzwinge Farben per CSS
if st.session_state.theme == 'dark':
    st.markdown("""
        <style>
        /* Hauptbereich */
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        /* Sidebar (Leiste links) */
        [data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #333333;
        }
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        /* Eingabefelder und Boxen */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>select {
            background-color: #1A1A1A;
            color: white;
            border: 1px solid #444;
        }
        /* Karten/Container im Grid */
        div[data-testid="stVerticalBlock"] > div[style*="border"] {
            background-color: #111111 !important;
            border: 1px solid #333 !important;
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #000000;
        }
        .stTabs [data-baseweb="tab"] {
            color: white !important;
        }
        h1, h2, h3, p {
            color: #ffffff !important;
        }
        </style>
        """, unsafe_allow_html=True)

# --- SIDEBAR EINSTELLUNGEN ---
with st.sidebar:
    st.title("Settings")
    st.button(f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode", on_click=toggle_theme)
    st.markdown("---")
    st.info("Hier kannst du Tags verwalten oder den Status filtern.")

# --- DATENBANK FUNKTIONEN (Gekürzt für Übersicht) ---
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ... (Hier die Funktionen prepare_and_classify wie im Original lassen) ...

def save_to_supabase(supabase, image, class_name, confidence_score, description, location, finder_name, tags):
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"fundstuecke/{timestamp}_{class_name}.png"
        
        supabase.storage.from_("fundbuero-bilder").upload(file_name, img_byte_arr)
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/fundbuero-bilder/{file_name}"

        # Tags werden hier als String gespeichert (z.B. "blau, metall, verkratzt")
        data = {
            "class_name": class_name,
            "confidence_score": confidence_score,
            "description": description,
            "location": location,
            "finder_name": finder_name,
            "tags": tags.lower(), 
            "image_url": image_url,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "gemeldet"
        }
        supabase.table("fundstuecke").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Fehler: {e}")
        return False

# --- UI LOGIK ---
def main():
    st.title("🔍 KI-Fundbüro (Pro)")
    supabase = init_supabase()
    # Modell-Laden hier einfügen...

    tab1, tab2 = st.tabs(["📤 Melden", "🔎 Suchen"])

    with tab1:
        st.subheader("Gegenstand beschreiben")
        # Bild-Upload Logik...
        tags_input = st.text_input("Tags / Merkmale", placeholder="Farbe, Material, Zustand (mit Komma trennen)")
        # Speichern Button Logik...

    with tab2:
        st.subheader("Datenbank durchsuchen")
        search_query = st.text_input("Suche nach Beschreibung oder Tags (z.B. 'schwarz' oder 'Leder')")
        
        # Abruf-Logik
        res = supabase.table("fundstuecke").select("*").execute()
        items = res.data
        
        if search_query:
            # Filtert sowohl in der Beschreibung als auch in den Tags
            items = [i for i in items if search_query.lower() in (i.get('description') or "").lower() 
                     or search_query.lower() in (i.get('tags') or "").lower()]

        # Anzeige der Items im Grid...
        for item in items:
            with st.container(border=True):
                st.write(f"### {item['class_name']}")
                if item.get('tags'):
                    # Zeigt die Tags als kleine Badges an
                    st.markdown(f"🏷️ `{item['tags'].replace(',', '` `')}`")
                st.image(item['image_url'])

if __name__ == "__main__":
    main()
