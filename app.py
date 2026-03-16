import streamlit as st
import pandas as pd
from ultralytics import YOLO  # NEU: YOLO statt TensorFlow
from PIL import Image
import numpy as np
import io
from supabase import create_client, Client
import datetime

# --- KONFIGURATION ---
SUPABASE_URL = "https://xbvffylpvjsdmjvwjaej.supabase.co"
SUPABASE_KEY = "DEIN_SUPABASE_KEY" 

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="KI-Fundbüro Black", page_icon="🔍", layout="wide")

# --- DEEP BLACK & WHITE TEXT CSS (Unverändert) ---
st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] { border-right: 1px solid #333333; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: #000000 !important; }
    .stTabs [data-baseweb="tab"] { color: #888888 !important; }
    .stTabs [aria-selected="true"] p { color: #FFFFFF !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #080808 !important;
        border: 1px solid #333333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNKTIONEN ---
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_yolo_model():
    # Lädt das YOLOv8 Nano Modell (schnell und leicht)
    # Du kannst auch 'best.pt' nutzen, falls du ein eigenes trainiert hast
    model = YOLO('yolov8n.pt') 
    return model

def classify_with_yolo(image, model):
    # YOLO braucht keine manuelle Skalierung auf 224x224
    results = model(image)
    
    # Wir nehmen das erste erkannte Objekt mit der höchsten Konfidenz
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        class_id = int(box.cls[0])
        name = model.names[class_id]
        conf = float(box.conf[0])
        return name, conf
    else:
        return "Unbekannt", 0.0

# --- APP LOGIK ---
def main():
    st.title("🔍 KI-Fundbüro")
    supabase = init_supabase()
    model = load_yolo_model()

    tab1, tab2 = st.tabs(["📤 Fund melden", "🔎 Suchen"])

    with tab1:
        st.header("Neues Fundstück erfassen")
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, width=350)
            
            if st.button("🔍 KI-Analyse starten", type="primary"):
                name, conf = classify_with_yolo(img, model)
                st.session_state.detected = {"name": name, "conf": conf, "img": img}
            
            if 'detected' in st.session_state:
                st.success(f"Erkannt: {st.session_state.detected['name']} ({st.session_state.detected['conf']:.1%})")
                
                with st.form("save_form"):
                    desc = st.text_area("Beschreibung (Farbe, Zustand, Marke)")
                    tags = st.text_input("Tags / Merkmale", placeholder="z.B. blau, verkratzt, Leder")
                    loc = st.text_input("Fundort")
                    
                    if st.form_submit_button("📦 In Datenbank speichern"):
                        try:
                            # Bild-Vorbereitung
                            img_io = io.BytesIO()
                            st.session_state.detected['img'].save(img_io, format='PNG')
                            file_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            
                            # Supabase Storage Upload
                            supabase.storage.from_("fundstuecke").upload(
                                path=file_name,
                                file=img_io.getvalue(),
                                file_options={"content-type": "image/png"}
                            )
                            
                            image_url = supabase.storage.from_("fundstuecke").get_public_url(file_name)

                            # DB Eintrag
                            data = {
                                "class_name": st.session_state.detected['name'],
                                "description": desc,
                                "location": loc,
                                "tags": tags.lower(),
                                "image_url": image_url,
                                "status": "gemeldet"
                            }
                            supabase.table("fundstuecke").insert(data).execute()
                            
                            st.balloons()
                            st.success("Erfolgreich gespeichert!")
                            del st.session_state.detected
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fehler: {e}")

    with tab2:
        st.header("Fundstücke durchsuchen")
        search_query = st.text_input("Suche nach Kategorien, Farben oder Tags", placeholder="z.B. rot...")
        
        try:
            res = supabase.table("fundstuecke").select("*").order("created_at", desc=True).execute()
            items = res.data
        except:
            items = []

        if search_query:
            q = search_query.lower()
            items = [i for i in items if q in (i.get('description') or "").lower() 
                     or q in (i.get('tags') or "").lower() 
                     or q in (i.get('class_name') or "").lower()]

        if items:
            cols = st.columns(3)
            for idx, item in enumerate(items):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.image(item['image_url'])
                        st.subheader(item['class_name'])
                        if item.get('tags'):
                            st.markdown(f"🏷️ `{item['tags']}`")
                        st.write(f"**Beschreibung:** {item['description']}")
                        st.write(f"**Ort:** {item['location']}")
                        st.caption(f"Datum: {item.get('created_at', '')[:10]}")
        else:
            st.info("Keine Einträge gefunden.")

if __name__ == "__main__":
    main()
