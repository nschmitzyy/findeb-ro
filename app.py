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
SUPABASE_KEY = "DEIN_SUPABASE_KEY" # Bitte hier deinen Key einfügen

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="KI-Fundbüro Black", page_icon="🔍", layout="wide")

# --- DEEP BLACK & WHITE TEXT CSS ---
st.markdown("""
    <style>
    /* Hintergrund & Sidebar auf Tiefschwarz */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
    }
    
    /* ALLE Texte auf Reinweiß */
    h1, h2, h3, h4, h5, h6, p, span, label, li, .stMarkdown, [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
    }

    /* Sidebar Trennlinie */
    [data-testid="stSidebar"] {
        border-right: 1px solid #333333;
    }

    /* Eingabefelder abdunkeln */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
    }

    /* Tabs (Reiter) */
    .stTabs [data-baseweb="tab-list"] { background-color: #000000 !important; }
    .stTabs [data-baseweb="tab"] { color: #888888 !important; }
    .stTabs [aria-selected="true"] p { color: #FFFFFF !important; }

    /* Container Boxen */
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
def load_keras_model():
    try:
        model = load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r") as f:
            class_names = f.readlines()
        return model, class_names
    except:
        return None, None

def prepare_and_classify(image, model, class_names):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = (np.asarray(image).astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    prediction = model.predict(data, verbose=0)
    idx = np.argmax(prediction)
    name = class_names[idx].strip()
    if ': ' in name: name = name.split(': ')[1]
    return name, float(prediction[0][idx])

# --- APP LOGIK ---
def main():
    st.title("🔍 KI-Fundbüro")
    supabase = init_supabase()
    model, class_names = load_keras_model()

    if model is None:
        st.error("Modell-Dateien nicht gefunden!")
        return

    tab1, tab2 = st.tabs(["📤 Fund melden", "🔎 Suchen"])

    with tab1:
        st.header("Neues Fundstück erfassen")
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, width=350)
            
            if st.button("🔍 KI-Analyse starten", type="primary"):
                name, conf = prepare_and_classify(img, model, class_names)
                st.session_state.detected = {"name": name, "conf": conf, "img": img}
            
            if 'detected' in st.session_state:
                st.success(f"Erkannt: {st.session_state.detected['name']} ({st.session_state.detected['conf']:.1%})")
                
                with st.form("save_form"):
                    desc = st.text_area("Beschreibung (Farbe, Zustand, Marke)")
                    tags = st.text_input("Tags / Merkmale", placeholder="z.B. blau, verkratzt, Leder")
                    loc = st.text_input("Fundort")
                    
                    if st.form_submit_button("📦 In Datenbank speichern"):
                        try:
                            # Bild-Upload
                            img_io = io.BytesIO()
                            st.session_state.detected['img'].save(img_io, format='PNG')
                            file_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            
                            supabase.storage.from_("fundstuecke").upload(
                                path=file_name,
                                file=img_io.getvalue(),
                                file_options={"content-type": "image/png"}
                            )
                            
                            # URL generieren
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
