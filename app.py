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
SUPABASE_KEY = "DEIN_KEY_HIER_EINTRAGEN" 

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Fundbüro - Deep Black", page_icon="🔍", layout="wide")

# --- DEEP BLACK CSS ---
# Dies erzwingt, dass alles Weiße schwarz wird und die Sidebar ebenfalls schwarz ist.
st.markdown("""
    <style>
    /* Hintergrund der gesamten App */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* Die linke Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
        border-right: 1px solid #333333;
    }

    /* Texte in Weiß erzwingen */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #FFFFFF !important;
    }

    /* Eingabefelder abdunkeln */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
    }

    /* Tabs-Leiste */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888888 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #FFFFFF !important;
    }

    /* Container für Fundstücke */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0c0c0c !important;
        border: 1px solid #333333 !important;
        border-radius: 10px;
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

# --- MAIN APP ---
def main():
    st.title("🔍 KI-Fundbüro (Deep Black Edition)")
    supabase = init_supabase()
    model, class_names = load_keras_model()

    if model is None:
        st.error("Modell-Dateien fehlen!")
        return

    tab1, tab2 = st.tabs(["📤 Fund melden", "🔎 Suchen"])

    with tab1:
        uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, width=300)
            
            if st.button("🔍 Analysieren", type="primary"):
                name, conf = prepare_and_classify(img, model, class_names)
                st.session_state.detected = {"name": name, "conf": conf, "img": img}
            
            if 'detected' in st.session_state:
                st.success(f"Erkannt: {st.session_state.detected['name']}")
                
                with st.form("save_form"):
                    desc = st.text_area("Beschreibung")
                    tags = st.text_input("Tags (z.B. blau, metall, verkratzt)")
                    loc = st.text_input("Fundort")
                    
                    if st.form_submit_button("📦 In Datenbank speichern"):
                        # BILD UPLOAD & DB INSERT
                        img_byte_arr = io.BytesIO()
                        st.session_state.detected['img'].save(img_byte_arr, format='PNG')
                        
                        path = f"fundstuecke/{datetime.datetime.now().timestamp()}.png"
                        supabase.storage.from_("fundbuero-bilder").upload(path, img_byte_arr.getvalue())
                        url = f"{SUPABASE_URL}/storage/v1/object/public/fundbuero-bilder/{path}"

                        # HIER lag der Fehler: 'tags' muss in Supabase existieren!
                        data = {
                            "class_name": st.session_state.detected['name'],
                            "confidence_score": st.session_state.detected['conf'],
                            "description": desc,
                            "location": loc,
                            "tags": tags.lower(),
                            "image_url": url,
                            "status": "gemeldet"
                        }
                        supabase.table("fundstuecke").insert(data).execute()
                        st.balloons()
                        st.success("Erfolgreich gespeichert!")
                        del st.session_state.detected

    with tab2:
        search = st.text_input("Suche nach Merkmalen oder Tags", placeholder="z.B. rot...")
        
        # Alle Daten holen
        query = supabase.table("fundstuecke").select("*").execute()
        items = query.data
        
        # Filterung
        if search:
            s = search.lower()
            items = [i for i in items if s in (i.get('description') or "").lower() or s in (i.get('tags') or "").lower()]

        # Anzeige
        cols = st.columns(3)
        for i, item in enumerate(items):
            with cols[i % 3]:
                with st.container(border=True):
                    st.image(item['image_url'])
                    st.subheader(item['class_name'])
                    if item.get('tags'):
                        st.markdown(f"🏷️ `{item['tags']}`")
                    st.write(f"📍 {item['location']}")

if __name__ == "__main__":
    main()
