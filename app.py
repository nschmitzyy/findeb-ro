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

# Seitenkonfiguration
st.set_page_config(
    page_title="Fundbüro - KI-Erkennung",
    page_icon="🔍",
    layout="wide"
)

# Supabase-Konfiguration (ersetze mit deinen Zugangsdaten)
SUPABASE_URL = "https://imntylvenimvnmocbtzy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltbnR5bHZlbmltdm5tb2NidHp5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMwNTk4NzcsImV4cCI6MjA4ODYzNTg3N30.48pIBqUdlqXTooorJXHm71icVSj1wdTwW4tg5m2ovns"

# Initialisiere Supabase Client
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Lade das KI-Modell
@st.cache_resource
def load_keras_model():
    try:
        model = load_model("keras_model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None, None

# Bild vorbereiten und klassifizieren
def prepare_and_classify(image, model, class_names):
    try:
        # Bild in RGB konvertieren
        image = image.convert("RGB")

        # Bild auf 224x224 zuschneiden
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # Bild in Numpy-Array konvertieren
        image_array = np.asarray(image)

        # Normalisieren
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Daten-Array erstellen
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Vorhersage
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])

        # Bereinige den Klassennamen (entferne Nummerierung)
        if ': ' in class_name:
            class_name = class_name.split(': ')[1]

        return class_name, confidence_score, index
    except Exception as e:
        st.error(f"Fehler bei der Bildklassifizierung: {e}")
        return None, None, None

# Element in Supabase speichern
def save_to_supabase(supabase, image, class_name, confidence_score, description, location, finder_name):
    try:
        # Bild in Bytes konvertieren für Supabase
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Eindeutigen Dateinamen erstellen
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"fundstuecke/{timestamp}_{class_name}.png"

        # Bild in Supabase Storage hochladen
        supabase.storage.from_("fundbuero-bilder").upload(
            file_name,
            img_byte_arr
        )

        # Öffentliche URL für das Bild generieren
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/fundbuero-bilder/{file_name}"

        # Datensatz in der Datenbank erstellen
        data = {
            "class_name": class_name,
            "class_index": int(confidence_score * 100),  # Speichere Konfidenz als Integer
            "confidence_score": confidence_score,
            "description": description,
            "location": location,
            "finder_name": finder_name,
            "image_url": image_url,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "gemeldet"
        }

        result = supabase.table("fundstuecke").insert(data).execute()
        return True, result
    except Exception as e:
        st.error(f"Fehler beim Speichern in Supabase: {e}")
        return False, None

# Fundstücke aus Supabase abrufen
def get_fundstuecke(supabase, filter_class=None, search_term=None):
    try:
        query = supabase.table("fundstuecke").select("*").order("created_at", desc=True)

        if filter_class and filter_class != "Alle":
            query = query.eq("class_name", filter_class)

        if search_term:
            query = query.ilike("description", f"%{search_term}%")

        result = query.execute()
        return result.data
    except Exception as e:
        st.error(f"Fehler beim Abrufen der Fundstücke: {e}")
        return []

# Haupt-App
def main():
    st.title("🔍 KI-Fundbüro")
    st.markdown("Lade ein gefundenes Objekt hoch oder suche nach verlorenen Gegenständen!")

    # Supabase und Modell initialisieren
    supabase = init_supabase()
    model, class_names = load_keras_model()

    if model is None or class_names is None:
        st.error("Das KI-Modell konnte nicht geladen werden. Bitte überprüfe die Dateien 'keras_Model.h5' und 'labels.txt'.")
        return

    # Extrahiere Klassennamen aus labels.txt
    available_classes = ["Alle"]
    for line in class_names:
        if ': ' in line:
            available_classes.append(line.split(': ')[1].strip())
        else:
            available_classes.append(line.strip())

    # Tabs für verschiedene Funktionen
    tab1, tab2 = st.tabs(["📤 Fundstück melden", "🔎 Nach verlorenen Sachen suchen"])

    # Tab 1: Fundstück melden
    with tab1:
        st.header("Neues Fundstück erfassen")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Wähle ein Bild des gefundenen Gegenstands aus",
                type=["jpg", "jpeg", "png", "bmp"]
            )

            if uploaded_file is not None:
                # Bild anzeigen
                image = Image.open(uploaded_file)
                st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        with col2:
            if uploaded_file is not None and st.button("🔍 Gegenstand erkennen", type="primary"):
                with st.spinner("KI analysiert das Bild..."):
                    image = Image.open(uploaded_file)
                    class_name, confidence_score, index = prepare_and_classify(image, model, class_names)

                    if class_name:
                        st.session_state['detected_class'] = class_name
                        st.session_state['detected_confidence'] = confidence_score
                        st.session_state['detected_image'] = image

                        st.success(f"✅ Erkannt: **{class_name}**")
                        st.info(f"Konfidenz: {confidence_score:.2%}")

            # Formular für Funddetails
            if 'detected_class' in st.session_state:
                with st.form("fund_form"):
                    st.subheader("Details zum Fundstück")

                    detected_class = st.text_input("Erkannte Kategorie",
                                                   value=st.session_state['detected_class'],
                                                   disabled=True)

                    description = st.text_area("Beschreibung",
                                              placeholder="z.B. Farbe, Marke, besondere Merkmale...")

                    location = st.text_input("Fundort",
                                            placeholder="Wo wurde der Gegenstand gefunden?")

                    finder_name = st.text_input("Name des Finders (optional)")

                    submitted = st.form_submit_button("📦 Fundstück speichern")

                    if submitted:
                        if description and location:
                            with st.spinner("Speichere in Datenbank..."):
                                success, result = save_to_supabase(
                                    supabase,
                                    st.session_state['detected_image'],
                                    detected_class,
                                    st.session_state['detected_confidence'],
                                    description,
                                    location,
                                    finder_name or "Anonym"
                                )

                                if success:
                                    st.success("✅ Fundstück erfolgreich gespeichert!")
                                    # Session State zurücksetzen
                                    del st.session_state['detected_class']
                                    del st.session_state['detected_confidence']
                                    del st.session_state['detected_image']
                                    st.rerun()
                        else:
                            st.warning("Bitte fülle alle Pflichtfelder aus (Beschreibung und Fundort).")

    # Tab 2: Suche nach verlorenen Sachen
    with tab2:
        st.header("Suche nach verlorenen Gegenständen")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            search_term = st.text_input("🔍 Suchbegriff", placeholder="z.B. 'rote Flasche'...")

        with col2:
            filter_class = st.selectbox("Kategorie filtern", available_classes)

        with col3:
            st.write("")  # Platzhalter für Alignment
            search_button = st.button("Suchen", type="primary", use_container_width=True)

        # Fundstücke anzeigen
        fundstuecke = get_fundstuecke(supabase, filter_class if filter_class != "Alle" else None, search_term)

        if fundstuecke:
            st.success(f"📊 {len(fundstuecke)} Fundstück(e) gefunden")

            # Grid-Layout für Fundstücke
            cols = st.columns(3)
            for idx, fund in enumerate(fundstuecke):
                with cols[idx % 3]:
                    with st.container(border=True):
                        # Bild anzeigen
                        if fund.get('image_url'):
                            st.image(fund['image_url'], use_column_width=True)

                        st.markdown(f"### {fund['class_name']}")
                        st.markdown(f"**Beschreibung:** {fund['description']}")
                        st.markdown(f"**Fundort:** {fund['location']}")
                        st.markdown(f"**Gemeldet von:** {fund['finder_name']}")
                        st.markdown(f"**Datum:** {fund['created_at'][:10]}")

                        # Konfidenz als Fortschrittsbalken anzeigen
                        confidence = fund.get('confidence_score', 0)
                        st.progress(confidence, text=f"KI-Konfidenz: {confidence:.1%}")

                        # Status anzeigen
                        status = fund.get('status', 'gemeldet')
                        if status == 'gemeldet':
                            st.caption("🟡 Noch nicht abgeholt")
                        else:
                            st.caption("✅ Bereits abgeholt")
        else:
            st.info("😕 Keine Fundstücke gefunden. Versuche andere Suchbegriffe oder lade ein neues Fundstück hoch!")

if __name__ == "__main__":
    main()
