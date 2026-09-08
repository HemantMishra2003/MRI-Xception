# ================= ALL IMPORTANT LIBRARIES =================
import os
import csv
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import requests
from PIL import Image

from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import qrcode
from streamlit_geolocation import streamlit_geolocation


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="🧠 MRI AI Cancer Detector Model",
    layout="centered"
)

#  QR DATA
QR_DATA = "https://github.com/HemantMishra2003"


#   UI
st.markdown("""
<style>
.stTextInput input {
    font-size: 17px !important;
    font-weight: 600 !important;
}
.field-label {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 4px;
    margin-top: 14px;
}
@keyframes blink {
  0% {opacity: 1;}
  50% {opacity: 0.3;}
  100% {opacity: 1;}
}
.detect-box {
  border: 2px solid red;
  padding: 18px;
  border-radius: 10px;
  text-align: center;
  animation: blink 1s infinite;
  margin-top: 20px;
}
.detect-title {
  color: red;
  font-size: 26px;
  font-weight: 800;
}
.detect-conf {
  color: red;
  font-size: 18px;
  margin-top: 6px;
  font-weight: 600;
}
.hospital-scroll-wrapper {
  display: flex;
  gap: 16px;
  overflow-x: auto;
  padding: 10px 4px 20px 4px;
}
.hospital-card {
  flex: 0 0 180px;
  background: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.hospital-card img {
  width: 100%;
  height: 110px;
  object-fit: contain;
  background: #f7f7f7;
  padding: 14px 0;
  display: block;
}
.hospital-card-body {
  padding: 10px 12px;
}
.hospital-card-name {
  margin: 0 0 4px 0;
  font-size: 14px;
  font-weight: 700;
  color: #111;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.hospital-card-distance {
  margin: 0;
  font-size: 13px;
  color: #555;
}
</style>
""", unsafe_allow_html=True)


#     ASSETS
CLASS_IMAGE_MAP = {
    "Glioma": "assets/Te-gl_0224.jpg",
    "Meningioma": "assets/Te-me_0010.jpg",
    "Pituitary": "assets/Te-pi_0025.jpg",
    "No Tumor": "assets/Te-no_0114.jpg",
}
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CANCER_LABEL_MAP = {
    "Glioma": "Glioma Cancer Detected",
    "Meningioma": "Meningioma Cancer Detected",
    "Pituitary": "Pituitary Gland Cancer Detected",
    "No Tumor": "No Tumor Detected"
}

PDF_CANCER_LABEL_MAP = {
    "Glioma": "GLIOMA BRAIN CANCER DETECTED",
    "Meningioma": "MENINGIOMA BRAIN CANCER DETECTED",
    "Pituitary": "PITUITARY GLAND CANCER DETECTED",
    "No Tumor": "NO TUMOR DETECTED"
}


# ================= MULTI-LANGUAGE SUPPORT =================
TRANSLATIONS = {
    "English": {
        "title": "🧠 MRI AI Cancer Detector Model",
        "subtitle": "(Xception | Transfer Learning)",
        "patient_name_label": "Patient Name",
        "patient_name_ph": "Enter Patient Name",
        "patient_age_label": "Patient Age",
        "upload_label": "Upload MRI Image",
        "model_loaded": "Your Model loaded successfully",
        "no_tumor": "No Tumor Detected",
        "reference_header": "Reference MRI Images with Confidence",
        "confidence_prefix": "Confidence",
        "generate_pdf_btn": "Generate MRI PDF Report",
        "download_pdf_btn": "Download MRI Report PDF",
        "please_fill": "👆 Please enter patient details and upload an MRI image.",
        "gradcam_header": "🔍 AI Focus Heatmap (Grad-CAM)",
        "gradcam_desc": "This shows which part of the scan the AI focused on to make its decision.",
        "symptom_header": "📝 Symptom Checklist",
        "symptom_desc": "Optionally check any symptoms the patient has experienced.",
        "combined_risk_header": "⚠️ Combined Risk Assessment",
        "email_header": "📧 Email This Report",
        "email_desc": "Send this PDF report directly to an email address.",
        "email_placeholder": "Enter recipient email address",
        "email_send_btn": "Send Report via Email",
        "timeline_header": "📈 Patient Visit History",
        "timeline_desc": "Past predictions recorded for this patient name.",
        "hospital_header": "🏥 Hospitals Near You",
        "hospital_desc": "Since a tumor was detected, here are hospitals close to your current location.",
    },
    "हिन्दी (Hindi)": {
        "title": "🧠 एमआरआई एआई कैंसर डिटेक्टर मॉडल",
        "subtitle": "(Xception | ट्रांसफर लर्निंग)",
        "patient_name_label": "रोगी का नाम",
        "patient_name_ph": "रोगी का नाम दर्ज करें",
        "patient_age_label": "रोगी की आयु",
        "upload_label": "MRI छवि अपलोड करें",
        "model_loaded": "आपका मॉडल सफलतापूर्वक लोड हो गया",
        "no_tumor": "कोई ट्यूमर नहीं पाया गया",
        "reference_header": "आत्मविश्वास के साथ संदर्भ MRI छवियां",
        "confidence_prefix": "आत्मविश्वास",
        "generate_pdf_btn": "MRI पीडीएफ रिपोर्ट बनाएं",
        "download_pdf_btn": "MRI रिपोर्ट पीडीएफ डाउनलोड करें",
        "please_fill": "👆 कृपया रोगी विवरण दर्ज करें और एक MRI छवि अपलोड करें।",
        "gradcam_header": "🔍 एआई फोकस हीटमैप (Grad-CAM)",
        "gradcam_desc": "यह दर्शाता है कि निर्णय लेने के लिए एआई ने स्कैन के किस भाग पर ध्यान केंद्रित किया।",
        "symptom_header": "📝 लक्षण चेकलिस्ट",
        "symptom_desc": "वैकल्पिक रूप से रोगी द्वारा अनुभव किए गए किसी भी लक्षण की जांच करें।",
        "combined_risk_header": "⚠️ संयुक्त जोखिम मूल्यांकन",
        "email_header": "📧 यह रिपोर्ट ईमेल करें",
        "email_desc": "इस पीडीएफ रिपोर्ट को सीधे ईमेल पते पर भेजें।",
        "email_placeholder": "प्राप्तकर्ता का ईमेल पता दर्ज करें",
        "email_send_btn": "ईमेल के माध्यम से रिपोर्ट भेजें",
        "timeline_header": "📈 रोगी विज़िट इतिहास",
        "timeline_desc": "इस रोगी नाम के लिए दर्ज की गई पिछली भविष्यवाणियां।",
        "hospital_header": "🏥 आपके आस-पास के अस्पताल",
        "hospital_desc": "चूंकि एक ट्यूमर का पता चला है, यहां आपके वर्तमान स्थान के पास के अस्पताल हैं।",
    }
}


def t(key, lang="English"):
    """Translate a UI string key into the selected language, with safe fallback."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(
        key, TRANSLATIONS["English"].get(key, key)
    )


# ================= LANGUAGE SELECTOR =================
lang = st.selectbox("🌐 Language / भाषा", options=list(TRANSLATIONS.keys()), index=0)

# ================= HEADER =================
st.title(t("title", lang))
st.caption(t("subtitle", lang))

# ================= INPUT =================
st.markdown(f"<p class='field-label'>{t('patient_name_label', lang)}</p>", unsafe_allow_html=True)
name = st.text_input("Patient Name", placeholder=t("patient_name_ph", lang), label_visibility="collapsed")

st.markdown(f"<p class='field-label'>{t('patient_age_label', lang)}</p>", unsafe_allow_html=True)
age = st.number_input("Patient Age", min_value=0, max_value=120, step=1, label_visibility="collapsed")

st.markdown(f"<p class='field-label'>{t('upload_label', lang)}</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")


# ================= MODEL LOAD (WEIGHTS BASED – FINAL SAFE) =================
@st.cache_resource
def load_model():
    img_shape = (299, 299, 3)

    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(
        repo_id="HemantMishraDeepak/newXception",
        filename="xception_brain_tumor_weights.weights.h5"
    )

    model.load_weights(weights_path)
    return model


model = load_model()
st.success(t("model_loaded", lang))


# ================= PREDICTION =================
def predict_mri(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(299, 299))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    probs = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2) for i in range(4)}
    detected = CLASS_NAMES[np.argmax(preds)]
    return detected, probs


# ================= GRAD-CAM HEATMAP =================
# Uses the ALREADY LOADED, ALREADY TRAINED model. No retraining, no weight
# changes. Grad-CAM only inspects gradients of an existing forward pass.
def _get_output_ndim(layer):
    """
    Returns number of dims of a layer's output shape, trying multiple
    attribute names for compatibility across TensorFlow/Keras versions.
    (Older Keras used `.output_shape`; Keras 3 removed it in favor of
    `.output.shape`, so we try the modern way first, then fall back.)
    """
    try:
        shape = layer.output.shape
        if shape is not None:
            return len(shape)
    except Exception:
        pass
    try:
        shape = layer.output_shape
        if shape is not None:
            return len(shape)
    except Exception:
        pass
    return None


def generate_gradcam_heatmap(uploaded_file, pred_index=None):
    """
    Generates a Grad-CAM heatmap overlay for the given uploaded MRI image
    using the existing trained `model`. Returns an RGB uint8 numpy array,
    or None if it could not be computed (fails safely, never breaks the
    main prediction flow).
    """
    try:
        img = image.load_img(uploaded_file, target_size=(299, 299))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        original_rgb = np.uint8(image.img_to_array(img))

        if len(model.layers) == 0:
            return None
        base_model = model.layers[0]

        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if _get_output_ndim(layer) == 4:
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(last_conv_layer).output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, base_output = grad_model(arr)
            x = base_output
            for layer in model.layers[1:]:
                x = layer(x)
            predictions = x

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlayed = cv2.addWeighted(heatmap_color, 0.4, original_rgb, 0.6, 0)
        return overlayed

    except Exception as e:
        st.warning(f"Grad-CAM heatmap could not be generated: {e}")
        return None


# ================= NEARBY HOSPITAL FINDER =================
# Uses OpenStreetMap's free Overpass API. The public overpass-api.de
# instance is known to be slow/unstable, so we try several public
# mirrors in sequence and stop at the first one that responds.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

HOSPITAL_PLACEHOLDER_IMG = "https://cdn-icons-png.flaticon.com/512/2966/2966327.png"


def _haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance (in km) between two GPS points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def find_nearby_hospitals(lat, lon, radius_km=5):
    """
    Searches for hospitals within `radius_km` of the given coordinates.
    Tries multiple Overpass API mirrors in sequence (the free public
    servers are sometimes slow/down). Returns (hospitals, error_message).
    If successful, error_message is None. If all mirrors fail,
    hospitals is [] and error_message explains what happened.
    """
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:20];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center;
    """

    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = requests.get(endpoint, params={"data": query}, timeout=20)
            if response.status_code != 200:
                last_error = f"Server returned status {response.status_code}"
                continue

            data = response.json()
            hospitals = []
            for element in data.get("elements", []):
                h_lat = element.get("lat") or element.get("center", {}).get("lat")
                h_lon = element.get("lon") or element.get("center", {}).get("lon")
                if h_lat is None or h_lon is None:
                    continue

                hospital_name = element.get("tags", {}).get("name", "Unnamed Hospital")
                distance_km = round(_haversine_km(lat, lon, h_lat, h_lon), 2)

                hospitals.append({
                    "name": hospital_name,
                    "latitude": h_lat,
                    "longitude": h_lon,
                    "distance_km": distance_km
                })

            hospitals.sort(key=lambda h: h["distance_km"])
            return hospitals, None  # success — stop trying other mirrors

        except requests.exceptions.Timeout:
            last_error = "The hospital search server took too long to respond."
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            continue
        except ValueError:
            last_error = "The hospital search server returned an unexpected response."
            continue

    # All mirrors failed
    return [], (last_error or "Could not reach any hospital search server.")


def render_nearby_hospitals_section(lang="English"):
    """
    Renders the 'Nearby Hospitals' section as a horizontally scrolling
    row of cards (image on top, hospital name + distance below), placed
    at the bottom of the page. Only called when a tumor is detected.
    """
    st.markdown("---")
    st.subheader(t("hospital_header", lang))
    st.write(t("hospital_desc", lang))

    radius_km = st.slider("Search radius (km)", min_value=1, max_value=20, value=5, key="hospital_radius")

    st.write("📍 Allow location access below so we can find hospitals near you:")
    location = streamlit_geolocation()

    if location and location.get("latitude") and location.get("longitude"):
        lat = location["latitude"]
        lon = location["longitude"]
        st.success(f"Location detected: ({lat:.4f}, {lon:.4f})")

        with st.spinner("Searching nearby hospitals..."):
            hospitals, error = find_nearby_hospitals(lat, lon, radius_km=radius_km)

        if error:
            st.error(
                f"Could not fetch hospital data right now ({error}). "
                "This is usually a temporary issue with the free map server — "
                "please wait a moment and try again."
            )
        elif hospitals:
            st.write(f"Found **{len(hospitals)}** hospital(s) within {radius_km} km:")

            cards_html = '<div class="hospital-scroll-wrapper">'
            for h in hospitals:
                safe_name = h["name"].replace("<", "").replace(">", "")
                cards_html += f"""
                <div class="hospital-card">
                    <img src="{HOSPITAL_PLACEHOLDER_IMG}" />
                    <div class="hospital-card-body">
                        <p class="hospital-card-name">{safe_name}</p>
                        <p class="hospital-card-distance">{h['distance_km']} km away</p>
                    </div>
                </div>
                """
            cards_html += "</div>"

            st.markdown(cards_html, unsafe_allow_html=True)
        else:
            st.info(
                f"No hospitals found within {radius_km} km using OpenStreetMap data. "
                "Try increasing the search radius above — some areas have limited "
                "hospital data mapped in OpenStreetMap."
            )
    else:
        st.warning("Click the location icon above and allow access to detect your position.")


# ================= SYMPTOM CHECKLIST + RISK FUSION =================
SYMPTOM_OPTIONS = {
    "headache": "Frequent headaches",
    "vision_issues": "Blurred or double vision",
    "seizures": "Seizures",
    "nausea": "Nausea or vomiting",
    "memory_issues": "Memory or concentration issues",
    "balance_issues": "Balance or coordination issues",
}
SYMPTOM_WEIGHTS = {
    "headache": 8,
    "vision_issues": 12,
    "seizures": 20,
    "nausea": 6,
    "memory_issues": 10,
    "balance_issues": 9,
}


def render_symptom_checklist(lang="English"):
    st.subheader(t("symptom_header", lang))
    st.caption(t("symptom_desc", lang))

    selected = []
    cols = st.columns(2)
    for i, (key, label) in enumerate(SYMPTOM_OPTIONS.items()):
        with cols[i % 2]:
            if st.checkbox(label, key=f"symptom_{key}"):
                selected.append(key)
    return selected


def compute_symptom_score(selected_symptoms):
    if not selected_symptoms:
        return 0.0
    raw = sum(SYMPTOM_WEIGHTS.get(s, 0) for s in selected_symptoms)
    max_possible = sum(SYMPTOM_WEIGHTS.values())
    return round((raw / max_possible) * 100, 1)


def compute_combined_risk(model_confidence_pct, symptom_score_pct, tumor_detected):
    if not tumor_detected:
        return {
            "combined_score": None,
            "label": "No imaging risk detected by AI.",
            "symptom_score": symptom_score_pct
        }

    combined = round(0.7 * model_confidence_pct + 0.3 * symptom_score_pct, 1)
    if combined >= 80:
        label = "High combined risk"
    elif combined >= 50:
        label = "Moderate combined risk"
    else:
        label = "Low-moderate combined risk"

    return {"combined_score": combined, "label": label, "symptom_score": symptom_score_pct}


def render_combined_risk_section(model_confidence_pct, tumor_detected, lang="English"):
    st.markdown("---")
    selected_symptoms = render_symptom_checklist(lang)
    symptom_score = compute_symptom_score(selected_symptoms)
    risk_result = compute_combined_risk(model_confidence_pct, symptom_score, tumor_detected)

    st.subheader(t("combined_risk_header", lang))
    if risk_result["combined_score"] is not None:
        st.metric("Combined Risk Score", f"{risk_result['combined_score']}%", risk_result["label"])
    else:
        st.info(risk_result["label"])
        st.write(f"Symptom concern score: **{risk_result['symptom_score']}%**")

    return risk_result


# ================= EMAIL REPORT =================
def send_report_email(sender_email, sender_app_password, receiver_email, pdf_path, patient_name):
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = f"MRI AI Diagnostic Report - {patient_name}"

        body = (
            f"Dear recipient,\n\n"
            f"Please find attached the MRI AI diagnostic report for patient: {patient_name}.\n\n"
            f"This report was generated automatically by the MRI AI Cancer Detector Model.\n\n"
            f"Regards,\nMRI AI Cancer Detector Model"
        )
        msg.attach(MIMEText(body, "plain"))

        with open(pdf_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
        part["Content-Disposition"] = f'attachment; filename="{os.path.basename(pdf_path)}"'
        msg.attach(part)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()

        return True, "Report emailed successfully."
    except Exception as e:
        return False, f"Could not send email: {e}"


def render_email_section(pdf_path, patient_name, lang="English"):
    st.markdown("---")
    st.subheader(t("email_header", lang))
    st.caption(t("email_desc", lang))

    with st.expander("Sender email settings (one-time setup)"):
        st.caption(
            "Uses your Gmail account to send the email. You need a Gmail "
            "'App Password' (Google Account > Security > App Passwords), "
            "not your normal Gmail password."
        )
        sender_email = st.text_input("Your Gmail address", key="sender_email_input")
        sender_password = st.text_input(
            "Gmail App Password", type="password", key="sender_password_input"
        )

    receiver_email = st.text_input(t("email_placeholder", lang), key="receiver_email_input")

    if st.button(t("email_send_btn", lang)):
        if not sender_email or not sender_password:
            st.error("Please enter sender Gmail address and App Password above.")
        elif not receiver_email:
            st.error("Please enter a recipient email address.")
        elif not os.path.isfile(pdf_path):
            st.error("Please generate the PDF report first before emailing it.")
        else:
            with st.spinner("Sending email..."):
                success, message = send_report_email(
                    sender_email, sender_password, receiver_email, pdf_path, patient_name
                )
            if success:
                st.success(message)
            else:
                st.error(message)


# ================= COMPARISON TIMELINE / HISTORY =================
HISTORY_FILE = "patient_history.csv"
HISTORY_COLUMNS = ["timestamp", "patient_name", "age", "detected", "confidence", "combined_risk"]


def log_prediction_to_history(patient_name, age, detected, confidence, combined_risk):
    try:
        file_exists = os.path.isfile(HISTORY_FILE)
        with open(HISTORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(HISTORY_COLUMNS)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                patient_name, age, detected, confidence,
                combined_risk if combined_risk is not None else ""
            ])
    except Exception:
        pass


def get_patient_history(patient_name):
    if not os.path.isfile(HISTORY_FILE):
        return []
    rows = []
    try:
        with open(HISTORY_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["patient_name"].strip().lower() == patient_name.strip().lower():
                    rows.append(row)
    except Exception:
        return []
    return rows


def render_timeline_section(patient_name, lang="English"):
    history = get_patient_history(patient_name)
    if len(history) < 2:
        return

    st.markdown("---")
    st.subheader(t("timeline_header", lang))
    st.caption(t("timeline_desc", lang))

    try:
        confidences = [float(r["confidence"]) for r in history]
        st.line_chart(confidences)
        st.dataframe(
            [
                {
                    "Visit Time": r["timestamp"],
                    "Result": r["detected"],
                    "Confidence (%)": r["confidence"],
                    "Combined Risk (%)": r["combined_risk"] if r["combined_risk"] else "N/A"
                }
                for r in history
            ],
            use_container_width=True
        )
    except Exception as e:
        st.info(f"Could not render timeline chart: {e}")


# ================= PDF GENERATOR  =================
def generate_pdf(img_path, name, age, detected, probs):
    pdf = "MRI_AI_Report.pdf"

    # ---------- QR ----------
    qr = qrcode.make(QR_DATA)
    qr_path = "temp_qr.png"
    qr.save(qr_path)

    c = canvas.Canvas(pdf, pagesize=A4)
    w, h = A4
    y = h - 40

    # ================= HEADER =================
    c.setFont("Helvetica-Bold", 19)
    c.drawCentredString(
        w/2, y,
        "SHRI HEMKUND INSTITUTE OF RADIOLOGY & IMAGING"
    )
    y -= 28

    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(w/2, y, "MRI AI Diagnostic Report")
    y -= 16

    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(
        w/2, y,
        "(Fine-Tuned Xception | Transfer Learning)"
    )
    y -= 25

    # QR top-right
    c.drawImage(qr_path, w-110, h-120, 70, 70)

    # ================= PATIENT INFO =================
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, f"Name : {name}")
    y -= 14
    c.drawString(50, y, f"Age  : {age}")
    y -= 14
    c.drawString(
        50, y,
        f"Date : {datetime.now().strftime('%d-%m-%Y')}"
    )
    y -= 25

    # ================= MAIN RESULT =================
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(
        w/2, y,
        PDF_CANCER_LABEL_MAP[detected]
    )
    y -= 18

    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(
        w/2, y,
        f"Confidence: {probs[detected]:.2f}%"
    )
    y -= 25

    # TWO IMAGES
    img_w, img_h = 200, 160
    left_x = 50
    right_x = w/2 + 20

    # Uploaded MRI (left)
    c.drawImage(img_path, left_x, y-img_h, img_w, img_h)

    # Reference MRI (right)
    c.drawImage(
        CLASS_IMAGE_MAP["No Tumor"],
        right_x, y-img_h, img_w, img_h
    )

    # Labels
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(
        left_x + img_w/2,
        y - img_h - 14,
        f"Uploaded MRI : {detected}"
    )
    c.drawCentredString(
        right_x + img_w/2,
        y - img_h - 14,
        "Reference MRI"
    )

    y -= img_h + 40

    # ================= THREE IMAGES (BOTTOM) =================
    small_w, small_h = 130, 130
    x_pos = [40, 200, 360]

    for i, cls in enumerate(["Glioma", "Meningioma", "Pituitary"]):
        c.drawImage(
            CLASS_IMAGE_MAP[cls],
            x_pos[i], y-small_h, small_w, small_h
        )
        c.drawCentredString(
            x_pos[i] + small_w/2,
            y - small_h - 14,
            cls
        )
        c.drawCentredString(
            x_pos[i] + small_w/2,
            y - small_h - 28,
            f"Confidence: {probs[cls]:.2f}%"
        )

    y -= small_h + 55

    # ================= FOOTER =================
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(
        w/2, y,
        "Scan my QR code to see my other Projects"
    )
    y -= 20

    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(
        w/2, y,
        "Developed under the Guidance of Dr. Vishwas Mishra Sir ( Rolls-Royce )"
    )

    c.save()
    os.remove(qr_path)
    return pdf


# ================= MAIN FLOW =================
if uploaded_file and name and age:
    with open("uploaded_temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    st.image(uploaded_file, caption="Uploaded MRI", width=320)

    detected, probs = predict_mri(uploaded_file)

    if detected != "No Tumor":
        st.markdown(
            f"""
            <div class="detect-box">
                <div class="detect-title">{CANCER_LABEL_MAP[detected]}</div>
                <div class="detect-conf">Confidence: {probs[detected]}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.success(f"{t('no_tumor', lang)} (Confidence: {probs['No Tumor']}%)")

    # ---- Grad-CAM heatmap, only shown when a tumor is detected ----
    if detected != "No Tumor":
        st.markdown("---")
        st.subheader(t("gradcam_header", lang))
        st.caption(t("gradcam_desc", lang))
        with st.spinner("Generating heatmap..."):
            heatmap_img = generate_gradcam_heatmap(uploaded_file)
        if heatmap_img is not None:
            st.image(heatmap_img, caption="AI Focus Heatmap", width=320)

    st.subheader(t("reference_header", lang))
    cols = st.columns(4)
    for col, cls in zip(cols, CLASS_IMAGE_MAP.keys()):
        with col:
            st.image(CLASS_IMAGE_MAP[cls], width=200)
            st.markdown(f"<b>{cls}</b>", unsafe_allow_html=True)
            st.markdown(f"{t('confidence_prefix', lang)}: {probs[cls]}%")

    # ---- Symptom checklist + combined risk fusion ----
    risk_result = render_combined_risk_section(probs[detected], detected != "No Tumor", lang)

    # ---- Log this prediction to history ONLY ONCE per uploaded file.
    # Streamlit reruns the whole script on every widget interaction, so
    # without this guard the same visit would be logged many times. ----
    file_size = getattr(uploaded_file, "size", len(uploaded_file.getvalue()))
    file_signature = f"{uploaded_file.name}_{file_size}_{name}"
    if st.session_state.get("last_logged_file") != file_signature:
        log_prediction_to_history(
            patient_name=name,
            age=age,
            detected=detected,
            confidence=probs[detected],
            combined_risk=risk_result["combined_score"]
        )
        st.session_state["last_logged_file"] = file_signature

    if st.button(t("generate_pdf_btn", lang)):
        pdf = generate_pdf("uploaded_temp.jpg", name, age, detected, probs)
        st.session_state["last_pdf_path"] = pdf
        st.session_state["last_pdf_patient"] = name

    # ---- Show download + email options if a PDF has been generated in this
    # session. Using session_state so these controls don't disappear when
    # the email button itself is clicked and Streamlit reruns the script. ----
    if st.session_state.get("last_pdf_path") and os.path.isfile(st.session_state["last_pdf_path"]):
        pdf = st.session_state["last_pdf_path"]
        with open(pdf, "rb") as f:
            st.download_button(t("download_pdf_btn", lang), f, file_name=pdf)

        render_email_section(pdf, st.session_state.get("last_pdf_patient", name), lang)

    # ---- Comparison timeline, shown only if patient has 2+ past visits ----
    render_timeline_section(name, lang)

    # ---- Footer section: nearby hospitals, shown at the bottom of the
    # page, only if a tumor was detected ----
    if detected != "No Tumor":
        render_nearby_hospitals_section(lang)

else:
    st.info(t("please_fill", lang))

