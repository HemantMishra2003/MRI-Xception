# ================= ALL IMPORTANT LIBRARIES =================
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from datetime import datetime
from PIL import Image

from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import qrcode

# ---- Added for "Nearby Hospitals" feature ----
import requests
from streamlit_geolocation import streamlit_geolocation

# ---- Added for Grad-CAM heatmap ----
import cv2

# ---- Added for Email report feature ----
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# ---- Added for Comparison timeline / history ----
import csv


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
label, .stTextInput label {
    margin-bottom: -25px !important;
}
.stTextInput {
    margin-top: -10px !important;
}
.stTextInput input {
    font-size: 17px !important;
    font-weight: 600 !important;
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

# ================= MULTI-LANGUAGE SUPPORT (NEW FEATURE) =================
# Simple key -> text dictionary translation. Does not touch model or PDF logic.
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
        "symptom_desc": "Optionally check any symptoms the patient has experienced. This adds a clinical context score alongside the AI's image-based result.",
        "combined_risk_header": "⚠️ Combined Risk Assessment",
        "email_header": "📧 Email This Report",
        "email_desc": "Send this PDF report directly to an email address.",
        "email_placeholder": "Enter recipient email address",
        "email_send_btn": "Send Report via Email",
        "timeline_header": "📈 Patient Visit History",
        "timeline_desc": "Past predictions recorded for this patient name.",
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
        "symptom_desc": "वैकल्पिक रूप से रोगी द्वारा अनुभव किए गए किसी भी लक्षण की जांच करें। यह एआई के छवि-आधारित परिणाम के साथ एक नैदानिक संदर्भ स्कोर जोड़ता है।",
        "combined_risk_header": "⚠️ संयुक्त जोखिम मूल्यांकन",
        "email_header": "📧 यह रिपोर्ट ईमेल करें",
        "email_desc": "इस पीडीएफ रिपोर्ट को सीधे ईमेल पते पर भेजें।",
        "email_placeholder": "प्राप्तकर्ता का ईमेल पता दर्ज करें",
        "email_send_btn": "ईमेल के माध्यम से रिपोर्ट भेजें",
        "timeline_header": "📈 रोगी विज़िट इतिहास",
        "timeline_desc": "इस रोगी नाम के लिए दर्ज की गई पिछली भविष्यवाणियां।",
    }
}


def t(key, lang="English"):
    """Translate a UI string key into the selected language, with safe fallback."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(
        key, TRANSLATIONS["English"].get(key, key)
    )


# ================= LANGUAGE SELECTOR (NEW FEATURE) =================
lang = st.selectbox("🌐 Language / भाषा", options=list(TRANSLATIONS.keys()), index=0)

# ================= HEADER =================
st.title(t("title", lang))
st.markdown(
    f"<p style='color:white; margin-top:-10px; margin-left:75px; "
    f"font-size:17px; font-weight:650;'>"
    f"{t('subtitle', lang)}</p>",
    unsafe_allow_html=True
)

# ================= INPUT =================
st.markdown(f"<p style='font-size:20px; font-weight:600;'>{t('patient_name_label', lang)}</p>", unsafe_allow_html=True)
name = st.text_input("", placeholder=t("patient_name_ph", lang))

st.markdown(f"<p style='font-size:20px; font-weight:600;'>{t('patient_age_label', lang)}</p>", unsafe_allow_html=True)
age = st.number_input("", min_value=0, max_value=120, step=1)

st.markdown(
    f"<p style='font-size:20px; font-weight:600;'>{t('upload_label', lang)}</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "png", "jpeg"]
)



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


# ================= GRAD-CAM HEATMAP (NEW FEATURE) =================
# Uses the ALREADY LOADED, ALREADY TRAINED model. No retraining, no weight
# changes. Grad-CAM only inspects gradients of an existing forward pass.
def generate_gradcam_heatmap(uploaded_file, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given uploaded MRI image using the
    existing trained `model`. Returns an RGB uint8 numpy array (the original
    image with the heatmap overlaid), or None if Grad-CAM could not be
    computed (fails safely so it never breaks the main prediction flow).
    """
    try:
        # Prepare input the same way predict_mri does
        img = image.load_img(uploaded_file, target_size=(299, 299))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        original_rgb = np.uint8(image.img_to_array(img))

        # The Xception base model is the first layer of our Sequential model
        base_model = model.layers[0]

        # Find the last convolutional layer inside Xception automatically
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if len(layer.output_shape) == 4:  # conv-like layers have 4D output
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
            # Pass base_model's output through the remaining layers
            # (GlobalAveragePooling2D -> Dropout -> Dense -> Dropout -> Dense)
            x = base_output
            for layer in model.layers[1:]:
                x = layer(x)
            predictions = x

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap to match original image size and colorize
        heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlayed = cv2.addWeighted(heatmap_color, 0.4, original_rgb, 0.6, 0)
        return overlayed

    except Exception as e:
        # Fail safely — Grad-CAM is a bonus feature, must never break the app
        st.warning(f"Grad-CAM heatmap could not be generated: {e}")
        return None


# ================= NEARBY HOSPITAL FINDER (NEW FEATURE) =================
# Uses OpenStreetMap's free Overpass API — no API key required.
def find_nearby_hospitals(lat, lon, radius_km=5):
    """
    Searches for hospitals within `radius_km` kilometers of the given
    latitude/longitude using the Overpass API (OpenStreetMap data).
    Returns a list of dicts: [{name, latitude, longitude, distance_km}, ...]
    """
    radius_m = int(radius_km * 1000)
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
    );
    out center;
    """

    try:
        response = requests.get(overpass_url, params={"data": query}, timeout=25)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"Could not fetch hospital data right now: {e}")
        return []

    hospitals = []
    for element in data.get("elements", []):
        # Nodes have lat/lon directly; ways/relations use "center"
        h_lat = element.get("lat") or element.get("center", {}).get("lat")
        h_lon = element.get("lon") or element.get("center", {}).get("lon")
        if h_lat is None or h_lon is None:
            continue

        name = element.get("tags", {}).get("name", "Unnamed Hospital")
        distance_km = round(_haversine_km(lat, lon, h_lat, h_lon), 2)

        hospitals.append({
            "name": name,
            "latitude": h_lat,
            "longitude": h_lon,
            "distance_km": distance_km
        })

    # Closest hospitals first
    hospitals.sort(key=lambda h: h["distance_km"])
    return hospitals


def _haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance (in km) between two GPS points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# A free, no-key-required placeholder hospital image (public domain style icon
# image hosted on a stable CDN). Used identically on every card since we are
# not using any paid photo API.
HOSPITAL_PLACEHOLDER_IMG = "https://cdn-icons-png.flaticon.com/512/2966/2966327.png"


def render_nearby_hospitals_section():
    """
    Renders the 'Nearby Hospitals' section as a horizontally scrolling
    row of cards (image on top, hospital name + distance below), placed
    at the bottom of the page. Only called when a tumor is detected.
    Does not affect model/prediction logic in any way.
    """
    st.markdown("---")
    st.subheader("🏥 Hospitals Near You")
    st.write("Since a tumor was detected, here are hospitals close to your current location.")

    radius_km = st.slider("Search radius (km)", min_value=1, max_value=20, value=5)

    st.write("📍 Allow location access below so we can find hospitals near you:")
    location = streamlit_geolocation()

    if location and location.get("latitude") and location.get("longitude"):
        lat = location["latitude"]
        lon = location["longitude"]
        st.success(f"Location detected: ({lat:.4f}, {lon:.4f})")

        with st.spinner("Searching nearby hospitals..."):
            hospitals = find_nearby_hospitals(lat, lon, radius_km=radius_km)

        if hospitals:
            st.write(f"Found **{len(hospitals)}** hospital(s) within {radius_km} km:")

            # ---- Build one card per hospital: image on top, name + distance below ----
            cards_html = '<div style="display:flex; gap:16px; overflow-x:auto; padding:10px 4px 20px;">'
            for h in hospitals:
                safe_name = h["name"].replace("<", "").replace(">", "")
                cards_html += f"""
                <div style="flex:0 0 180px; background:#ffffff; border:1px solid #e0e0e0;
                            border-radius:12px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.08);">
                    <img src="{HOSPITAL_PLACEHOLDER_IMG}" style="width:100%; height:110px;
                         object-fit:contain; background:#f7f7f7; padding:14px 0;" />
                    <div style="padding:10px 12px;">
                        <p style="margin:0 0 4px; font-size:14px; font-weight:700; color:#111;
                                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                            {safe_name}
                        </p>
                        <p style="margin:0; font-size:13px; color:#555;">
                            {h['distance_km']} km away
                        </p>
                    </div>
                </div>
                """
            cards_html += "</div>"

            st.markdown(cards_html, unsafe_allow_html=True)

        else:
            st.info(
                "No hospitals found in this radius. Try increasing the search radius above."
            )
    else:
        st.warning("Click the location icon above and allow access to detect your position.")


# ================= SYMPTOM CHECKLIST + RISK FUSION (NEW FEATURE) =================
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
    """
    Renders a symptom checklist UI and returns the list of symptom keys
    the user checked. Purely additive — does not affect prediction.
    """
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
    """Returns a 0-100 clinical concern score based on checked symptoms."""
    if not selected_symptoms:
        return 0.0
    raw = sum(SYMPTOM_WEIGHTS.get(s, 0) for s in selected_symptoms)
    max_possible = sum(SYMPTOM_WEIGHTS.values())
    return round((raw / max_possible) * 100, 1)


def compute_combined_risk(model_confidence_pct, symptom_score_pct, tumor_detected):
    """
    Fuses the AI model's confidence with the symptom-based score.
    Model confidence is weighted higher (70%) since it is image-based
    evidence; symptoms contribute the remaining 30% as clinical context.
    """
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
    """
    Renders the symptom checklist and the resulting combined risk score.
    Returns the risk_result dict so callers (PDF/email/history) can use it.
    """
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


# ================= EMAIL REPORT (NEW FEATURE) =================
def send_report_email(sender_email, sender_app_password, receiver_email, pdf_path, patient_name):
    """
    Sends the generated PDF report via Gmail SMTP. Requires a Gmail
    address + an "App Password" (not the normal Gmail password) for
    the sender account. Returns (success: bool, message: str).
    """
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
    """Renders the 'Email This Report' UI section."""
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


# ================= COMPARISON TIMELINE / HISTORY (NEW FEATURE) =================
HISTORY_FILE = "patient_history.csv"
HISTORY_COLUMNS = ["timestamp", "patient_name", "age", "detected", "confidence", "combined_risk"]


def log_prediction_to_history(patient_name, age, detected, confidence, combined_risk):
    """Appends one prediction record to a local CSV file. Fails silently
    (never breaks the app) if the file system is not writable."""
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
    """Returns a list of dict rows for all past predictions matching this
    patient name (case-insensitive). Returns [] if no history exists yet."""
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
    """Renders the patient's past-visit history as a simple line chart
    (confidence over time) plus a table, if at least 2 records exist."""
    history = get_patient_history(patient_name)
    if len(history) < 2:
        return  # Nothing meaningful to compare yet

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

    # ---- Added: Grad-CAM heatmap, only shown when a tumor is detected ----
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

    # ---- Added: Symptom checklist + combined risk fusion ----
    risk_result = render_combined_risk_section(probs[detected], detected != "No Tumor", lang)

    # ---- Added: log this prediction to history ONLY ONCE per uploaded file.
    # Streamlit reruns the whole script on every widget interaction (checking
    # a symptom box, moving a slider, switching language), so without this
    # guard the same visit would be logged many times. ----
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
    # session. Using session_state (not nesting inside the button's if-block)
    # so these controls don't disappear when the email button itself is
    # clicked and Streamlit reruns the script. ----
    if st.session_state.get("last_pdf_path") and os.path.isfile(st.session_state["last_pdf_path"]):
        pdf = st.session_state["last_pdf_path"]
        with open(pdf, "rb") as f:
            st.download_button(t("download_pdf_btn", lang), f, file_name=pdf)

        # ---- Added: Email report section, shown once PDF exists ----
        render_email_section(pdf, st.session_state.get("last_pdf_patient", name), lang)

    # ---- Added: Comparison timeline, shown only if patient has 2+ past visits ----
    render_timeline_section(name, lang)

    # ---- Added: footer section — nearby hospitals, shown at the bottom
    # of the page, only if a tumor was detected ----
    if detected != "No Tumor":
        render_nearby_hospitals_section()


else:
    st.info(t("please_fill", lang))
