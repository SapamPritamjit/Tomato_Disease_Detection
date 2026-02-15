import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import tempfile

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅")
# -----------------------
# Load external CSS
def load_css():
    with open("styles.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------
# THEME INITIALIZATION
# -----------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_toggle = st.sidebar.toggle(
    "🌗 Dark Mode",
    value=(st.session_state.theme == "dark")
)

st.session_state.theme = "dark" if theme_toggle else "light"

# Theme colors
if st.session_state.theme == "dark":
    navbar_bg = "rgba(15, 23, 42, 0.85)"
    navbar_text = "white"
    body_bg = "linear-gradient(-45deg, #0f172a, #0b1120, #111827, #0f172a)"
    body_text = "white"
else:
    navbar_bg = "rgba(255, 255, 255, 0.85)"
    navbar_text = "#111827"
    body_bg = "linear-gradient(135deg, #f9fafb, #e5e7eb)"
    body_text = "#111827"

# Apply background
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: {body_bg};
    color: {body_text};
    transition: background 0.4s ease, color 0.4s ease;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------
# FLOATING NAVBAR
# -----------------------
st.markdown(f"""
<div style="
    position:fixed;
    top:0;
    left:0;
    width:100%;
    padding:15px 30px;
    backdrop-filter: blur(14px);
    background:{navbar_bg};
    color:{navbar_text};
    display:flex;
    justify-content:space-between;
    align-items:center;
    z-index:999;
    font-weight:600;
">
    <div>🍅 AgroScan AI</div>
    <div>{"🌙 Dark Mode" if st.session_state.theme=="dark" else "☀️ Light Mode"}</div>
</div>
""", unsafe_allow_html=True)

# Add spacing so content isn't hidden behind navbar
st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)

# CONFIG
# -----------------------
IMG_SIZE = 300
THRESHOLD = 0.40

label_columns = [
    "Early blight",
    "Healthy",
    "Late blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus"
]

translations = {
    "English": {
        "upload": "Upload Tomato Leaf Image",
        "analyzing": "🤖 AI is analyzing the leaf...",
        "primary": "🌟 Primary Disease",
        "secondary": "⚠ Possible Secondary Diseases",
        "description": "📝 Description",
        "treatment": "💊 Treatment Recommendation",
        "spray": "🌿 Spray Suggestion",
        "download": "📄 Download PDF Report",
        "download_btn": "Download Report",
        "healthy": "✅ Plant appears healthy.",
        "uploaded_caption": "Uploaded Image",
        "confidence": "Confidence",
        "disease_info": "📘 Disease Information",
        "more_about": "📘 More about"
    },
    "Hindi": {
        "upload": "टमाटर पत्ती की तस्वीर अपलोड करें",
        "analyzing": "🤖 AI पत्ती का विश्लेषण कर रहा है...",
        "primary": "🌟 मुख्य रोग",
        "secondary": "⚠ संभावित अन्य रोग",
        "description": "📝 विवरण",
        "treatment": "💊 उपचार सुझाव",
        "spray": "🌿 स्प्रे सुझाव",
        "download": "📄 पीडीएफ रिपोर्ट डाउनलोड करें",
        "download_btn": "रिपोर्ट डाउनलोड करें",
        "healthy": "✅ पौधा स्वस्थ दिखाई देता है।",
        "uploaded_caption": "अपलोड की गई तस्वीर",
        "confidence": "विश्वास स्तर",
        "disease_info": "📘 रोग की जानकारी",
        "more_about": "📘 और जानकारी"
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "English"

language = st.sidebar.selectbox(
    "Select Language / भाषा चुनें",
    ["English", "Hindi"],
    index=["English", "Hindi"].index(st.session_state.lang)
)

st.session_state.lang = language
T = translations[language]
# -----------------------
# DISEASE INFO DATABASE
# -----------------------
disease_info = {
    "Early blight": {
        "info": "Fungal disease causing brown concentric spots.",
        "treatment": "Remove infected leaves. Improve air circulation.",
        "spray": "Spray Mancozeb or Chlorothalonil every 7-10 days."
    },
    "Late blight": {
        "info": "Serious fungal disease causing dark lesions.",
        "treatment": "Remove affected plants immediately.",
        "spray": "Use Metalaxyl + Mancozeb spray."
    },
    "Leaf Miner": {
        "info": "Insect larvae feeding inside leaves.",
        "treatment": "Remove affected leaves.",
        "spray": "Spray Spinosad or Neem oil."
    },
    "Magnesium Deficiency": {
        "info": "Yellowing between leaf veins.",
        "treatment": "Apply Epsom salt solution.",
        "spray": "Foliar spray of Magnesium Sulfate."
    },
    "Nitrogen Deficiency": {
        "info": "Older leaves turn pale yellow.",
        "treatment": "Add nitrogen-rich fertilizer.",
        "spray": "Apply Urea solution carefully."
    },
    "Pottassium Deficiency": {
        "info": "Leaf edge burning and yellowing.",
        "treatment": "Apply potash fertilizer.",
        "spray": "Use Potassium nitrate spray."
    },
    "Spotted Wilt Virus": {
        "info": "Virus causing ring spots and distortion.",
        "treatment": "Remove infected plants.",
        "spray": "Control thrips using recommended insecticide."
    },
    "Healthy": {
        "info": "Plant appears healthy.",
        "treatment": "Maintain proper irrigation and nutrition.",
        "spray": "No spray required."
    }
}

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tomato_multilabel_final.keras")

model = load_model()

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def predict_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    results = []
    for i, prob in enumerate(preds):
        if prob > THRESHOLD:
            results.append((label_columns[i], float(prob)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# -----------------------
# PDF GENERATION
# -----------------------
def generate_pdf(predictions):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Tomato Disease Detection Report", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    for disease, prob in predictions:
        elements.append(Paragraph(f"Disease: {disease}", styles["Heading2"]))
        elements.append(Paragraph(f"Confidence: {prob*100:.2f}%", styles["Normal"]))
        elements.append(Paragraph(f"Info: {disease_info[disease]['info']}", styles["Normal"]))
        elements.append(Paragraph(f"Treatment: {disease_info[disease]['treatment']}", styles["Normal"]))
        elements.append(Paragraph(f"Spray Suggestion: {disease_info[disease]['spray']}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    return temp_file.name

st.markdown("<h1 style='margin-top:20px;'>🍅 AgroScan AI</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    T["upload"],
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.image(image, use_container_width=True)
        st.caption(T["uploaded_caption"])

    with col2:

        # Animated Loading Spinner
        with st.spinner(T["analyzing"]):
            predictions = predict_image(image)

        if predictions:

            primary = predictions[0]
            secondary = predictions[1:]

            # -------- PRIMARY GLASS CARD --------
            confidence_color = (
            "#22c55e" if primary[1] > 0.75
                else "#facc15" if primary[1] > 0.50
                else "#ef4444"
            )

            st.markdown(
                f"""
                <div class="glass-card primary-disease">
                    <h2>{T['primary']}</h2>
                    <h1>{primary[0]}</h1>
                    <h3 style="color:{confidence_color}; font-weight:600;">
                        {primary[1]*100:.2f}% {T['confidence']}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )


            # PRIMARY DISEASE INFO
            with st.expander(f"{T['disease_info']} - {primary[0]}", expanded=False):
                st.markdown(f"### {T['description']}")
                st.write(disease_info[primary[0]]["info"])

                st.markdown(f"### {T['treatment']}")
                st.write(disease_info[primary[0]]["treatment"])

                st.markdown(f"### {T['spray']}")
                st.write(disease_info[primary[0]]["spray"])

            # -------- SECONDARY --------
            if secondary:
                st.markdown(f"### {T['secondary']}")

                for disease, prob in secondary:

                    # Secondary Card
                    st.markdown(
                        f"""
                        <div class="glass-card secondary-disease">
                            <h4>{disease}</h4>
                            <p>{prob*100:.2f}% {T['confidence']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Expandable Info for EACH secondary
                    with st.expander(f"{T['more_about']} {disease}", expanded=False):

                        st.markdown(f"### {T['description']}")
                        st.write(disease_info[disease]["info"])

                        st.markdown(f"### {T['treatment']}")
                        st.write(disease_info[disease]["treatment"])

                        st.markdown(f"### {T['spray']}")
                        st.write(disease_info[disease]["spray"])


            # -------- PDF --------
            if st.checkbox(T["download"]):
                pdf_path = generate_pdf(predictions)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=T["download_btn"],
                        data=f,
                        file_name="Tomato_Disease_Report.pdf",
                        mime="application/pdf"
                    )

        else:
            st.success(T["healthy"])


st.markdown("""
<div class="footer">
AgroScan AI © 2026 | AI for Sustainable Agriculture 🌱
</div>
""", unsafe_allow_html=True)
