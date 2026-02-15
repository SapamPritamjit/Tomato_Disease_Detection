# AgroScan AI  
### AI-Powered Tomato Disease Detection System

AgroScan AI is a deep learning-based web application that detects tomato leaf diseases using computer vision.  
It analyzes uploaded images and predicts primary and possible secondary diseases with confidence scores and treatment recommendations.

**Live App:**  
https://tomatodiseasedetection-vhsiya38oknzqbpst4tuqj.streamlit.app

---

## Features

-  Multi-label tomato disease detection
-  EfficientNet-based deep learning model
-  Confidence score visualization
-  Primary & secondary disease prediction
-  Detailed disease information
-  Treatment & spray recommendations
-  Multi-language support (English & Hindi)
-  Downloadable PDF report
-  Deployed on Streamlit Cloud
-  Auto model download from Google Drive

---

## Model Details

- Architecture: EfficientNet (Transfer Learning)
- Framework: TensorFlow / Keras
- Image Size: 300x300
- Output: Multi-label classification
- Threshold: 0.40
- Model Size: ~124MB

Diseases Detected:
- Early Blight
- Late Blight
- Leaf Miner
- Magnesium Deficiency
- Nitrogen Deficiency
- Potassium Deficiency
- Spotted Wilt Virus
- Healthy

---

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- PIL
- ReportLab (PDF generation)
- gdown (Google Drive model download)
- dotenv (Environment variables)

---

## Project Structure

app.py              → Main Streamlit application  
styles.css          → UI styling  
requirements.txt    → Dependencies  
.env                → Google Drive model ID  
README.md           → Project documentation  


## Install Dependencies (Run Locally)

pip install -r requirements.txt

## Run App

streamlit run app.py
