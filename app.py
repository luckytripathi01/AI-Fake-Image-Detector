import streamlit as st # Library to create the web interface
import tensorflow as tf # For running the AI model
import numpy as np # For mathematical calculations
from PIL import Image # For image processing
import requests # To fetch images from internet URL
from io import BytesIO # To handle image data streams
import plotly.graph_objects as go # For professional charts

# 1. PAGE SETUP
st.set_page_config(page_title="AI Face Authenticator Pro", layout="wide")

# 2. CSS STYLING (The Final UI Polish)
# 2. CSS STYLING (The Ultimate Dark Fix)
st.markdown("""
    <style>
    /* 1. Hide top white header and sidebar */
    header[data-testid="stHeader"], [data-testid="stSidebar"] { display: none !important; }
    
    /* 2. Overall Dark Background */
    .stApp { background-color: #0d1117 !important; color: white; }

    /* 3. Container Boxes */
    .header-box {
        background-color: #161b22; border-radius: 15px; padding: 25px;
        border: 1px solid #30363d; margin-bottom: 25px; text-align: center;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background-color: #161b22; border-radius: 12px; padding: 20px;
        border: 1px solid #30363d; margin-bottom: 15px;
    }
         /* 4. --- FORCE DARK FIX & BOX DESIGN --- */
    /* Target uploader to create the dashed box */
    [data-testid="stFileUploader"] {
        background-color: #161b22 !important;
        border: 2px dashed #444 !important; /* This creates the box border */
        border-radius: 12px !important;
        padding: 15px !important;
        margin-top: 10px !important;
    }

    /* Target EVERY element inside to stay dark */
    [data-testid="stFileUploader"] * {
        background-color: transparent !important;
        color: white !important;
    }

    
    
    /* Remove white inner bar and borders */
    [data-testid="stFileUploader"] > section {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Style the Browse button */
    [data-testid="stFileUploader"] button {
        background-color: #30363d !important;
        border: 1px solid #444 !important;
        color: white !important;
    }

    /* 5. Typography and Labels */
    div[data-testid="stWidgetLabel"] p, div[data-testid="stRadio"] label p {
        color: white !important;
        font-weight: 500;
    }
    .result-header { font-size: 26px; font-weight: bold; }
    .fake-color { color: #ff4b4b !important; }
    .real-color { color: #28a745 !important; }

    /* 6. Progress Bar */
    .bar-container { background-color: #2d333b; border-radius: 10px; width: 100%; height: 28px; margin: 15px 0; overflow: hidden; border: 1px solid #444; }
    .bar-fill { height: 100%; border-radius: 10px; text-align: center; color: white; font-weight: bold; line-height: 28px; transition: 0.8s; }
    </style>
    """, unsafe_allow_html=True)

# 3. LOAD THE TRAINED MODEL
@st.cache_resource
def load_my_model():
    try:
        # Load your saved Keras model
        return tf.keras.models.load_model('model.keras')
    except:
        return None

model = load_my_model()

# 4. MAIN TITLE
st.markdown("""
    <div class='header-box'>
        <h1 style='margin:0;'>🕵️‍♂️ AI Fake Image Detection System</h1>
        <p style='margin:10px 0 0 0; opacity:0.8; font-size: 16px;'>Scan Local Files or Internet URL for Synthetic Pattern Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# 5. MAIN 3-COLUMN LAYOUT
col1, col2, col3 = st.columns([1, 1.3, 0.9], gap="medium")
img = None
file_size = "N/A"

with col1:
    st.markdown("### 📥 Input Image")
    # Selection for upload method
    input_type = st.radio("Select Method:", ("Local Upload", "Image URL"), horizontal=True)
    
    if input_type == "Local Upload":
        # Added 'jfif' and 'webp' to support Google Images
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "jfif", "webp"], label_visibility="collapsed")
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            file_size = f"{round(uploaded_file.size/1024, 2)} KB"
    else:
        url_input = st.text_input("Paste Image URL", placeholder="https://example.com")
        if url_input:
            try:
                resp = requests.get(url_input, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                file_size = f"{round(len(resp.content)/1024, 2)} KB"
            except:
                st.error("Fetch failed. Try a different URL.")

    if img:
        st.markdown("<br>", unsafe_allow_html=True)
        # Display small centered image preview
        st.image(img, caption="Scanning Pixels...", width=260)

with col2:
    if img and model:
        # Preprocessing for MobileNetV2 input size (224x224)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner('Analyzing patterns...'):
            pred_raw = model.predict(img_array)
            score = float(pred_raw.item())
        
        # Calculate probabilities
        fake_prob = (1 - score) * 100
        real_prob = score * 100
        is_fake = fake_prob > 50
        
        res_label = "FAKE IMAGE 🚫" if is_fake else "REAL IMAGE ✅"
        res_class = "fake-color" if is_fake else "real-color"

        # Result Display Card
        st.markdown(f"""
            <div style='background-color:#0d1117; padding:20px; border-radius:12px; border:1px solid #30363d; margin-bottom:15px;'>
                <div class='result-header'>Result: <span class='{res_class}'>{res_label}</span></div>
                <p style='margin-top:5px; font-size:18px;'><b>Confidence:</b> {max(fake_prob, real_prob):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("Fake Probability Index")
        st.markdown(f"""
            <div class='bar-container'>
                <div class='bar-fill' style='width: {fake_prob}%; background-color: #ff4b4b;'>{fake_prob:.1f}% Fake</div>
            </div>
        """, unsafe_allow_html=True)

        # Confidence Bar Chart
        fig = go.Figure(go.Bar(x=['Real', 'Fake'], y=[real_prob, fake_prob], marker_color=['#28a745', '#ff4b4b'], width=[0.4, 0.4]))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=230, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### 🔍 Analysis Panel")
        st.info("System Standby. Please provide an input image.")

with col3:
    st.markdown("### 📝 Details")
    # Metadata extraction
    res_text = f"{img.size[0]}x{img.size[1]} px" if img else "N/A"
    status_text = "Scan Complete" if img else "Idle"
    
    st.markdown(f"""
        <div style='background-color:#0d1117; padding:15px; border-radius:10px; border:1px solid #30363d; margin-bottom:20px;'>
            <p><b>Resolution:</b> {res_text}</p>
            <p><b>Size:</b> {file_size}</p>
            <p><b>Status:</b> {status_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 AI Engine")
    st.markdown(f"""
        <div style='background-color:#0d1117; padding:15px; border-radius:10px; border:1px solid #30363d;'>
            <p><b>Model:</b> MobileNetV2</p>
            <p><b>Environment:</b> Local PC</p>
            <p><b>Build:</b> v2.6.0 Stable</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center; opacity:0.4; font-size:12px;'>Developed by Lucky | Deep Learning Project</p>", unsafe_allow_html=True)
