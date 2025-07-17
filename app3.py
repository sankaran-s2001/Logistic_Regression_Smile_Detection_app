import streamlit as st
import joblib
from PIL import Image
import numpy as np
from time import sleep

st.set_page_config(
    page_title="SmileScan AI",
    page_icon="😊")

# Load pre trained logistic regression model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("img_class_lgr_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.sidebar.header("SmileScan AI-Information")
st.sidebar.markdown("""
**Model Details:**
- Algorithm: Logistic Regression
- Input: 64×64 grayscale images
- Training Accuracy: 89%
- Features: 4,096 pixels (64×64)

**How It Works:**
1. Upload any face image
2. The model analyzes pixel patterns
3. Returns smile probability

**Tips for Best Results:**
- Use clear front-facing photos
- Avoid extreme angles
- Works best with human faces
""")

def preprocess_img(image):
    image = image.resize((64,64))   # Resize using PIL
    image = image.convert("L")  # Convert to gray scale
    image = np.array(image).flatten().reshape(1, -1)    # Flatten for logistic regression
    image = scaler.transform(image) # Apply standard scaler
    return image

st.title("SmileScan AI - Smile Detection App")
st.markdown("Upload a face image to detect if the person is smiling or not")

# Example images section
st.subheader("Quick Test Examples:")
col1, col2 = st.columns(2)

example_img_paths = {
    "Smile": r"D:\Data Science Training\Bootcamp\DS_with_keerthi\ML\Logistic Regression\Project\Big smile image.jpg",
    "No Smile": r"D:\Data Science Training\Bootcamp\DS_with_keerthi\ML\Logistic Regression\Project\Non smile image.jpg"
}

with col1:
    if st.button("😁 Smile Example", help="Click to test with a smiling face"):
        img = Image.open(example_img_paths["Smile"])
        st.session_state.example_img = img
with col2:
    if st.button("😐 No Smile Example", help="Click to test with a neutral face"):
        img = Image.open(example_img_paths["No Smile"])
        st.session_state.example_img = img

# File uploader
uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

# Use either uploaded file or example image
if uploaded_file is not None:
    img = Image.open(uploaded_file)
elif 'example_img' in st.session_state:
    img = st.session_state.example_img
else:
    img = None

if img is not None:
    st.image(img, caption="Your image: ")
    
    with st.spinner("🧐 Analyzing facial expression..."):
        sleep(1)  # Simulate processing time
    
        processed_img = preprocess_img(img)
        prediction_probability = model.predict_proba(processed_img)
        prediction = model.predict(processed_img)
    
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        smile_score = int(prediction_probability[0][1] * 100)
        st.success(f"✅ Smiling Detected! (Confidence score: {smile_score}%)")
        st.balloons()
    else:
        Non_smile_score = int(prediction_probability[0][0] * 100)
        st.warning(f"❌ No smile Detected! (Confidence score: {Non_smile_score}%)")
    
    # Show probability breakdown
    st.write("Probability Breakdown:")
    st.write(f"- Not Smiling: {prediction_probability[0][0]*100:.2f}%")
    st.write(f"- Smiling: {prediction_probability[0][1]*100:.2f}%")