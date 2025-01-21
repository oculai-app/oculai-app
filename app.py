import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="OculAI - Diabetic Retinopathy",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MODEL_URL = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/20250121_DR_DRgaussian_effnetb0_e30.pth"
CATEGORIES = ["No Diabetic Retinopathy", "Mild", "Moderate", "Severe", "Proliferative"]
CONDITION_DESCRIPTIONS = {
    "No Diabetic Retinopathy": "The eye appears healthy with no signs of diabetic retinopathy.",
    "Mild": "Early signs of diabetic retinopathy with small areas of damage to the retina.",
    "Moderate": "More extensive damage to the retina, requiring closer monitoring.",
    "Severe": "Significant damage to the retina that may lead to vision loss without treatment.",
    "Proliferative": "Advanced stage of diabetic retinopathy with abnormal blood vessel growth, posing a high risk of vision loss."
}
COLORS = {
    "No Diabetic Retinopathy": "#00ff00",  # Green
    "Mild": "#ffff00",  # Yellow
    "Moderate": "#ffa500",  # Orange
    "Severe": "#ff4500",  # Red-Orange
    "Proliferative": "#ff0000"  # Red
}

# Preprocess image with caching
@st.cache_data(show_spinner=False)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Load model with caching
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, len(CATEGORIES))
        state_dict = torch.load(io.BytesIO(response.content), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

# Prediction function
@torch.no_grad()
def predict(image_tensor, model):
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# Initialize session state for file uploader key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Initialize session state for current view
if 'current_view' not in st.session_state:
    st.session_state.current_view = None

# Sidebar for Input Method Selection and Image Upload/Capture
with st.sidebar:
    st.header("Input Image")

    # Display current viewed image at the top of the sidebar
    if st.session_state.current_view:
        st.image(st.session_state.current_view[1], caption=st.session_state.current_view[0], use_column_width=True)
        st.markdown("---")

    # Clear Data Button
    if st.button("Clear Data"):
        st.session_state.uploader_key += 1  # Increment key to reset file uploader
        st.session_state.current_view = None
        st.experimental_rerun()  # Reload app to apply changes

    # Input Method Selection
    input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

    images = []
    if input_method == "Upload Image":
        uploaded_files = st.file_uploader(
            "Upload Eye Image(s)",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    images.append((uploaded_file.name, img))
                except Exception as e:
                    st.error(f"Invalid image file: {e}")
    elif input_method == "Capture from Camera":
        camera_image = st.camera_input("Capture Eye Image")
        if camera_image:
            try:
                img = Image.open(camera_image).convert("RGB")
                images.append(("Captured Image", img))
            except Exception as e:
                st.error(f"Invalid camera input: {e}")

# Main Content Area for Analysis and Diagnosis
st.title("👁️ OculAI - Diabetic Retinopathy")
st.subheader("AI-Powered Detection of Diabetic Retinopathy Stages")
st.markdown("Upload or capture an eye image from the sidebar to analyze diabetic retinopathy stages.")

# Model Loading Spinner
with st.spinner("Loading AI Model..."):
    model = load_model()
st.success("Model loaded successfully!")

if images:
    for image_name, img in images:
        col1, col2, col3 = st.columns([8, 1, 1])

        with st.spinner(f"Analyzing {image_name}..."):
            try:
                input_tensor = preprocess_image(img)
                probabilities = predict(input_tensor, model)

                prediction_idx = np.argmax(probabilities)
                prediction = CATEGORIES[prediction_idx]
                confidence_score = probabilities[prediction_idx] * 100

                with col1:
                    st.markdown(
                        f"**{image_name}**: <span style='color:{COLORS[prediction]}'>{prediction}</span> ({confidence_score:.2f}%)",
                        unsafe_allow_html=True,
                    )

                    # Display detailed description for the predicted category
                    st.markdown(f"<p>{CONDITION_DESCRIPTIONS[prediction]}</p>", unsafe_allow_html=True)

                with col2:
                    if st.button("View", key=f"view_btn_{image_name}"):
                        st.session_state.current_view = (image_name, img)
                with col3:
                    if st.button("✕", key=f"close_btn_{image_name}"):
                        if st.session_state.current_view and st.session_state.current_view[0] == image_name:
                            st.session_state.current_view = None

            except Exception as e:
                st.error(f"Error during prediction for {image_name}: {e}")
else:
    st.info("Please upload or capture an eye image from the sidebar to proceed.")
