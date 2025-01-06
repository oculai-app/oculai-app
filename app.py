import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="OculAI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MODEL_URL = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/found_eyegvd_94.pth"
CATEGORIES = ["Normal", "Cataracts", "Diabetic Retinopathy", "Glaucoma"]
CONDITION_DESCRIPTIONS = {
    "Normal": "The eye appears healthy with no detected abnormalities.",
    "Cataracts": "A clouding of the lens in the eye that affects vision.",
    "Diabetic Retinopathy": "Damage to the retina caused by complications of diabetes.",
    "Glaucoma": "A group of eye conditions that damage the optic nerve, often due to high pressure."
}
COLORS = {
    "Normal": "#00ff00",
    "Cataracts": "#ffff00",
    "Diabetic Retinopathy": "#ff0000",
    "Glaucoma": "#0082cb"
}

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

# Prediction function
@torch.no_grad()
def predict(image_tensor, model):
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# Sidebar for Input Method Selection and Image Upload/Capture
with st.sidebar:
    st.header("Input Image")
    input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

    img = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            try:
                img = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Invalid image file: {e}")
    elif input_method == "Capture from Camera":
        camera_image = st.camera_input("Capture Eye Image")
        if camera_image:
            try:
                img = Image.open(camera_image).convert("RGB")
            except Exception as e:
                st.error(f"Invalid camera input: {e}")

# Main Content Area for Analysis and Diagnosis
st.title("👁️ OculAI")
st.subheader("One Model, Countless Diseases")
st.markdown("Upload or capture an eye image from the sidebar to analyze potential eye conditions.")

# Model Loading Spinner
with st.spinner("Loading AI Model..."):
    model = load_model()

if img:
    # Display Selected Image in Main Content Area
    st.image(img, caption="Selected Image", use_column_width=True)

    # Analysis and Prediction Section
    with st.spinner("Analyzing..."):
        try:
            input_tensor = preprocess_image(img)
            probabilities = predict(input_tensor, model)

            # Display Predicted Category and Description
            prediction_idx = np.argmax(probabilities)
            prediction = CATEGORIES[prediction_idx]
            confidence_score = probabilities[prediction_idx] * 100

            st.markdown(f"<h3 style='color: {COLORS[prediction]}'>Predicted Category: {prediction}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p>{CONDITION_DESCRIPTIONS[prediction]}</p>", unsafe_allow_html=True)
            st.markdown(f"<strong>Confidence Score:</strong> {confidence_score:.2f}%", unsafe_allow_html=True)

            # Display Probabilities with Progress Bars and Colors
            st.markdown("<h3>Category Probabilities:</h3>", unsafe_allow_html=True)
            for category, prob in zip(CATEGORIES, probabilities):
                st.markdown(f"<strong>{category}:</strong> {prob * 100:.2f}%", unsafe_allow_html=True)
                progress_html = f"""
                <div style="background-color: #e0e0e0; border-radius: 25px; width: 100%; height: 18px; margin-bottom: 10px;">
                    <div style="background-color: {COLORS[category]}; width: {prob * 100}%; height: 100%; border-radius: 25px;"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)

            # Additional Analysis Features (Optional)
            st.markdown("<h3>Additional Insights:</h3>", unsafe_allow_html=True)
            if prediction != "Normal":
                st.warning(
                    f"The AI detected signs of {prediction}. Please consult an ophthalmologist for further evaluation."
                )
            else:
                st.success("The eye appears healthy! No abnormalities detected.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload or capture an eye image from the sidebar to proceed.")
