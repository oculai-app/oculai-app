import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
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
    "Normal": "#00ff00",  # Green
    "Cataracts": "#ffcc00",  # Yellow-Orange
    "Diabetic Retinopathy": "#ff3300",  # Red
    "Glaucoma": "#3399ff"  # Blue
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

# Initialize session state for managing uploads and toggle
if 'focused_diagnosis' not in st.session_state:
    st.session_state.focused_diagnosis = False
if 'images' not in st.session_state:
    st.session_state.images = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar for input and controls
with st.sidebar:
    st.header("Input Image")

    # Clear Data button to reset file uploader and session state
    if st.button("Clear Data"):
        st.session_state.images.clear()
        st.session_state.uploader_key += 1  # Increment key to reset file uploader
        st.session_state.focused_diagnosis = False  # Ensure Focused Diagnosis is disabled

    # File uploader with dynamic key for resetting
    uploaded_files = st.file_uploader(
        "Upload Eye Image(s)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.session_state.images.append((uploaded_file.name, img))
            except Exception as e:
                st.error(f"Invalid image file: {e}")

        # Automatically enable Focused Diagnosis if multiple files are uploaded
        if len(st.session_state.images) > 1:
            st.session_state.focused_diagnosis = True

# Main content area
st.title("👁️ OculAI")
st.subheader("One Model, Countless Diseases")
st.markdown("Upload or capture an eye image from the sidebar to analyze potential eye conditions.")

# Model Loading Spinner
with st.spinner("Loading AI Model..."):
    model = load_model()

st.success("Model loaded successfully!")

if st.session_state.images:
    if st.session_state.focused_diagnosis:  # Focused Diagnosis mode
        st.info("Focused Diagnosis mode activated: Displaying results for all uploaded images.")
        
        for image_name, img in st.session_state.images:
            with st.spinner(f"Analyzing {image_name}..."):
                try:
                    input_tensor = preprocess_image(img)
                    probabilities = predict(input_tensor, model)

                    # Get prediction and confidence score for this image
                    prediction_idx = np.argmax(probabilities)
                    prediction = CATEGORIES[prediction_idx]
                    confidence_score = probabilities[prediction_idx] * 100

                    # Display results for this image (minimal display for multiple images)
                    st.markdown(
                        f"**{image_name}**: <span style='color:{COLORS[prediction]}'>{prediction}</span> ({confidence_score:.2f}%)",
                        unsafe_allow_html=True,
                    )

                    # Display category probabilities for each image in Focused Diagnosis mode
                    for category, prob in zip(CATEGORIES, probabilities):
                        progress_bar_text = f"{category}: {prob * 100:.2f}%"
                        progress_color = COLORS[category]
                        progress_value = prob * 100

                        # Render progress bar and text below it
                        st.progress(prob)  # Streamlit's built-in progress bar widget
                        st.markdown(f"<p style='color:{progress_color}'>{progress_bar_text}</p>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during prediction for {image_name}: {e}")
                    
    else:  # Single Image Mode (Default)
        image_name, img = st.session_state.images[-1]  # Show only the latest uploaded image
        
        # Display selected image and bold text results in larger font size.
        st.image(img, caption=f"Selected Image: {image_name}", use_column_width=True)

        with st.spinner(f"Analyzing {image_name}..."):
            try:
                input_tensor = preprocess_image(img)
                probabilities = predict(input_tensor, model)

                # Get prediction and confidence score
                prediction_idx = np.argmax(probabilities)
                prediction = CATEGORIES[prediction_idx]
                confidence_score = probabilities[prediction_idx] * 100

                # Display detailed results for a single image with bold and larger text.
                st.markdown(
                    f"<h2 style='color: {COLORS[prediction]}'>Predicted Category: <b>{prediction}</b></h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='font-size:18px'>{CONDITION_DESCRIPTIONS[prediction]}</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<strong>Confidence Score:</strong> <span style='font-size:18px'>{confidence_score:.2f}%</span>",
                    unsafe_allow_html=True,
                )

                # Display category probabilities with progress bars.
                for category, prob in zip(CATEGORIES, probabilities):
                    progress_bar_text = f"{category}: {prob * 100:.2f}%"
                    progress_color = COLORS[category]

                    # Render progress bar and text below it.
                    st.progress(prob)  # Streamlit's built-in progress bar widget.
                    st.markdown(f"<p style='color:{progress_color}'>{progress_bar_text}</p>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction for {image_name}: {e}")
else:
    st.info("Please upload an eye image to proceed.")
