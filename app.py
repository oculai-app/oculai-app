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
    initial_sidebar_state="auto",
)

# Custom GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()

def show_cam_on_image(img, mask):
    heatmap = np.uint8(255 * mask)
    heatmap = Image.fromarray(heatmap).convert('RGB')
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap = np.array(heatmap)
    
    superimposed_img = heatmap * 0.4 + img * 255
    superimposed_img = np.uint8(superimposed_img / np.max(superimposed_img))
    return superimposed_img

# Load model with caching
@st.cache_resource
def load_model():
    try:
        url = "https://huggingface.co/oculotest/smart-scanner-model/resolve/main/found_eyegvd_92.pth"
        response = requests.get(url)
        response.raise_for_status()

        # Load pretrained EfficientNet-B0 and modify classifier
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, 4)

        state_dict = torch.load(io.BytesIO(response.content), map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise e

model = load_model()

# Preprocess image with data augmentation options
def preprocess_image_with_augmentation(image, apply_augmentation=False):
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    if apply_augmentation:
        transform_list.insert(0, transforms.RandomHorizontalFlip())
        transform_list.insert(1, transforms.RandomRotation(15))
    
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)

@torch.no_grad()
def predict(image_tensor):
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().tolist()
    return probabilities

# Grad-CAM for visualization
def generate_grad_cam(image_tensor):
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layer=target_layer)
    
    grayscale_cam = cam(image_tensor)
    rgb_image = image_tensor.squeeze().permute(1, 2, 0).numpy()
    rgb_image = (rgb_image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    cam_image = show_cam_on_image(rgb_image, grayscale_cam)
    return cam_image

st.title("OculAI")
st.subheader("One Model, Countless Diseases")

# Input method selection
input_method = st.radio("Choose Input Method", ("Upload Image", "Capture from Camera"))

img = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
elif input_method == "Capture from Camera":
    camera_image = st.camera_input("Capture Eye Image")
    if camera_image:
        img = Image.open(camera_image).convert("RGB")

if img:
    with st.spinner("Analyzing..."):
        st.image(img, caption="Selected Image", use_column_width=True)

        # Option to apply data augmentation
        apply_augmentation = st.checkbox("Apply Data Augmentation")
        
        input_tensor = preprocess_image_with_augmentation(img, apply_augmentation=apply_augmentation)
        
        try:
            probabilities = predict(input_tensor)

            categories = ["Normal", "Cataracts", "Diabetic Retinopathy", "Glaucoma"]
            prediction_idx = np.argmax(probabilities)
            prediction = categories[prediction_idx]

            # Display predictions and probabilities
            st.markdown(f"<h3 style='font-size:28px;'><strong>Predicted Category:</strong> {prediction}</h3>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='font-size:24px;'>Probabilities:</h3>", unsafe_allow_html=True)
            
            colors = {
                "Normal": "#00ff00",
                "Cataracts": "#ffcc00",
                "Diabetic Retinopathy": "#ff3300",
                "Glaucoma": "#3399ff"
            }
            
            for category, prob in zip(categories, probabilities):
                st.markdown(f"<p style='font-size:20px;'><strong>{category}:</strong> {prob * 100:.2f}%</p>", unsafe_allow_html=True)
                
                # Render progress bar with corresponding color
                progress_bar_html = f"""
                <div style="background-color: #e6e6e6; border-radius: 10px; width: 100%; height: 20px; margin-bottom: 10px;">
                    <div style="background-color: {colors[category]}; width: {prob * 100}%; height: 100%; border-radius: 10px;"></div>
                </div>
                """
                st.markdown(progress_bar_html, unsafe_allow_html=True)

            # Grad-CAM visualization
            cam_image = generate_grad_cam(input_tensor)
            st.image(cam_image, caption="Grad-CAM Visualization", use_column_width=True)
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload or capture an eye image to proceed.")
