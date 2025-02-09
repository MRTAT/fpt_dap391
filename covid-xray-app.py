import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from model_lab1 import Cnn
import cv2
import base64


def get_base64_of_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def add_bg_from_local(image_path):
    bg_img = get_base64_of_bg_image(image_path)
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{bg_img}");
                background-attachment: fixed;
                background-size: cover;
            }}

            /* Custom styles for the main container */
            .main-container {{
                background-color: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin: 2rem;
            }}

            /* Custom button style */
            .stButton>button {{
                width: 100%;
                padding: 0.75rem 1.5rem;
                font-size: 1.1rem;
                font-weight: 600;
                color: white;
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
            }}

            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
            }}

            /* Custom file uploader */
            .uploadedFile {{
                background-color: white;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                border: 2px dashed #4CAF50;
            }}

            /* Custom progress bar */
            .stProgress > div > div {{
                background-color: #4CAF50;
            }}

            /* Custom headers */
            h1 {{
                color: #2E4053;
                text-align: center;
                font-size: 2.5rem;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }}

            h3 {{
                color: #2E4053;
                margin-top: 1.5rem;
            }}

            /* Custom text */
            .prediction-text {{
                font-size: 1.2rem;
                padding: 1rem;
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                margin: 1rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn(num_classes=4).to(device)
    checkpoint = torch.load("trained_model/best_cnn.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def process_image(image, image_size=224):
    image_array = np.array(image)

    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    image_array = cv2.resize(image_array, (image_size, image_size))
    image_array = np.transpose(image_array, (2, 0, 1)) / 255.
    image_tensor = torch.from_numpy(image_array).float()
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def main():
    bg_image_path = "/home/mrly/Documents/ai_vietnguyen/deeplearning/fptu/dap/lab_1/code/theme.jpg"
    add_bg_from_local(bg_image_path)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.title("AI X-ray Disease Classification")
    st.markdown("""
        <p style='text-align: center; font-size: 1.2rem; color: #34495E;'>
            Upload your X-ray image for instant analysis and classification into one of these categories:<br>
            <strong>COVID-19 | Normal | Pneumonia | Tuberculosis</strong>
        </p>
    """, unsafe_allow_html=True)

    model, device = load_model()

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption='Uploaded X-ray image', use_container_width=True)

        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                processed_image = process_image(image)
                processed_image = processed_image.to(device)

                with torch.no_grad():
                    output = model(processed_image)
                    probabilities = nn.Softmax(dim=1)(output)

                categories = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[0][pred_idx].item() * 100

                st.markdown("""
                    <div class='prediction-text'>
                        <h3 style='color: #2E4053; text-align: center;'>Analysis Results</h3>
                        <p style='text-align: center; font-size: 1.5rem;'>
                            Predicted Condition: <strong style='color: #4CAF50;'>{}</strong><br>
                            Confidence: <strong style='color: #4CAF50;'>{:.2f}%</strong>
                        </p>
                    </div>
                """.format(categories[pred_idx], confidence), unsafe_allow_html=True)

                st.markdown("<h3 style='text-align: center;'>Probability Distribution</h3>", unsafe_allow_html=True)
                for idx, category in enumerate(categories):
                    prob = probabilities[0][idx].item() * 100
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(prob / 100)
                    with col2:
                        st.write(f"{prob:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()