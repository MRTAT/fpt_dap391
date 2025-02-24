
import streamlit as st
import os
import google.generativeai as genai
import torchvision.transforms as transforms
import torch
from PIL import Image
from model_lab1 import Cnn


# Gemini API setup
def setup_gemini():
    genai.configure(api_key='AIzaSyCLXHE3z4S8EzumKaIzBVZ0RGPL256oOmc')
    model = genai.GenerativeModel('gemini-pro')
    return model


# Disease prediction using Gemini
def get_gemini_response(model, symptoms):
    prompt = f"""
    Based on the following symptoms, suggest possible medical conditions and provide brief explanations:
    Symptoms: {symptoms}

    Please format your response as follows:
    1. Most likely conditions
    2. Brief explanation for each condition
    3. General recommendations

    Note: This is not a medical diagnosis. Please consult a healthcare professional for proper medical advice.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating response: {str(e)}"


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize về kích thước phù hợp
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_path).unsqueeze(0)  # Thêm chiều cho ảnh


def load_model():
    # Sử dụng thiết bị phù hợp (CPU hoặc GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình Cnn với 4 lớp phân loại
    model = Cnn(num_classes=4).to(device)

    # Đường dẫn đến file checkpoint
    checkpoint_path = os.path.join("trained_model", "best_cnn.pt")

    if not os.path.exists(checkpoint_path):
        st.error(f"Không tìm thấy file model tại {checkpoint_path}")
        return None

    try:
        # Cách 1: Load với weights_only=False (cách an toàn nhất nếu bạn tin tưởng nguồn checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        try:
            # Cách 2: Thử thêm numpy.core.multiarray.scalar vào danh sách safe globals
            import numpy as np
            from torch.serialization import add_safe_globals

            # Thêm numpy.core.multiarray.scalar vào danh sách safe globals
            add_safe_globals([np.core.multiarray.scalar])

            # Thử load lại với weights_only=True (mặc định)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
        except Exception as inner_e:
            st.error(f"Không thể load model: {str(inner_e)}")
            return None

    # Chuyển sang chế độ dự đoán
    model.eval()

    return model


# Streamlit page config
st.set_page_config(page_title="Chatbot AI", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 5px 0; }
        .user { background-color: #dcf8c6; text-align: right; }
        .bot { background-color: #f1f1f1; }

        /* CSS để tự động cuộn xuống phần chat */
        #chat-container {
            max-height: 600px;
            overflow-y: auto;
        }

        /* Xóa padding dưới cho container */
        .block-container {
            padding-bottom: 2rem;
        }

        /* Make columns equal height */
        .main-columns {
            display: flex;
            min-height: 700px;
        }

        /* Add some space between columns */
        .column-gap {
            padding: 0 10px;
        }
    </style>

    <script>
        // JavaScript để tự động cuộn xuống cuối phần chat
        function scrollToBottom() {
            var chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        // Chạy hàm khi trang đã tải xong
        window.addEventListener('load', scrollToBottom);
    </script>
    """,
    unsafe_allow_html=True,
)

# Call GPT
gemini_model = setup_gemini()

# Tiêu đề chính cho trang
st.markdown("<h1 style='text-align: center;'>💬 Chatbot AI - Healthcare services 🇻🇳</h1>", unsafe_allow_html=True)
# Create two main columns for the page layout
left_col, right_col = st.columns(2)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# RIGHT COLUMN - X-ray Prediction
# Add this new function after your get_gemini_response function
def get_treatment_recommendations(model, condition):
    prompt = f"""
    Provide evidence-based treatment recommendations and care guidelines for a patient with {condition} diagnosis from an X-ray.

    Please format your response as follows:
    1. Brief explanation of the condition
    2. Standard treatment approaches
    3. Home care recommendations
    4. When to see a doctor

    Note: This is for informational purposes. Always consult healthcare professionals for proper medical advice.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in generating treatment recommendations: {str(e)}"


# Then modify the prediction section in the right_col block to this:
# In the RIGHT COLUMN section, modify the code to include a toggle for the image

with right_col:
    st.subheader("📸 X-ray Prediction")

    uploaded_file = st.file_uploader("Choose X-ray (JPG/PNG)", type=["jpg", "png"])

    if uploaded_file is not None:
        st.success("✅ Image uploaded successfully!")

        # Add a toggle to show/hide the image
        show_image = st.checkbox("Show Image", value=True)

        # Only display the image if the toggle is on
        if show_image:
            st.image(uploaded_file, caption=" ", use_container_width=True)

        predict_button = st.button("Predict")

        if predict_button:
            try:
                # Image preprocessing
                image = Image.open(uploaded_file).convert("RGB")

                # Lấy thiết bị hiện tại
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Dự đoán kết quả
                model_lung = load_model()

                if model_lung is not None:
                    tensor = preprocess_image(image).to(device)
                    with torch.no_grad():
                        output = model_lung(tensor)
                        prediction = torch.argmax(output, 1).item()

                        classes = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']
                        condition = classes[prediction]
                        result = f"🧑🏽‍⚕️ 🙏🏽 **Prediction:** {condition}"
                        st.session_state.prediction_result = result
                        st.success(result)

                        # Get treatment recommendations for the condition
                        if condition != "Normal":
                            st.info("Generating treatment recommendations...")
                            treatment_info = get_treatment_recommendations(gemini_model, condition)
                            st.markdown("### Treatment Recommendations")
                            st.markdown(treatment_info)
                        else:
                            st.markdown("### Normal X-ray")
                            st.markdown(
                                "No specific treatment needed as the X-ray appears normal. Continue with regular health practices and consult with a healthcare provider for any persistent symptoms.")

                else:
                    st.error("Cannot load model. Please, check your path.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# LEFT COLUMN - Chat Interface
with left_col:
    st.subheader("Chat with AI 🖥️ ")

    # Display chat history with ID for JavaScript
    st.markdown('<div id="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Form để tránh auto-rerun khi nhập liệu
    with st.form(key="chat_form", clear_on_submit=True):

        input_text = st.text_input("", placeholder="Clearly state your symptoms here, please 🎀", key="user_input")
        submit_button = st.form_submit_button("Send")

    # Xử lý khi có tin nhắn mới
    if submit_button and input_text.strip():
        # Thêm tin nhắn người dùng vào session state
        st.session_state.messages.append({"role": "user", "content": f"**You:** {input_text}"})

        # Generate AI response
        with st.spinner("🤖 Thinking..."):
            response = get_gemini_response(gemini_model, input_text)

        # Thêm phản hồi của AI vào session state
        st.session_state.messages.append({"role": "assistant", "content": f"**AI:** {response}"})

        # Làm mới trang một lần để hiển thị tin nhắn mới
        st.rerun()