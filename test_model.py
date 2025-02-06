from dataset import CovidData
from model import Cnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from sklearn.metrics import classification_report, accuracy_score
from torchvision.transforms.v2 import ToTensor, Resize, Compose
from PIL import Image
from argparse import ArgumentParser  # using for change params on terminal instead of open scrip



def get_args():  # sử dụng để thay đổi tham số trên terminal
    parser = ArgumentParser(description="Cnn Inference")
    parser.add_argument("--root", "-r", type=str,
                        default="/home/mrly/Documents/ai_vietnguyen/deeplearning/fptu/dap/lab_1/dataset_covid19_big",
                        help="Root_Covid19_dataset")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Images size")
    parser.add_argument("--num_classes", type=int, default=4, help="num_classes (The number of class prediction)")
    parser.add_argument("--trained_model", "-t", type=str, default="trained_model",
                        help="Path to trained model")  # Đường dẫn đến mô hình đã huấn luyện
    args = parser.parse_args()
    return args


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    if torch.cuda.is_available():
        model.cuda()
    return model


def predict_image(image_path, model, transform):
    # Read Image
    image = Image.open(image_path).convert("RGB")
    # Apply (resize, chuyển đổi thành tensor)
    image = transform(image).unsqueeze(0)  # unsqueeze để thêm một chiều batch (1, C, H, W)

    if torch.cuda.is_available():
        image = image.cuda()

    # Predict
    model.eval()  # Chuyển mô hình sang chế độ evaluation
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, 1)  # take class has the highest p

    return predicted


if __name__ == '__main__':
    args = get_args()

    # Khởi tạo transform
    trans = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])

    # Init model CNN
    model = Cnn(num_classes=args.num_classes)

    # Load model trained --> mode: best
    model = load_model(model, "{}/best_cnn.pt".format(args.trained_model))

    image_path = input("Enter path photos: ")

    predicted_index = predict_image(image_path, model, trans)

    class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
    predicted_class = class_names[predicted_index]

    print(f"Predict: {predicted_class}")  # In ra kết quả dự đoán
    