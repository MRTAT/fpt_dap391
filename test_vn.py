from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from model_lab1 import Cnn
import cv2
from torchsummary import summary
from sklearn.metrics import classification_report

def get_args():  # using for change params on terminal instead of open scrip
    parser = ArgumentParser(description="Cnn Inference")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Images size")
    parser.add_argument("--image_path", "-t", type=str, default="/home/mrly/Documents/ai_vietnguyen/deeplearning/fptu/dap/lab_1/code/images_test/normal_1.jpeg", help="Path for testing image")
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_model/best_cnn.pt", help="File_Trained")  # path where trained
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    categories = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Cnn(num_classes=4).to(device)  # to(device) -> set cuda
    summary(model, (3, 224, 224))  # Show architecture: how many layer - decrease

    if args.checkpoint:  # Load checkpoint to take best model
        checkpoint = torch.load(args.checkpoint)  # Syntax load model in torch
        model.load_state_dict(checkpoint["model"])
    else:
        print("Not exit checkpoint!")
        exit(0)
    model.eval()

    # Processing input image

    # dung cv2 but ban chat as pil -> su dung cho quen | if using pil then add transform: convert('RGB'), Resize, To_tensor
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)  # cuz cv2 -> BGR -> convert RGB |train process have to same test process
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1)) / 255.  # like To_tensor: (H, W, C) -> (C, H, W) --> change position of channel  | / 255. -> convert [0-255] -> [0, 1]

    # Add one dim cause model require 4 dim: Batch_size, C, H, W
    image = image[None, :, :, :]  # or .unsqueeze(0)   | [None, :, :, :] <=> 1 x 3 x 224 x 224
    image = torch.from_numpy(image).to(device).float() #  convert to tensor and Fix bug DoubleTensor

    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        prob = softmax(output)
    max_idx = torch.argmax(prob)
    predicted_class = categories[max_idx]
    print("Test image is:| {} | with probability score: {}".format(predicted_class, prob[0, max_idx]))
    cv2.imshow("{} --> {:.2f}%".format(predicted_class, prob[0, max_idx]*100), ori_image)  # cause prob.shape -> [1, 10] -> 2dim -> max just one dim
    # so using -> prob[0, max_idx]*100) -> disable the first dim  | *100 cal percent
    cv2.waitKey(0)



