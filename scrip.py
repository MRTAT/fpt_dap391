
from dataset import CovidData
from model_lab1 import Cnn
from torch.utils.data import DataLoader

from argparse import ArgumentParser  # using for change params on terminal instead of open scrip
from tqdm import tqdm # Show progress
from tqdm.autonotebook import tqdm # Show progress on jupiter notebook
from torch.utils.tensorboard import SummaryWriter  # we can see history of progress train (loss - accuracy ..)
import os
import shutil  # using for delete tensorboard
import cv2

import torch.nn as nn
import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torchvision.transforms.v2 import ToTensor, Resize, Compose, RandomAffine, ColorJitter

# Add confusion matrix into tensorboard
import numpy as np
import matplotlib.pyplot as plt


def get_args():  # using for change params on terminal instead of open scrip
    parser = ArgumentParser(description="Cnn Training")
    parser.add_argument("--root", "-r", type=str, default="/home/mrly/Documents/ai_vietnguyen/deeplearning/fptu/dap/lab_1/dataset_covid19_big", help="Root_Covid19_Dataset")

    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")  # "-e" -- type on terminal more quickly
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Number of batch_size")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Images size")
    parser.add_argument("--num_workers", type=int, default=10, help="num_workers (high with if strong pc)")
    parser.add_argument("--num_classes", type=int, default=4, help="num_classes (The number of class prediction)")
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rate: Default: 1e-3")
    parser.add_argument("--momentum", type=int, default=0.95, help="Momentum: Default: 0.95")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="History training")
    parser.add_argument("--trained_model", "-t", type=str, default="trained_model", help="File_Trained")  # path where trained
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="File_Trained")  # path where trained
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm , class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap='Wistia')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel("True_Label")
    plt.xlabel("Predict_Label")
    writer.add_figure("confusion_matrix", figure, epoch)


if __name__ == '__main__':
    # epoches = 100     # | change by parser

    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: device = torch.device("cpu")

    # --------------------------------------------

    train_trans = Compose([
        RandomAffine(  # Data augmentation about : rotation - shear - zoom
            degrees=(-5, 5), # rotation random from -5 to 5 deg
            translate=(0.05, 0.05),  # move random 5 per by hori and verti
            scale=(0.85, 1.15),  # zoom to 1.15 and zoom nho 0.85
            shear=5
        ),
        Resize((args.image_size,args.image_size)),
        ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.25,
            hue=0.1
        ),
        ToTensor(),
    ])

    test_trans = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])


    # --------------------------------------------
    train_dataset = CovidData(
        root=args.root,
        train=True,
        transform=train_trans
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # Visualization image cause using Data augmentation

    # image, _ = train_dataset.__getitem__(20)
    # image = (torch.permute(image, (1, 2, 0))*255.).numpy().astype(np.uint8)  # s1: convert to BGR fix To tensor (cv2)  s2: convert tensor to numpy and astype(np.uint8) -> cause photos in opencv have to be int khong dau 8 bit
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Test image", image)
    # cv2.waitKey(0)
    # exit()

    # End -----------------

    test_dataset = CovidData(
        root=args.root,
        train=False,
        transform=test_trans

    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    # cuz logging after train then having many so we will xoa bot no di
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    # Create path to save model
    if not os.path.isdir(args.trained_model):  # check exit
        os.mkdir(args.trained_model)  # if not yet then create

    writer = SummaryWriter(args.logging)
    iters = len(train_loader)

    # Initialization
    model = Cnn(num_classes=args.num_classes).to(device)  # to(device) -> set cuda
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Load model when had checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    else:
        start_epoch = 0
        best_accuracy = 0
    # End load




    # Training
    for epoch in range(start_epoch, args.epochs):  # start_epoch in case: train continuous
        model.train()
        progress_bar = tqdm(train_loader, colour="green")  #cyan
        for iter, (images, labels) in enumerate(progress_bar):  # progress_bar change for train_loader
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            progress_bar.set_description("epoch: {}/{} | interation: {}/{} | Loss: {:.3f} ".format(epoch+1, args.epochs, iter+1, iters, loss))
            writer.add_scalar("Train/Loss", loss, epoch*iters + iter)  # history of loss
            # Backward
            optimizer.zero_grad()  # not storage gradient after backward
            loss.backward()
            optimizer.step()

        model.eval()
        all_prediction = list()
        all_labels = list()
        for iter, (images, labels) in enumerate(test_loader):
            all_labels.extend(labels)

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predict = model(images)

                indices = torch.argmax(predict, dim=1)
                all_prediction.extend(indices)
                loss = criterion(predict, labels)

        all_labels = [label.item() for label in all_labels]
        all_prediction = [predict.item() for predict in all_prediction]

        # Classification - Report
        report = classification_report(all_labels, all_prediction, target_names=test_dataset.categories)
        print(report)

        # Confusion matrix
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_prediction), class_names=test_dataset.categories, epoch=epoch)
        # test_dataset.categories : --> Extract list of classes
        # End

        accuracy = accuracy_score(all_labels, all_prediction)
        print("Epoch: {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)  # just epoch cuz in test : done every epoch then save

        # SAVING MODEL -------------------------------------

        # Save last model for tomorrow train
        checkpoint = {
            "epoch" : epoch+1,  # epoch+1 -> next epoch train
            "model" : model.state_dict(),  # Save wight and bias
            "optimizer" : optimizer.state_dict()  # train dung learning rate hom qua stop
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_model))  # if hom nay trained 20 epochs, ngay mai muon train 80 epochs then dung last

        # Save best accuracy
        if accuracy > best_accuracy:
            checkpoint = {
                "epoch": epoch + 1,  # epoch+1 -> next epoch train
                "best_accuracy": best_accuracy,
                "model": model.state_dict(),  # Save wight and bias
                "optimizer": optimizer.state_dict()  # train dung learning rate hom qua stop
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_model))  # using if wanna sent cho colleague or boss
            best_accuracy = accuracy










