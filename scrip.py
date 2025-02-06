from dataset import CovidData
from model import Cnn
from torch.utils.data import DataLoader

from argparse import ArgumentParser  # using for change params on terminal instead of open scrip
from tqdm import tqdm # Show progress
from tqdm.autonotebook import tqdm # Show progress on jupiter notebook
from torch.utils.tensorboard import SummaryWriter  # we can see history of progress train (loss - accuracy ..)
import os
import shutil  # using for delete tensorboard

import torch.nn as nn
import torch
from sklearn.metrics import classification_report, accuracy_score
from torchvision.transforms.v2 import ToTensor, Resize, Compose


def get_args():  # using for change params on terminal instead of open scrip
    parser = ArgumentParser(description="Cnn Training")
    parser.add_argument("--root", "-r", type=str, default="/home/mrly/Documents/ai_vietnguyen/deeplearning/fptu/dap/lab_1/dataset_covid19_big", help="Root_Dataset_Covid19")

    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")  # "-e" -- type on terminal more quickly
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Number of batch_size")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Images size")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers (high with if strong pc)")
    parser.add_argument("--num_classes", type=int, default=4, help="num_classes (The number of class prediction)")  # important
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rate: Default: 1e-3")
    parser.add_argument("--momentum", type=int, default=0.90, help="Momentum: Default: 0.90")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="History training")
    parser.add_argument("--trained_model", "-t", type=str, default="trained_model", help="File_Trained")  # path where trained
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # epoches = 100     # | change by parser

    args = get_args()
    trans = Compose([
        Resize((args.image_size,args.image_size)),
        ToTensor()
    ])
    train_dataset = CovidData(
        root=args.root,
        train=True,
        transform=trans
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_dataset = CovidData(
        root=args.root,
        train=False,
        transform=trans

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
    model = Cnn(num_classes=args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Call Cuda
    if torch.cuda.is_available():
        model.cuda()

    best_accuracy = 0

    # Training
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour="green")  #cyan
        for iter, (images, labels) in enumerate(progress_bar):  # progress_bar change for train_loader
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

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

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                predict = model(images)

                indices = torch.argmax(predict, dim=1)
                all_prediction.extend(indices)  # if using append will be bug like :  a Tensor with 2 elements cannot be converted to Scalar
                loss = criterion(predict, labels)

        all_labels = [label.item() for label in all_labels]
        all_prediction = [predict.item() for predict in all_prediction]
        accuracy = accuracy_score(all_labels, all_prediction)
        print("Epoch: {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)  # just epoch cuz in test : done every epoch then save
        # SAVING MODEL
        # Save last
        torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_model))  # if hom nay trained 20 epochs, ngay mai muon train 80 epochs then dung last
        # Save best accuracy
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "{}/best_cnn.pt".format(args.trained_model))  # using if wanna sent cho colleague or boss
            best_accuracy = accuracy

            










