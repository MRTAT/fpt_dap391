from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms.v2 import Resize, Compose, ToTensor

class CovidData(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train: mode = "train"
        else: mode = "test"
        root = os.path.join(root, mode)
        # print(sorted(os.listdir(root)))

        self.transform = transform

        self.image_paths = list()
        self.labels = list()
        self.categories = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(root, category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name)
                self.image_paths.append(file_path)
                self.labels.append(i)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    root = "/home/mrly/Documents/ai_vietnguyen/deeplearning/fptu/dap/lab_1/dataset_covid19_big"

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    dataset = CovidData(root, train=True, transform=transform)
    image, label = dataset.__getitem__(88)
    print(dataset.__len__())

