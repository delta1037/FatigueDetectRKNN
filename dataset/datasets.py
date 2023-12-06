import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
import glob
import random
random.seed(128)


def get_name(file_name):
    return file_name.split("/")[-1].split(".")[0]


class Data(data.Dataset):
    def __init__(self, data_path, transform):
        print(data_path)
        self.transform = transform

        self.img_paths = []
        self.labels = []
        open_img_paths = sorted(glob.glob(os.path.join(data_path, './open/*')), key=get_name)
        self.img_paths.extend(open_img_paths)
        self.labels.extend([0] * len(open_img_paths))
        close_img_paths = sorted(glob.glob(os.path.join(data_path, './close/*')), key=get_name)
        self.img_paths.extend(close_img_paths)
        self.labels.extend([1] * len(close_img_paths))

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).resize((64, 64))
        return self.transform(img), self.labels[index]

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    Data("../data_eye/train", transform)

