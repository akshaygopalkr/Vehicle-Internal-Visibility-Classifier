import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import torch
import pdb
import os

class CarImageDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.data_file = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):

        # TODO: Different for my PC
        img_path = self.img_dir + '\\' + self.data_file.iloc[idx, 0]
        image = cv2.imread(img_path)
        image = torch.tensor(image)
        image = torch.reshape(image, (3, image.size()[0], image.size()[1]))
        label = self.data_file.iloc[idx, 1]

        # Perform transformations on image if needed
        if self.transform:
            image = self.transform(image)

        return image, label
