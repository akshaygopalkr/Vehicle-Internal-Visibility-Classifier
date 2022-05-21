import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image


class CarImageDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.data_file = pd.read_csv(csv_file, header = None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):


        img_path = self.img_dir + '\\' + self.data_file.iloc[idx, 0]
        image = read_image(img_path)
        label = self.data_file.iloc[idx, 1]

        # Perform transformations on image if needed
        if self.transform:
            image = self.transform(image)

        return image, label