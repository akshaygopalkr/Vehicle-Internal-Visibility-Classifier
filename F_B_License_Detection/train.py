
from torch.cuda import is_available
from torch import manual_seed
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from resnet import ResBlock
from car_dataset import CarImageDataset

if __name__ == '__main__':

    # TODO: Change the file/directory for this
    dataset = CarImageDataset(
        csv_file='train_data.csv',
        img_dir='.\\images',
        transform=transforms.Compose([
            transforms.Resize(100)
        ])
    )

    # TODO: Change the sizes for these splits
    manual_seed(0)
    train_set, valid_set, test_set = random_split(dataset, [3, 1, 1])

    # 3 Separate Data Loaders for train, validation, and test
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    epochs = 5

    for e in range(epochs):
        train_loss = 0.0
        for batch, (image, label) in enumerate(train_loader):


            # Transfer Data to GPU if available
            if is_available():
                image, label = image.cuda(), label.cuda()