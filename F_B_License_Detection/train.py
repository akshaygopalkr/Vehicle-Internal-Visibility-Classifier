from torch.cuda import is_available
from torch import manual_seed, nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from resnet import ResBlock, ResNet18
from car_dataset import CarImageDataset
import torch


def save_model():

    # TODO: Change this for LISA
    path = ".\\model.pth"
    torch.save(model.state_dict(), path)


def train(criterion, optimizer, train_loader, valid_loader, model):

    epochs = 175
    best_valid_accuracy = 0.0

    for e in range(epochs):

        # Set model to train
        train_loss = 0.0
        model.train()
        running_train_accuracy = 0.0
        num_train_images = 0

        for batch, (image, labels) in enumerate(train_loader):

            # Transfer Data to GPU if available
            if is_available():
                image, labels = image.cuda(), labels.cuda()

            image = image.float()
            labels = labels.float()
            num_train_images += image.size()[0]

            # Clear gradients
            optimizer.zero_grad()

            # Run target through model and format the output
            target = model(image)
            target = target.view(image.size()[0])

            if is_available():
                target = target.cuda()

            # Find the loss for this batch
            loss = criterion(target, labels)

            # backpropagation
            loss.backward()

            # adjust parameters
            optimizer.step()

            # Calculate the # of correct predictions and training loss
            train_predictions = (target > 0.5).float()
            running_train_accuracy += (train_predictions == labels).float().sum()
            train_loss += loss.item()

        train_accuracy = 100 * running_train_accuracy / num_train_images
        num_images = 0

        with torch.no_grad():
            # Set model to eval
            valid_loss = 0.0
            running_valid_accuracy = 0.0
            model.eval()

            for batch, (image, labels) in enumerate(valid_loader):
                if is_available():
                    image, labels = image.cuda(), labels.cuda()

                image = image.float()
                labels = labels.float()
                num_images += image.size()[0]

                target = model(image)
                target = target.view(image.size()[0])

                if is_available():
                    target = target.cuda()

                loss = criterion(target, labels)
                valid_loss += loss.item()
                valid_predictions = (target > 0.5).float()
                running_valid_accuracy += (valid_predictions == labels).float().sum()

            valid_accuracy = 100 * running_valid_accuracy / num_images

        # Save the model if it improved validation accuracy
        if valid_accuracy > best_valid_accuracy:
            save_model()
            best_valid_accuracy = valid_accuracy

        print(
            f'Epoch {e + 1} \t\t Training Loss: {train_loss / num_train_images} \t\t Validation Loss: {valid_loss / num_images}')
        print('Training Accuracy = {}'.format(train_accuracy))
        print('Validation Accuracy = {}'.format(valid_accuracy))
        print('\n')

    print('Maximum validation accuracy = {}'.format(best_valid_accuracy))


# Function to test the model
def test(test_loader):

    # Load the model that we saved at the end of the training loop
    model = ResNet18(in_channels=3, resblock=ResBlock, outputs=100)
    path = "model.pth"
    model.load_state_dict(torch.load(path))

    if is_available():
        model = model.cuda()

    with torch.no_grad():
        num_images = 0
        running_test_accuracy = 0.0
        for batch, (image, labels) in enumerate(test_loader):
            if is_available():
                image, labels = image.cuda(), labels.cuda()

            image = image.float()
            labels = labels.float()
            num_images += image.size()[0]

            target = model(image)
            target = target.view(image.size()[0])

            if is_available():
                target = target.cuda()

            test_predictions = (target > 0.5).float()
            running_test_accuracy += (test_predictions == labels).float().sum()


        print('Accuracy of the model based on the test set of', num_images,
              'inputs is: %d %%' % (100 * running_test_accuracy / num_images))


if __name__ == '__main__':


    # TODO: img_dir is different for my PC
    dataset = CarImageDataset(
        csv_file='train_data.csv',
        img_dir='.\\carsforvisibilitypred',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
        ])
    )

    manual_seed(0)
    train_size = int(0.75 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])

    # 3 Separate Data Loaders for train, validation, and test
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    model = ResNet18(in_channels=3, resblock=ResBlock, outputs=100)

    if is_available():
        model = model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Send to train method
    train(criterion=criterion, optimizer=optimizer, train_loader=train_loader, valid_loader=valid_loader, model=model)

    # Evaluate best model on test set
    test(test_loader)
