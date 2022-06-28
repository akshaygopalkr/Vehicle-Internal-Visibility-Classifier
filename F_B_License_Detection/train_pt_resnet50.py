from torch.cuda import is_available
from torch import manual_seed, nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models
from F_B_License_Detection.car_dataset import CarImageDataset
import torch


def save_model(file_folder):

    # TODO: Change this for LISA
    path = file_folder + "/pt_resnet50.pth"
    torch.save(model.state_dict(), path)


def train(criterion, optimizer, train_loader, valid_loader, model, file_folder):
    epochs = 100
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
            save_model(file_folder=file_folder)
            best_valid_accuracy = valid_accuracy

        print(
            f'Epoch {e + 1} \t\t Training Loss: {train_loss / num_train_images} \t\t Validation Loss: {valid_loss / num_images}')
        print('Training Accuracy = {}'.format(train_accuracy))
        print('Validation Accuracy = {}'.format(valid_accuracy))
        print('\n')

    print('Maximum validation accuracy = {}'.format(best_valid_accuracy))


# Function to test the model
def test(test_loader, file_folder):

    # Load the model that we saved at the end of the training loop
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=2048, out_features=1),
        torch.nn.Sigmoid()
    )

    # TODO: Different path for LISA Computers
    path = file_folder + "/pt_resnet50.pth"
    model.load_state_dict(torch.load(path))

    if is_available():
        model = model.cuda()

    with torch.no_grad():
        num_images = 0
        running_test_accuracy = 0.0
        running_tp = 0.0
        running_tn = 0.0
        running_fp = 0.0
        running_fn = 0.0
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
            running_test_accuracy += (test_predictions == labels).float().sum().item()

            pred_list = test_predictions.tolist()
            labels_list = labels.tolist()
            running_tp += sum(pred_list[i] == 1 and labels_list[i] == 1 for i in range(len(pred_list)))
            running_tn += sum(pred_list[i] == 0 and labels_list[i] == 0 for i in range(len(pred_list)))
            running_fp += sum(pred_list[i] == 1 and labels_list[i] == 0 for i in range(len(pred_list)))
            running_fn += sum(pred_list[i] == 0 and labels_list[i] == 1 for i in range(len(pred_list)))



        print('Accuracy of the model based on the test set of', num_images,
              'inputs is: %', (100 * running_test_accuracy / num_images))
        print('True Positive Rate: %', (100*running_tp/num_images))
        print('True Negative Rate: %', (100 * running_tn / num_images))
        print('False Positive Rate: %', (100 * running_fp / num_images))
        print('False Negative Rate: %', (100 * running_fn / num_images))



if __name__ == '__main__':

    file_num = 'a'

    # TODO: Change file brackets for this part
    file_dict = {1: './rear_plate/rear_plate_data.csv', 2: './front_plate/front_plate_data.csv',
                 3: './rear_L_light/rear_L_light_data.csv', 4: './rear_R_light/rear_R_light_data.csv',
                 5: './front_L_light/front_L_light_data.csv', 6: './front_R_light/front_R_light_data.csv'
                 }

    while not file_num.isdigit() or (file_num.isdigit() and not 1 <= int(file_num) <= 6):
        file_num = input('Enter a number for class to train on: \n 1. Rear Plate 2. Front Plate 3. Left-Rear Light '
                         '\n 4. Right-Rear Light 5. Left-Front Light 6. Right-Front Light\n')

    file_path = file_dict[int(file_num)]

    # TODO: Change for LISA
    file_folder = './' + file_path.split('/')[1]

    # TODO: img_dir is different for my PC
    dataset = CarImageDataset(
        csv_file=file_path,
        img_dir='./carsforvisibilitypred',
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    )

    manual_seed(0)
    train_size = int(0.75 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])

    # 3 Separate Data Loaders for train, validation, and test
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=2048, out_features=1),
        torch.nn.Sigmoid()
    )

    if is_available():
        model = model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Send to train method
    train(criterion=criterion, optimizer=optimizer, train_loader=train_loader, valid_loader=valid_loader, model=model,
          file_folder=file_folder)

    # Evaluate best model on test set
    test(test_loader, file_folder=file_folder)
