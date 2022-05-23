import cv2
import csv
import os
import pandas as pd
import random


def update_file(file_name, new_train_data):
    df = pd.DataFrame(new_train_data)
    df.to_csv(file_name, mode='a', header=False, index=False)


def add_data(img_folder, train_data):
    new_train_data = {'Image': [], 'Label': []}

    image_list = os.listdir(img_folder)
    random.shuffle(image_list)
    image_idx = 0
    user_in = ''

    while image_idx < len(image_list):

        # if the image has already been looked at then skip it
        if image_list[image_idx] in train_data['Image']:
            image_idx += 1
            continue

        path = img_folder + image_list[image_idx]
        img = cv2.imread(path)
        # img = cv2.resize(img, (500, 500))
        cv2.imshow('Image', img)
        user_in = chr(cv2.waitKey(0))

        # In case user gives invalid input
        while user_in != '1' and user_in != '0' and user_in != 'q' and user_in != 'n':
            cv2.imshow('Image', img)
            user_in = chr(cv2.waitKey(0))

        if user_in == 'q':
            break
        elif user_in == 'n':
            pass

        # Use user input to make label
        label = 0 if user_in == '0' else 1
        new_train_data['Image'].append(image_list[image_idx])
        new_train_data['Label'].append(label)
        image_idx += 1
    return new_train_data


def load_data(file_name):
    # Load data from csv
    tr_data = {'Image': [], 'Label': []}

    with open(file_name) as inp:
        reader = csv.reader(inp)

        for row in reader:
            tr_data['Image'].append(row[0])
            tr_data['Label'].append(row[1])

    return tr_data


if __name__ == '__main__':
    print('-------------------------INSTRUCTIONS---------------------------')
    print('DON\'T CLICK EXIT BUTTON ON IMAGE WINDOW TO STOP LABELING, JUST press \'q\' ')
    print(
        'Note: Some of the images may have a grey box over the license plate. \n If you see this still press 1 as a '
        'positive detection.')
    print('1 -> Rear license plate in photo')
    print('0 -> Rear license plate not in photo')
    print('n -> Vehicle is not very visible so don\'t include in training set')
    print('q -> quit')

    # TODO: These will be different on LISA computers
    train_data = load_data('train_data.csv')
    new_data = add_data('./carsforvisibilitypred/', train_data)
    update_file('train_data.csv', new_data)
