# Vehicle Internal Classifier
* This github repository contains code inside the `F_B_License_Detection` folder to train a model to classify whether a specific vehicle internal (front left light, front right light, rear left light, rear right light, rear license plate, front license plate) is visible or not. All the installation requirements are in the provided folder.

## Downloading Dataset
* To train and label data for vehicle internal visibility classification, we use a dataset of detected cars from the ApolloCar3D Dataset [1]. To download this data use this link: https://drive.google.com/file/d/1oMPV9ov5tucrNUJyM_dah-2brTQF9WGc/view?usp=sharing. Once the data is downloaded, put it inside the `F_B_License_Detection` folder. 

## Adding Training Data
* If you want to add visibility annotations to a specific vehicle internal, run the `generate_data.py` file.
  * This file will ask you first what specific vehicle internal you want to add visibility labels too.
  * Once the vehicle internal of interest is chosen, a vehicle a specific image will appear. Press 1 if the vehicle internal is present, 0 if it isn't present, 'n' if you would like to skip the image, and 'q' if you want to end annotating.

 ## Training Visibility Models
 * There are multiple files that can be used to train the visibility classifier:
   * `train_pt_resnet18.py`: Fine-tuning a pretrained ResNet-18 for visibility classificiation.
   * `train_pt_resnet34.py`: Fine-tuning a pretrained ResNet-34 for visibility classificiation.
   * `train_pt_resnet50.py`: Fine-tuning a pretrained ResNet-50 for visibility classificiation.
   * `train_ut_resnet18.py`: Training a ResNet-18 from scratch for visibility classification.
 * Once you run one of these files, you will be able to choose what light of interest you would like train a model for the visibility classification

## References
[1] Song, Xibin, et al. "Apollocar3d: A large 3d car instance understanding benchmark for autonomous driving." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
