# CRNN - Text Recognition of Historical Document

## Overview
This project uses a Convolutional Recurrent Neural Network (CRNN) to recognize handwritten historical text. It loads a dataset containing images of handwritten text and corresponding transcriptions, preprocesses the data, and trains a CRNN model to identify the text.

## Dataset
The Rodrigo Corpus dataset is used in this project. It consists of images of handwritten historical text stored in the 'images' folder and transcriptions stored in a text file named 'transcriptions.txt'. The dataset is divided into training, validation, and testing partitions, each specified in separate text files.

It can be downloaded from [Link](https://zenodo.org/records/1490009)

## Preprocessing
- Images are loaded and preprocessed, including binarization, noise reduction, and resizing.
- Transcriptions are loaded and mapped to numerical representations.
- Data is split into training and validation sets.

## Model
The CRNN model is built using TensorFlow/Keras:
- Convolutional layers extract features from input images.
- Recurrent layers (LSTM) capture temporal dependencies in the sequences.
- TimeDistributed layer is used to apply operations to each time step independently.
- The model outputs probabilities for each character in the sequence.

## Hyperparameter Tuning
Hyperparameters such as convolutional filters, LSTM units, dropout rate, kernel size, and learning rate are tuned using RandomSearch from Keras Tuner.

## Training
The model is trained on the training set with a validation split. Training progress and performance metrics are monitored.

## Evaluation
Model performance is evaluated on the validation set using accuracy and loss metrics.

## Usage
1. Set the path to your dataset.
2. Initialize a DataLoader object and load the dataset.
3. Load images and transcriptions.
4. Preprocess images and encode transcriptions.
5. Build and compile the CRNN model.
6. Use Keras Tuner to search for optimal hyperparameters.
7. Train the model and monitor training progress.
8. Evaluate model performance on the validation set.
9. Use the trained model for handwritten text recognition.

## Dependencies
- NumPy
- OpenCV (cv2)
- TensorFlow/Keras
- sci-kit-learn
- Keras Tuner
