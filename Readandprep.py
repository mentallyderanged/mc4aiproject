from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# def function 1
def load_dataset(ds_path):
    folders = os.listdir(ds_path)
    label_mapping = {folder: i for i, folder in enumerate(folders)}
    X, y = [], []
    y_label = []

    # Read files in folder
    for folder in folders:
        files = os.listdir(os.path.join(ds_path, folder))
        for f in files:
            if f.endswith('.png'):
                img = Image.open(os.path.join(ds_path, folder, f))
                img = np.array(img)
                X.append(img)
                y.append(label_mapping[folder])
                y_label.append(folder)
            
    return X, y,y_label
            
# define function 2
def prep_dataset(X, y, testsize):
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, shuffle=True)

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train_ohe = to_categorical(y_train, num_classes=y.max() + 1)
    y_test_ohe = to_categorical(y_test, num_classes=y.max() + 1)

    return X_train, X_test, y_train_ohe, y_test_ohe