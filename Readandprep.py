from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#def function 1: load_dataset
def load_dataset():
    ds_path = input("Enter the path to the sample dataset directory: ").strip()

    folders = os.listdir(ds_path)
    label_mapping = {folder: i for i, folder in enumerate(folders)}
    X, y = [], []

    # Read files in folder
    for folder in folders:
        files = os.listdir(os.path.join(ds_path, folder))
        for f in files:
            if f.endswith('.png'):
                img = Image.open(os.path.join(ds_path, folder, f))
                img = np.array(img)
                X.append(img)
                y.append(label_mapping[folder])
            
    return X, y
            
# define function 2: prepare dataset
def prep_dataset(X,y):
    testsize = float(input("Enter the test size (between 0 and 1): ").strip())
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize,   shuffle=True)

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0


    # Optional visualization of some samples
    # fig, axs = plt.subplots(10, 10)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # for i in range(10):
    #     ids = np.where(y_train == i)[0]
    #     for j in range(10):
    #         target = np.random.choice(ids)
    #         axs[i][j].axis('off')
    #         axs[i][j].imshow(X_train[target], cmap='gray')
    # plt.show()

    y_train_ohe = to_categorical(y_train, num_classes=y.max() + 1)
    y_test_ohe = to_categorical(y_test, num_classes=y.max() + 1)

    # print(y_train.shape, y_train_ohe.shape, y_test.shape, y_test_ohe.shape)
    return X_train, X_test, y_train_ohe, y_test_ohe

prep_dataset(*load_dataset())