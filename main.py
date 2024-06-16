from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Flatten
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.backend import clear_session

clear_session()
set_random_seed(42)
np.random.seed(42)

ds_path = input("Enter the path to the sample dataset directory: ").strip()
testsize = float(input("Enter the test size (between 0 and 1): ").strip())
epoch = int(input("Enter the number of epochs: ").strip())

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

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, shuffle=True)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training data shape: ", X_train.shape)

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

print(y_train.shape, y_train_ohe.shape, y_test.shape, y_test_ohe.shape)

# Define the model
model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(Flatten())
model.add(Dense(y.max()+1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_ohe, epochs = epoch, verbose=1)

# test
# plt.figure(figsize=(8,4))
# 
# plt.subplot(1,2,1)
# plt.title('Loss')
# plt.xlabel('Epochs')
# plt.plot(history.history['loss'])
# 
# plt.subplot(1,2,2)
# plt.title('Accuracy')
# plt.xlabel('Epochs')
# plt.plot(history.history['accuracy'])
# 
# plt.show()
