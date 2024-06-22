import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Flatten
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.backend import clear_session
import numpy as np

clear_session()
set_random_seed(42)
np.random.seed(42)

def trainmodel(X_train, y_train_ohe, epochs):
    # Define the model
    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(y_train_ohe.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train_ohe, epochs=epochs, verbose=1)

    return model

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