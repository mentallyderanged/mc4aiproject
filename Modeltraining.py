import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Flatten
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.backend import clear_session
import numpy as np
#from tensorflow.keras.optimizers import Adam


clear_session()
set_random_seed(42)
np.random.seed(42)

def trainmodel(X_train, y_train_ohe, epochs):
    # Define the model
    #Optimizer
    #optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999 )
    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(y_train_ohe.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()

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