import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow.keras
import cv2
import os
import glob
from tensorflow.keras import utils
# Import of keras model and hidden layers for our convolutional network
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout

dir_path = "/Volumes/GoogleDrive/My Drive/Projects/RPS_Data/Image_Data"
categories = ['none', 'paper', 'replay', 'rock', 'scissors']
image_data = []
for cat in categories:
    path = os.path.join(dir_path, cat)
    for img in glob.glob(path + "/*.jpg"):
        img_arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (227, 227))
        image_data.append([img_arr, categories.index(cat)])

random.shuffle(image_data)
input_data = []
label = []

for X, y in image_data:
    input_data.append(X)
    label.append(y)

plt.figure(1, figsize=(15, 10))
for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.imshow(image_data[i][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(categories[label[i]])
plt.show()

input_data = np.array(input_data)
label = np.array(label)
input_data = input_data / 255.0
input_data.shape

label = utils.to_categorical(label, num_classes=5, dtype='i1')

input_data.shape = (-1, 227, 227, 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.2, random_state=0)

model = tensorflow.keras.models.Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(227, 227, 1)))
model.add(Activation('relu'))

model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(454, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.3)

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: {:2.2f}%'.format(test_accuracy * 100))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cat = [c for c in categories]
plt.figure(figsize=(10, 10))
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
sns.heatmap(cm, annot=True, xticklabels=cat, yticklabels=cat)
plt.show()

##Saving and Loading model for prediction
model.save(dir_path + "/cnnmodel.h5")

from tensorflow.keras.models import load_model

model = load_model(dir_path + "/cnnmodel.h5")

categories = ['none', 'paper', 'replay', 'rock', 'scissors']

x = cv2.imread(dir_path+"/paper/992_mirrored.jpg", cv2.IMREAD_GRAYSCALE)
x = cv2.resize(x, (227, 227))
x = np.array(x) / 255.0
x.shape = (-1, 227, 227, 1)
model.predict(x)
