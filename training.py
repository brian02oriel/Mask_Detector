import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

img_rows, img_cols = 200, 200

def creating_dataset(data_path, label):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, 0)

        Training_Data.append(images)
        Labels.append(label)
    return Training_Data, Labels

positive_data, positive_labels = creating_dataset('./dataset/positive/', 1)
negative_data, negative_labels = creating_dataset('./dataset/negative/', 0)
X = np.asarray(positive_data + negative_data)
y = np.asarray(positive_labels + negative_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# Import Necessary Keras Libraries
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D

# Define model parameters
num_classes = 2
batch_size = 32

# Build CNN model using Sequential API
model = Sequential()

# First layer group containing Convolution, Relu and MaxPooling layers
model.add(Conv2D(64, (3,3), input_shape=(100, 100, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second layer group containing Convolution, Relu and MaxPooling layers
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Dropout Layer to stack the output convolutions above as well as cater overfitting
model.add(Flatten())
model.add(Dropout(0.5))

# Softmax Classifier
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

# Plot the model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='face_mask_detection_architecture.png')

# Train the model
from keras.optimizers import Adam
epochs = 50

model.compile(loss= 'categorical_crossentropy', optimizer= Adam(lr=0.001), metrics= ['accuracy'])

fitted_model = model.fit(np.asarray(X_train), np.asarray(y_train), epochs= epochs, validation_split= 0.25)
