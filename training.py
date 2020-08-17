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


