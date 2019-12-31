from sklearn import neighbors, linear_model
from sklearn.externals import joblib
import os
import cv2
import numpy as np

# image to feature 
def image_to_feature(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    img = np.array(img) / 10
    img[img < 20] = 0
    img = img.astype("uint8")
    return img.reshape(1, -1)[0]

# To load images to features and labels
def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)        
        feature = image_to_feature(image_file_name)
        features_data.append(feature)
        label_data.append(image_label)

    return features_data, label_data

# Load your own images to training and test data
X_train = []
y_train = []

X_train, y_train = load_images_to_data('0', '.\\Training\\data\\0', X_train, y_train)
X_train, y_train = load_images_to_data('1', '.\\Training\\data\\1', X_train, y_train)
X_train, y_train = load_images_to_data('2', '.\\Training\\data\\2', X_train, y_train)
X_train, y_train = load_images_to_data('3', '.\\Training\\data\\3', X_train, y_train)
X_train, y_train = load_images_to_data('4', '.\\Training\\data\\4', X_train, y_train)
X_train, y_train = load_images_to_data('5', '.\\Training\\data\\5', X_train, y_train)
X_train, y_train = load_images_to_data('6', '.\\Training\\data\\6', X_train, y_train)
X_train, y_train = load_images_to_data('7', '.\\Training\\data\\7', X_train, y_train)
X_train, y_train = load_images_to_data('8', '.\\Training\\data\\8', X_train, y_train)
X_train, y_train = load_images_to_data('9', '.\\Training\\data\\9', X_train, y_train)

classifier = neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto', p=2, metric='minkowski', metric_params=None, n_jobs=1)
classifier.fit(X_train, y_train)

joblib.dump(classifier, ".\\model.pkl", compress=3)

## Below code is for testing the model
model = joblib.load(".\\model.pkl")
# TEST 1
input = image_to_feature('.\\Training\\data\\1\\1_1.png')
prediction = model.predict([input])[0]
print(prediction)

#TEST 2
input = image_to_feature('.\\Training\\data\\9\\9_6.png')
prediction = model.predict([input])[0]
print(prediction)