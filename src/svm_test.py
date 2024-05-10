import pathlib

# For visualizations
import random

import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog


def load_image_data():

    data_fol = "./Rice_Image_Dataset"
    data_fol = pathlib.Path(data_fol)


    arborio_list = list(data_fol.glob('Arborio/*'))[:100]
    basmati_list = list(data_fol.glob('Basmati/*'))[:100]
    ipsala_list = list(data_fol.glob('Ipsala/*'))[:100]
    jasmine_list = list(data_fol.glob('Jasmine/*'))[:100]
    karacadag_list = list(data_fol.glob('Karacadag/*'))[:100]




    rice_images = {
        'arborio': arborio_list,
        'basmati': basmati_list,
        'ipsala': ipsala_list,
        'jasmine': jasmine_list,
        'karacadag': karacadag_list
    }

    X = []
    y = []

    for label, fold_path in rice_images.items():
        for image_path in fold_path:
            image = plt.imread(image_path)
            image = resize(image, (100,100))
            image = rgb2gray(image)
            features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            X.append(features)
            y.append(label)

    indices = list(range(len(X)))
    random.shuffle((indices))

    X = [X[i] for i in indices]
    y = [y[i] for i in indices]

    return X, y


def split_data(X_data, y_labels, train_split=.8):
    data_length = len(X_data)
    split_index = int(train_split * data_length)

    X_train, X_test = X_data[:split_index], X_data[split_index+1:]
    y_train, y_test = y_labels[:split_index], y_labels[split_index + 1:]

    return X_train, X_test, y_train, y_test


def combined_pred(X_test, models, names):
    predictons = []

    for sample in X_test:
        class_probs = [model.decision_function([sample]) for model in models]

        pred_class_ind = np.argmax(class_probs)
        pred_class = names[pred_class_ind]
        predictons.append(pred_class)

    return predictons

X_data, y_data = load_image_data()
X_train, X_test, y_train, y_test = split_data(X_data, y_data)

class_names = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']
class_models = []

for cat in class_names:
    model = svm.SVC(kernel='linear')
    y_train_modified = [1 if x == cat else 0 for x in y_train]
    print(cat)
    print(y_train)
    print(y_train_modified)
    model.fit(X_train, y_train_modified)
    class_models.append(model)


for index, cat in enumerate(class_names):
    model = class_models[index]
    y_pred = model.predict(X_test)
    y_test_modfied = [1 if x == cat else 0 for x in y_test]
    accuracy = accuracy_score(y_pred, y_test_modfied)
    print(f"Accuracy for class {cat}: {accuracy}")

comb_pred = combined_pred(X_test, class_models, class_names)
accuracy = accuracy_score(comb_pred, y_test)
print(f'Total accuracy: {accuracy}')
