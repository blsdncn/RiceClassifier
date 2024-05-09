import pathlib

# For visualizations
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog


def load_image_data():

    data_fol = "./Rice_Image_Dataset"
    data_fol = pathlib.Path(data_fol)


    arborio_list = list(data_fol.glob('Arborio/*'))[:50]
    basmati_list = list(data_fol.glob('Basmati/*'))[:50]
    ipsala_list = list(data_fol.glob('Ipsala/*'))[:50]
    jasmine_list = list(data_fol.glob('Jasmine/*'))[:50]
    karacadag_list = list(data_fol.glob('Karacadag/*'))[:50]

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

    return X, y


def split_data(X_data, y_labels, train_split=.8):
    data_length = len(X_data)
    split_index = int(train_split * data_length)

    X_train, X_test = X_data[:split_index], X_data[split_index+1:]
    y_train, y_test = y_labels[:split_index], y_labels[split_index + 1:]

    return X_train, X_test, y_train, y_test



X_data, y_data = load_image_data()
X_train, X_test, y_train, y_test = split_data(X_data, y_data)

class_names = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']
class_models = []
categories = 2 # len(class_names)
for cat in range(categories):
    model = svm.SVC(kernel='linear')
    y_train_modified = list(map(lambda x: x if x == class_names[cat] else "", y_train))
    print(cat)
    print(y_train)
    print(y_train_modified)
    model.fit(X_train, y_train_modified)
    class_models.append(model)

for cat in range(categories):
    y_pred = model.predict(X_test)
    y_test_modfied = list(map(lambda x: x if x == class_names[cat] else "", y_test))
    accuracy = accuracy_score(y_pred, y_test_modfied)
    print("Accuracy:", accuracy)


