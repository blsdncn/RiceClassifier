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


    arborio_list = list(data_fol.glob('Arborio/*'))
    basmati_list = list(data_fol.glob('Basmati/*'))
    ipsala_list = list(data_fol.glob('Ipsala/*'))
    jasmine_list = list(data_fol.glob('Jasmine/*'))
    karacadag_list = list(data_fol.glob('Karacadag/*'))

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


'''
Some basic SVM code
Does not work on the multiclass data as, might consider an SVM model for each category
Takes about 11 min to run on my machine

-James
'''

X_data, y_data = load_image_data()
X_train, X_test, y_train, y_test = split_data(X_data, y_data)

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
