import pathlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

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
            image = resize(image, (100, 100))
            image = rgb2gray(image)
            features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            X.append(features)
            y.append(label)

    return X, y

def split_data(X_data, y_labels, train_split=.8):
    return train_test_split(X_data, y_labels, train_size=train_split, random_state=42)

X_data, y_data = load_image_data()
X_train, X_test, y_train, y_test = split_data(X_data, y_data)

# initializes and trains k classifier
k = 5  # Neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict test data
y_pred = knn.predict(X_test)

# Calculate the current accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

# Create a report based upon classification
print("The current classification:")
print(classification_report(y_test, y_pred))
