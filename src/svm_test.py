import pathlib
import random
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog


def load_image_data():

    current_fold = os.path.dirname(__file__)
    data_fol = os.path.join(current_fold, '..', 'data', 'Rice_Image_Dataset')
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
    images = []

    for label, fold_path in rice_images.items():
        for image_path in fold_path:
            image = plt.imread(image_path)
            images.append(image)
            image = resize(image, (100,100))
            image = rgb2gray(image)
            features = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            X.append(features)
            y.append(label)

    # Shuffle indices
    indices = list(range(len(X)))
    random.shuffle((indices))
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    images = [images[i] for i in indices]


    return X, y, images


# Splits the data into a training and a testing set
def split_data(X_data, y_labels, images, train_split=.8):
    data_length = len(X_data)
    split_index = int(train_split * data_length)

    X_train, X_test = X_data[:split_index], X_data[split_index+1:]
    y_train, y_test = y_labels[:split_index], y_labels[split_index + 1:]
    test_images = images[split_index+1:]

    return X_train, X_test, y_train, y_test, test_images

# Takes in an array of SVM models and combines their predicitons to return a single set of multi-class predicitons
def combined_pred(X_test, models, names):
    predictons = []

    for sample in X_test:
        class_probs = [model.decision_function([sample]) for model in models]

        pred_class_ind = np.argmax(class_probs)
        pred_class = names[pred_class_ind]
        predictons.append(pred_class)

    return predictons


# function taken from HW 4
def func_confusion_matrix(y_test, y_pred):
    """ this function is used to calculate the confusion matrix and a set of metrics.
    INPUT:
        y_test, ground-truth lables;
        y_pred, predicted labels;
    OUTPUT:
        CM, confuction matrix
        acc, accuracy
        arrR[], per-class recall rate,
        arrP[], per-class prediction rate.
    """

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    unique_values = set(y_test)
    sorted(unique_values)
    num_classes = len(unique_values)
    unique_values = np.array(list(unique_values))  # change to array so can use indexes
    possible_string_dict = {}
    # make sure all values are 0 based, so can use built-in "zip" function
    if (issubclass(type(y_test[0]), np.integer)):  # if values are integers
        y_test_min = y_test.min()
        if (y_test_min != 0):  # if does not contain 0, reduce both test and pred by min value to get 0 based for both
            y_test = y_test - y_test_min;
            y_pred = y_pred - y_test_min;
    else:
        # assume values are strings, change to integers
        y_test_int = np.empty(len(y_test), dtype=int)
        y_pred_int = np.empty(len(y_pred), dtype=int)
        for index in range(0, num_classes):
            current_value = unique_values[index]
            possible_string_dict[index] = current_value
            y_test_int[y_test == current_value] = index
            y_pred_int[y_pred == current_value] = index
        y_test = y_test_int
        y_pred = y_pred_int

    ## your code for creating confusion matrix;
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for a, p in zip(y_test, y_pred):
        # print(conf_matrix)
        conf_matrix[a][p] += 1

    ## your code for calcuating acc;
    accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()

    ## your code for calcualting arrR and arrP;
    recall_array = np.empty(num_classes, dtype=float)
    precision_array = np.empty(num_classes, dtype=float)
    for index in range(0, num_classes):
        value = conf_matrix[index, index]
        recall_sum = conf_matrix[index, :].sum()
        precision_sum = conf_matrix[:, index].sum()
        recall_array[index] = value / recall_sum
        precision_array[index] = value / precision_sum

    return conf_matrix, accuracy, recall_array, precision_array

def make_and_train_OvO(X_train, y_train, X_test, y_test, C, kernel, subset_split_index = 0):
    class_names = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']
    class_models = []
    # Create a model for each class
    for cat in class_names:
        model = svm.SVC(C = C, kernel=kernel)
        y_train_modified = [1 if x == cat else 0 for x in y_train]
        if subset_split_index != 0:
            X_train = X_train[:subset_split_index]
            y_train_modified = y_train_modified[:subset_split_index]
            X_test = X_test[:subset_split_index]

        model.fit(X_train, y_train_modified)
        class_models.append(model)

    # Process all class models
    for index, cat in enumerate(class_names):
        model = class_models[index]
        y_pred = model.predict(X_test)

    # Combine the results of the class models
    comb_pred = combined_pred(X_test, class_models, class_names)
    return comb_pred


X_data, y_data, images = load_image_data()
X_train, X_test, y_train, y_test, images = split_data(X_data, y_data, images)


c_range = [.01, .5, 1, 2, 4]
kernel_range = ['linear', 'poly','sigmoid', 'rbf']
test_sample_size = 50
# Test various C values
c_accuracies = []
for c in c_range:
    model_pred = make_and_train_OvO(X_train, y_train, X_test, y_test, c, 'linear', subset_split_index = test_sample_size)
    c_accuracies.append(accuracy_score(model_pred, y_test[:test_sample_size]))

plt.plot(c_range, c_accuracies)
plt.title('SVM by C Values')
plt.xlabel('C Values')
plt.ylabel('Accuracy')
plt.show()

# Test various kernels
kernel_accuracies = []
for kernel in kernel_range:
    make_and_train_OvO(X_train, y_train, X_test, y_test, 1, kernel, subset_split_index = test_sample_size)
    kernel_accuracies.append(accuracy_score(model_pred, y_test[:test_sample_size]))

plt.plot(kernel_range, kernel_accuracies)
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.xticks(kernel_range)
plt.show()

best_c_index = np.argmax(c_accuracies)
best_kenel_index = np.argmax(kernel_accuracies)

# use the best found parameters for the final model
best_model_pred = make_and_train_OvO(X_train, y_train, X_test, y_test, c_range[best_c_index], kernel_range[best_kenel_index])


# Get confusion matrix statistics
conf_matrix, acc, rec_array, prec_array = func_confusion_matrix(y_test, best_model_pred)

print(f"Matrix:\n{conf_matrix}")
print(f"Accuracy: {acc}")
print(f"Recall : {rec_array}")
print(f"Precision : {prec_array}\n\n")

# Get the indices of non-matching elements
wrong_indices = []
for idx, y_true in enumerate(y_test):
    if y_true != best_model_pred[idx]:
        wrong_indices.append(idx)

# Print and display images for the non-matching elements
for idx in wrong_indices[:10]:
    print("Index:", idx)
    print("True Label:", y_test[idx])
    print("Predicted Label:", best_model_pred[idx])
    plt.imshow(images[idx], cmap='gray')
    plt.show()


