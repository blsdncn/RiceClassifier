import numpy as np
import matplotlib.pyplot as plt
import os
import keras
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


def plotConvCurveFromHistory(history,save_path=os.path.join(os.path.dirname(__file__),'..','metric_images',"CS 549 Project CNN")):
# Plot training & validation loss values
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(os.path.join(save_path,'lossCurve.jpg'))


# Plot training & validation accuracy values
    plt.figure(figsize=(8, 4))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(os.path.join(save_path,'accuracyCurve.jpg'))


def getDataAndSplit(data_dir,verbose=False):
    dataset = keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels='inferred',  
        label_mode='categorical',  
        batch_size=32,  
        image_size=(256, 256),  
        shuffle=True,  
        seed=42,  
        interpolation='bilinear'  
    )
    dataset = dataset.map(lambda x,y: (x/255,y))
    train_size = int(len(dataset)*.7)
    val_size = int(len(dataset)*.2)
    test_size = int(len(dataset)*.1)
    if(verbose): print(f"train: {train_size} | val: {val_size} | test: {test_size}")
    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size+val_size).take(test_size)
    return train,val,test
