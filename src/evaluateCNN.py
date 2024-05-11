from matplotlib import pyplot as plt
from util import func_confusion_matrix, plotConvCurveFromHistory, getDataAndSplit
import os
import pickle as pkl
import keras 
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import random


labelmap = {0:"Arborio",1:"Basmati",2:"Ipsala",3:"Jasmine",4:"Karacadag"}
model_dir = os.path.join(os.path.dirname(__file__),'..','models','CNN')
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir,'..','data','Rice_Image_Dataset')
history_path = os.path.join(model_dir,'cnnTrainHistory.pkl')
model_path = os.path.join(model_dir,'CNN_Model.keras')

_,_,test = getDataAndSplit(data_dir,verbose=True)

with open(history_path,"rb") as file:
    history = pkl.load(file)

model = keras.models.load_model(model_path)

plotConvCurveFromHistory(history.history)
# Lists for all predictions and true labels
y_true = []
y_pred = []
x_misclassified = []
y_true_misclassified = []
y_pred_misclassified = []

for i, batch in enumerate(test.as_numpy_iterator()):
    print(f"Processing batch {i+1}...", end='\r', flush=True)
    X, y = batch
    pred = model.predict(X, verbose=0)
    y_true_batch = np.argmax(y, axis=1)
    y_pred_batch = np.argmax(pred, axis=1)
    
    # Extend the full lists of predictions and true labels
    y_true.extend(y_true_batch)
    y_pred.extend(y_pred_batch)
    
    # Identify and store misclassifications
    misclassified_indices = np.where(y_true_batch != y_pred_batch)[0]
    for idx in misclassified_indices:
        x_misclassified.append(X[idx])
        y_true_misclassified.append(y_true_batch[idx])
        y_pred_misclassified.append(y_pred_batch[idx])
print("Done")

# Now, calculate your confusion matrix and other statistics
conf_matrix, acc, rec_array, prec_array = func_confusion_matrix(y_true, y_pred)
print(f"Matrix:\n{conf_matrix}")
print(f"Accuracy: {acc}")
print(f"Recall: {rec_array}, mean = {np.mean(rec_array)}")
print(f"Precision: {prec_array}, mean = {np.mean(prec_array)}\n\n")

# Display misclassified images
num_images_to_display = min(5, len(x_misclassified))
fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))
if num_images_to_display == 1:
    axes = [axes]

for i in range(num_images_to_display):
    img = x_misclassified[i] * 255
    img = img.astype(np.uint8)
    ax = axes[i]
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    title_text = f'Predicted: {labelmap[y_pred_misclassified[i]]}\n True: {labelmap[y_true_misclassified[i]]}'
    ax.set_title(title_text, fontsize=10)

plt.show()
