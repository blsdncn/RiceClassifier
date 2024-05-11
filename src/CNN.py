import os
import numpy as np
import keras
import random
import pickle as pkl
import pandas
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from util import func_confusion_matrix, getDataAndSplit
from tensorflow.keras.regularizers import l2 #adding regularization to address overfitting

labelmap = {0:"Arborio",1:"Basmati",2:"Ipsala",3:"Jasmine",4:"Karacadag"}
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir,'..','data','Rice_Image_Dataset')
classes = [class_name for class_name in os.listdir(data_dir) if class_name]
classes.remove('Rice_Citation_Request.txt')
print(classes)
fig, axs = plt.subplots(len(classes), 3, figsize=(5, 1 * len(classes)))
for i, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)
    images = os.listdir(class_path)
    random_images = random.choices(images, k=3)  # Select 3 images for 3 columns
    for j in range(3):
        img_path = os.path.join(class_path, random_images[j])
        img = keras.utils.load_img(img_path)
        axs[i, j].imshow(img)
        axs[i, j].axis('off')
    
    # Add a title above each row
    axs[i, 1].set_title(class_name, fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
plt.show()

train, val, test = getDataAndSplit(data_dir,True)

model = Sequential()
model.add(Conv2D(16, (3,3),1,activation='relu',input_shape=(256,256,3), kernel_regularizer=l2(0.02)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu', kernel_regularizer=l2(0.02)))
model.add(MaxPooling2D())
model.add(Conv2D(16,(3,3),1,activation='relu', kernel_regularizer=l2(0.02)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5,activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
history = model.fit(train,
                validation_data=val,
                epochs=10)

models_dir = os.path.join(current_dir,'..','models','CNN')
os.makedirs(models_dir,exist_ok="True")
model.save(os.path.join(models_dir,'CNN_Model.keras'))
with open(os.path.join(models_dir,'cnnTrainHistory.pkl'),"wb") as file:
    pkl.dump(history.history,file)
