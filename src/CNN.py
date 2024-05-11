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
from util import func_confusion_matrix
from tensorflow.keras.regularizers import l2 #adding regularization to address overfitting

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

data_iterator = dataset.as_numpy_iterator()
batch = data_iterator.next()
print(batch[0].shape)
print(batch[1])

dataset = dataset.map(lambda x,y: (x/255,y))

train_size = int(len(dataset)*.7)
val_size = int(len(dataset)*.2)
test_size = int(len(dataset)*.1)
print(f"train: {train_size} | val: {val_size} | test: {test_size}")
train = dataset.take(train_size)
val = dataset.skip(train_size).take(val_size)
test = dataset.skip(train_size+val_size).take(test_size)

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
    pkl.dump(history,file)
# Plot training & validation loss values
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()