#!/usr/bin/env python
# coding: utf-8

# In[1]:


#important libiraires
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications.efficientnet import *
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.preprocessing.image import DirectoryIterator


# In[2]:


#CNN Model
image_dir = Path('C:\\Users\\lenovo\\OneDrive\\Desktop\\Images')
filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
image_df = pd.concat([filepaths, labels], axis=1)
image_df


# In[3]:


image_df = image_df.sample(frac=1).reset_index(drop = True)
image_df.head(15)


# In[4]:


fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7),subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[i]))
    ax.set_title(image_df.Label[i])
plt.tight_layout()
plt.show()


# In[5]:


train_set, test_set = train_test_split(image_df, test_size=0.3)


# In[6]:


train_set = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2, 
    zoom_range=0.2,
    validation_split=0.2
)

test_set = ImageDataGenerator(
    rescale=1./255
)


# In[7]:


cnn = Sequential()
cnn=keras.models.Sequential()
cnn.add(keras.layers.Conv2D(filters=32,kernel_size=3,
                            padding='valid',activation='relu',input_shape=(224,224,3)))
cnn.add(keras.layers.MaxPool2D(pool_size=2))
cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(units = 128 , activation='relu'))
cnn.add(keras.layers.Dropout(rate= 0.1, seed= 100))
cnn.add(keras.layers.Dense(units = 1 , activation='sigmoid'))
cnn.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics=['accuracy'])


# In[8]:


training_set = train_set.flow_from_directory(
        'C:\\Users\\lenovo\\OneDrive\\Desktop\\Images',
        target_size=(64,64),  
        batch_size=128,
        class_mode='binary',
        shuffle=True,)
test_set = test_set.flow_from_directory(
        'C:\\Users\\lenovo\\OneDrive\\Desktop\\Images',
        target_size=(64, 64), 
        batch_size=128,
        class_mode='binary',
        shuffle=False)


# In[9]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


model.fit_generator(
        training_set,
        steps_per_epoch=len(training_set),
        epochs=10,
        validation_data = test_set,
        validation_steps=len(test_set))


# In[11]:


loss, accuracy = model.evaluate(test_set)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[12]:


results = model.evaluate(test_set, verbose=0)
print("Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[13]:


# Get the predictions and ground truth labels
pred = model.predict(test_set)
labels = np.array(test_set.classes)
# Round the predictions to the nearest integer
pred = np.round(pred)
# Calculate the precision, recall, f1-score, and support
report = classification_report(labels, pred, output_dict=True)
# Print the results
for key in report:
    print(key, report[key])


# In[14]:


# Calculate the confusion matrix
cf_matrix = confusion_matrix(labels, pred)
# Print the confusion matrix
print(cf_matrix)


# In[15]:


#Calculate the confusion matrix
cf_matrix = confusion_matrix(labels, pred)
# Plot the confusion matrix as a heat map
plt.figure()
plt.imshow(cf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.colorbar()
plt.show()


# In[16]:


# Convert the DataFrame object to a list of strings.
filepaths = image_df.Filepath.tolist()

# Create a DirectoryIterator object for the test set.
test_set = DirectoryIterator('C:\\Users\\lenovo\\OneDrive\\Desktop\\Images', image_data_generator=test_set)
_
# Create a figure and axes.
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7),subplot_kw={'xticks': [], 'yticks': []})

# Iterate over the axes and display the images.
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(filepaths[i]))
   

# Tighten the layout.
plt.tight_layout()

# Show the figure.
plt.show()


# In[17]:


#ANN model
# Load the dataset and preprocess the images
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\lenovo\\OneDrive\\Desktop\\Images',
    validation_split=0.3,
    subset='training',
    seed=42,
    image_size=(224, 224),
    batch_size=32
)


# In[18]:


# Normalize pixel values of images
normalized_dataset = dataset.map(lambda x, y: (tf.image.per_image_standardization(x), y))


# In[19]:


# Split the dataset into training and testing sets
train_dataset = normalized_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\lenovo\\OneDrive\\Desktop\\Images',
    validation_split=0.3,
    subset='validation',
    seed=42,
    image_size=(224, 224),
    batch_size=32
)
test_dataset = test_dataset.map(lambda x, y: (tf.image.per_image_standardization(x), y))


# In[20]:


# Define and train the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])


# In[21]:


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(train_dataset, epochs=10)


# In[22]:


# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[23]:


# Make predictions on the test dataset
test_images = []
test_labels = []
for images, labels in test_dataset:
    test_images.extend(images.numpy())
    test_labels.extend(labels.numpy())

test_images = np.array(test_images)
test_labels = np.array(test_labels)

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)


# In[24]:


# Calculate evaluation metrics
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels)
f1 = f1_score(test_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[25]:


# Create a grid of images with their predicted labels
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

for i in range(16):
    axes[i].imshow(test_images[i])
    axes[i].set_title("Predicted: {}".format(predicted_labels[i]))
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




