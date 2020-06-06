import os
import cv2
import csv
import random
import time
import numpy as np
import pandas as pd
import matplotlib.image as mpimg    # For reading images
import matplotlib.pyplot as plt     # For showing images
import seaborn as sns

# Keras deep learning package
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import utils as np_utils
from keras import backend as K
from keras import optimizers

# Tensorflow deep learning package
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


# Show package version
print(pd.__version__)
print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
################################# Read data set label #################################
Sample_label = pd.read_csv("Sample_Label.csv", encoding = "utf8")

# Show data set label
Sample_label.head()

# Connect images folder
Sample_pics_path = os.path.join("sample_image")
train_mango_fnames = os.listdir(Sample_pics_path)

# print(train_mango_fnames[0])
# print(train_mango_fnames[1])
# print(train_mango_fnames[2])

# For browsering label file
label_Survey = pd.read_csv("Sample_SurveyLabel.csv", encoding = "utf8")
label_Survey.head()

sns.countplot(label_Survey['Level'], hue = label_Survey["Level"])

################################# Create label and data set #################################
csvfile = open('Sample_Label.csv')
reader = csv.reader(csvfile)

labels = []
for line in reader:
    tmp = [line[0], line[1]]
    labels.append(tmp)

csvfile.close()

picnum = len(labels)
print("Number of mango images : ", picnum)

X = []
Y = []

# Switch images' label
for i in range(len(labels)) :
    labels[i][1] = labels[i][1].replace("等級A", "0")
    labels[i][1] = labels[i][1].replace("等級B", "1")
    labels[i][1] = labels[i][1].replace("等級C", "2")

a = 0
items = []

for a in range(0, picnum) :
    items.append(a)

for i in random.sample(items, picnum) :
    img = cv2.imread("sample_image/" + labels[i][0])
    res = cv2.resize(img, (800, 800), interpolation = cv2.INTER_LINEAR)
    res = img_to_array(res)
    X.append(res)
    Y.append(labels[i][1])

y_label_org = Y

assert len(X) == len(Y)
    
X = np.array(X)
Y = np.array(Y)

# Convert data set and label into float type
for i in range(len(X)) :
    X[i] = X[i].astype('float32')

Y = tf.strings.to_number(Y, out_type = tf.float32)

# Implementing one-hotencoding
Y = np_utils.to_categorical(Y, num_classes = 3)

################################# Distribute the ratio of training set and testing set #################################
x_train = X[:84]
y_train = Y[:84]
x_test = X[84:]
y_test = Y[84:]

# for i in range(0, picnum) :
#     print(X[i])
#     print(Y[i])

y_train_label = [0., 0., 0.]
for i in range(0, len(y_train)) :
    y_train_label = y_train[i] + y_train_label

y_test_label = [0., 0., 0.]
for i in range(0, len(y_test)) :
    # print(y_test[i])
    y_test_label = y_test[i] + y_test_label

# print(y_train_label)
# print(y_test_label)

################################# Construct CNN model #################################
model = tf.keras.Sequential()
model.add(layers.Conv2D(16, (3, 3),
                 strides = (1, 1),
                 input_shape = (800, 800, 3),
                 padding = 'same',
                 activation = 'relu',
                 ))

model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = None))

model.add(layers.Conv2D(32, (3, 3),
                 strides = (1, 1),
                 padding = 'same',
                 activation = 'relu',
                 ))

model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = None))

model.add(layers.Conv2D(64, (3, 3),
                 strides = (1, 1),
                 padding = 'same',
                 activation = 'relu',
                 ))

model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = None))
model.add(layers.Flatten())

model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(128, activation = 'relu'))

model.add(layers.Dropout(0.2))
model.add(layers.Dense(3, activation = 'softmax'))
model.summary()

# Add optimizer 
adam = optimizers.adam(lr = 5)
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['acc'])

## parameters explaniation ##
### zca_whitening           -- implementing zca-whitening
### rotation_range          -- angle for image to rotate when data increasing
### width_shift_range       -- amplitude of horizontal offset when data increasing
### shear_range             -- strength of shear
### zoom_range              -- amplitude of random zoom
### horizontal_flip         -- implementing random horizontal rotation
### fill_mode               -- method for out-of-bound poinst while transformin

datagen = ImageDataGenerator(
    zca_whitening = False,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

# Import reforcement parameters for images
datagen.fit(x_train)
x_train = x_train / 255
y_train = y_train / 255

# Set HyperParameters
batch_size = 4
epochs = 10

file_name = str(epochs) + '_' + str(batch_size)

# Add earlyStopping and Tensorboard 
CB = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2)
TB = keras.callbacks.TensorBoard(log_dir = './log' + "_" + file_name, histogram_freq = 1)

history = model.fit(
    x = x_train, y = y_train,
    batch_size = batch_size, 
    epochs = epochs,
    validation_split = 0.2,
    callbacks = [CB]
)

################################# Draw Model learning result #################################
def plot_learning_curves(history) :
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)
