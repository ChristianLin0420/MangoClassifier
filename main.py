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

# External files
from Adaboost import ADABoost

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

# sns.countplot(label_Survey['Level'], hue = label_Survey["Level"])

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

################################# ADABoost #################################
x_train = X[:94]
y_train = Y[:94]
x_test = X[:94]
y_test = Y[:94]


model = ADABoost(x_train, y_train).adaboost()

################################# Predict images #################################
# test_mango_dir = os.path.join("test_image")
# test_mango_fnames = os.listdir(test_mango_dir)

# img_files = [os.path.join(test_mango_dir, f) for f in test_mango_fnames]
# img_path = random.choice(img_files)

# # Read the testing image and show it
# img = load_img(img_path, target_size = (800, 800)) # This is a PIL image
# plt.title(img_path)
# plt.grid(False)
# plt.imshow(img)

# labels = ['等級A', '等級B', '等級C']

# # Convert image to analysible format for model (800 x 800 x 3, float32)
# x = img_to_array(img)
# x = x.reshape((1,) + x.shape)
# x /= 255

# start = time.time()
# result = model.predict(x)
# finish = time.time()

# pred = result.argmax(axis = 1)[0]
# pred_prob = result[0][pred]

# print("Result = %f" %pred_prob)
# print("test time : %f second" %(finish - start))
# print("Has {: .2f}% probabilty is {}" .format(pred_prob * 100, labels[pred]))

################################# Accuray of the model to predict training set #################################

y_pred = model.predict(x_test)

count = 0
for i in range(len(y_pred)) :
    if(np.argmax(y_pred[i]) == np.argmax(y_test[i])) :
        count += 1
    
score = count / len(y_pred)
print("Accuray is %.2f%s" % (score * 100, '%'))

# Label after predicting
predict_label = np.argmax(y_pred, axis = 1)
print(predict_label)
print(len(predict_label))

true_label = y_label_org[84:]
true_label = np.array(true_label)
print(true_label)
print(len(true_label))

predictions = model.predict_classes(x_test)
print(predictions)
print(len(predictions))

pd.crosstab(true_label,predict_label,rownames=['Actual value'],colnames=['Predicted value'])