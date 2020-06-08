import numpy as np
import pandas as pd
import matplotlib.image as mpimg    # For reading images
import matplotlib.pyplot as plt     # For showing images
import seaborn as sns
import sys

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

class ADABoost :
    def __init__(self, img_data, label_data):
        self.x_train = img_data
        self.y_train = label_data
        self.x_test = img_data
        self.y_test = label_data
        self.batch_size = 4
        self.epochs = 20
        self.weight_list = np.repeat(float(1/self.epochs), self.epochs)
        self.hypothesis_list = []
        self.hypothesis_weight_list = []
        
        print(self.weight_list)

    # Draw Model learning result
    def plot_learning_curves(self, history) :
        pd.DataFrame(history.history).plot(figsize = (8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()

    ################################# Distribute the ratio of training set and testing set #################################
    ################################# Construct CNN model #################################
    def CNN_model(self) :
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
        # ### zca_whitening           -- implementing zca-whitening
        # ### rotation_range          -- angle for image to rotate when data increasing
        # ### width_shift_range       -- amplitude of horizontal offset when data increasing
        # ### shear_range             -- strength of shear
        # ### zoom_range              -- amplitude of random zoom
        # ### horizontal_flip         -- implementing random horizontal rotation
        # ### fill_mode               -- method for out-of-bound poinst while transformin

        # datagen = ImageDataGenerator(
        #     zca_whitening = False,
        #     rotation_range = 40,
        #     width_shift_range = 0.2,
        #     height_shift_range = 0.2,
        #     shear_range = 0.2,
        #     zoom_range = 0.2,
        #     horizontal_flip = True,
        #     fill_mode = 'nearest'
        # )

        # # Import reforcement parameters for images
        # datagen.fit(x_train)
        # x_train = x_train / 255
        # y_train = y_train / 255

        file_name = str(self.epochs) + '_' + str(self.batch_size)

        # Add earlyStopping and Tensorboard 
        CB = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2)
        TB = keras.callbacks.TensorBoard(log_dir = './log' + "_" + file_name, histogram_freq = 1)

        history = model.fit(
            x = self.x_train, y = self.y_train,
            batch_size = self.batch_size, 
            epochs = self.epochs,
            validation_split = 0.2,
            callbacks = [TB, CB]
        )

        ## self.plot_learning_curves(history)

        return model
    
    # Implement ADABoost
    def adaboost(self) :
        for i in range(self.epochs) :
            new_model = self.CNN_model()
            self.hypothesis_list.append(new_model)

            y_pred = new_model.predict(self.x_test)
            error = 0
            count = 0

            for j in range(len(y_pred)) :
                if(np.argmax(y_pred) == np.argmax(self.y_test)) :
                    self.weight_list[j] = self.weight_list[j] * error / (1 - error)
                    count += 1
                else :
                    error += self.weight_list[j]

            score = count / len(y_pred)
            print(i + "-iteration: Accuracy is %.2f%s" % (score * 100, '%'))

            weight_sum = 0

            # Normalized weight list
            for k in range(len(self.weight_list)) :
                weight_sum += self.weight_list[k]
            
            print("Sum of weight list is ", weight_sum)

            for k in range(len(self.weight_list)) :
                self.weight_list[k] /= weight_sum

            new_hypothesis_weight = np.log(1 - error) / error

            print("New hypothesis weight is ", new_hypothesis_weight)

            self.hypothesis_weight_list.append(new_hypothesis_weight)
        
        # Choose best classifier
        assert len(self.hypothesis_list) == len(self.hypothesis_weight_list)

        max_performence = 0
        max_performence_index = -1

        for i in range(len(self.hypothesis_list)) :
            print(i + " -> hypothesis weight is %.2f" %(self.hypothesis_weight_list[i]))

            if (self.hypothesis_weight_list[i] > max_performence) :
                max_performence = self.hypothesis_weight_list[i]
                max_performence_index = i

        return self.hypothesis_list[max_performence_index]



