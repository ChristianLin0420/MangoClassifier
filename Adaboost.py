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

# Sklearn deep learning package
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import VotingClassifier

class ADABoost :
    def __init__(self, img_data, label_data):
        self.x_train = img_data
        self.y_train = label_data
        self.x_test = img_data
        self.y_test = []
        self.batch_size = 5
        self.epochs = 20
        self.estimator_count = 10
        self.weight_list = np.repeat(float(1/len(self.x_train)), len(self.x_train))
        self.hypothesis_list = []
        self.hypothesis_weight_list = []

        for i in range(len(label_data)) :
            self.y_test.append(np.argmax(label_data[i]))

    # Draw Model learning result
    def plot_learning_curves(self, history) :
        pd.DataFrame(history.history).plot(figsize = (8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()

    ################################# Construct CNN model #################################
    def CNN_model(self) :
        model = tf.keras.Sequential() 
        model._estimator_type = "classifier"

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

        return model

    def VotingClassifier_Tensorflow(self, classifier_list) :
        probability_result = []
        predict_result = []

        assert len(classifier_list) == len(self.hypothesis_weight_list)

        for i in range(len(classifier_list)) :
            new_y_pred = classifier_list[i].predict(self.x_test) * self.hypothesis_weight_list[i]
            probability_result.append(new_y_pred)
        
        for i in range(len(self.x_test)) :
            temp_result = [0, 0, 0]

            for j in range(len(classifier_list)) :
                temp_result = temp_result + probability_result[j][i]
            
            predict_result.append(temp_result)

        assert len(self.x_test) == len(predict_result)

        return predict_result
    
    # Implement ADABoost
    def adaboost(self) :
        for i in range(self.estimator_count) :
            new_model = self.CNN_model()
            self.hypothesis_list.append(new_model)

            y_pred = new_model.predict(self.x_test)

            error = 0
            error_count = 0
            total_error = 0

            for j in range(len(y_pred)) :
                total_error += self.weight_list[j]

                if(np.argmax(y_pred[j]) != self.y_test[j]) :
                    error += self.weight_list[j]
                    error_count += 1

            print("Total error is " + str(total_error) + ", current error is " + str(error)) 

            error /= total_error
            error_count /= len(y_pred)
            new_hypothesis_weight = np.log((1 - error) / error)

            print("error of the decision tree is : " + str(error_count))
            print("New hypothesis weight is " + str(new_hypothesis_weight))
            print(str(i) + "-iteration: Accuracy is %.2f%s" % (error * 100, '%'))

            # update training data weight
            for j in range(len(self.weight_list)) :
                if(np.argmax(y_pred[j] != self.y_test[j])) :
                    self.weight_list[j] *= np.exp(new_hypothesis_weight)

            self.hypothesis_weight_list.append(new_hypothesis_weight)
        
        # Choose best classifier
        assert len(self.hypothesis_list) == len(self.hypothesis_weight_list)
        assert len(self.hypothesis_list) == self.estimator_count

        max_performence = 0
        max_performence_index = -1

        for i in range(len(self.hypothesis_list)) :
            print(str(i) + " -> hypothesis weight is %.2f" %(self.hypothesis_weight_list[i]))

            if (self.hypothesis_weight_list[i] > max_performence) :
                max_performence = self.hypothesis_weight_list[i]
                max_performence_index = i

        # Create vote classifier 
        estimators_list = []

        for i in range(len(self.hypothesis_list)) :
            new_estimator = (str(i), self.hypothesis_list[i])
            estimators_list.append(new_estimator)

        assert len(estimators_list) == self.estimator_count

        y_pred = self.VotingClassifier_Tensorflow(self.hypothesis_list)

        return y_pred 



