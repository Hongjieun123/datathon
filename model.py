#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import PIL.Image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import concatenate, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import PIL.Image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import layers, Sequential, Model, Input
from tensorflow.keras.layers import concatenate, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping 
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt


# In[4]:


def network(X_train, y_train, X_test, y_test, modelpath, csv_path):
    
    im_shape=(X_train.shape[1], X_train.shape[-1])
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Conv1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    conv2_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool1)
    conv2_1=BatchNormalization()(conv2_1)
    pool2=MaxPool1D(pool_size=(2), strides=(2), padding='same')(conv2_1)
    conv3_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool2)
    conv3_1=BatchNormalization()(conv3_1)
    pool3=MaxPool1D(pool_size=(2), strides=(2), padding='same')(conv3_1)
    conv4_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool3)
    conv4_1=BatchNormalization()(conv4_1)
    pool4=MaxPool1D(pool_size=(2), strides=(2), padding='same')(conv4_1)
    conv5_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool4)
    conv5_1=BatchNormalization()(conv5_1)
    pool5=MaxPool1D(pool_size=(2), strides=(2), padding='same')(conv5_1)
    conv6_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool5)
    conv6_1=BatchNormalization()(conv6_1)
    pool6=MaxPool1D(pool_size=(2), strides=(2), padding='same')(conv6_1)
    conv7_1=Conv1D(64, (3), activation='relu', input_shape=im_shape)(pool6)
    conv7_1=BatchNormalization()(conv7_1)
    pool7=MaxPool1D(pool_size=(2), strides=(2), padding='same')(conv7_1)
    flatten=Flatten()(pool7)
    dense_end1=Dense(64, activation='relu')(flatten)
    dense_end2=Dense(32, activation='relu')(dense_end1)
    main_output=Dense(1, activation='sigmoid', name='main_output')(dense_end2)
    model = Model(inputs=inputs_cnn, outputs=main_output)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['acc'])
    csv_logger = CSVLogger(csv_path, append=False, separator=',')
    callbacks = [EarlyStopping(monitor='val_acc', patience=20),
             ModelCheckpoint(filepath=modelpath, monitor='val_acc', save_best_only=True, save_weights_only=False, verbose=1), 
                 CSVLogger(csv_path, append=False, separator=',')]
    
    history=model.fit(X_train, y_train, epochs=40, shuffle = True, callbacks=callbacks, batch_size=32, validation_data=(X_test, y_test))
    model.load_weights(modelpath)
    return (model, history)


# In[ ]:




