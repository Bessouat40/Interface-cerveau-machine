import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import random

# Deep Learning
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense,  SeparableConv2D, Conv2D , MaxPool2D , Flatten, Input, Concatenate , Dropout , BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
from keras.preprocessing.image import ImageDataGenerator

def import_data(nom_electrode, racine, labels, img_size=(250,250)) :
    """Fonction qui permet d'importer les données
    img_size : de la forme ex : (300,300)"""
    data = []
    for j in range(len(labels)) :
        #on trouve les chemins vers les données scalo et spectro
        path_img_scalo = racine + "/" + labels[j] + "/" + nom_electrode + "/"
        #on récupère la liste des chemins pour les images scalo et spectro
        list_scalo = glob.glob(path_img_scalo + '*png' ) 
        #on enregistre chaque image à sa place 
        for l in range(len(list_scalo)) :
            #scalo
            img_scalo = cv2.imread(list_scalo[l])
            img = cv2.resize(img_scalo, img_size)
            #spectro
            data.append([img, labels[j]])
    return data

def separation_data(data):
    """Fonction qui permet de séparer la bdd en bdd d'entrainement, test, validation"""
    X = []
    y = []

    for i in range(len(data)):
        X.append(data[i][0])
        y.append(int(data[i][1]))

    X_shuf = []
    y_shuf = []
    index_shuf = list(range(len(X)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        X_shuf.append(X[i])
        y_shuf.append(y[i])

    n = int(len(X) * 0.8)
    n_ = int(len(X)-5)
    X_train = X_shuf[:n]
    X_test = X_shuf[n:n_]
    X_val = X_shuf[n_:]

    y_train = y_shuf[:n]
    y_test = y_shuf[n:n_]
    y_val = y_shuf[n_:]

    def normalize_image(img_set):
        img_set = np.array(img_set)/255
        return img_set
    X_train = normalize_image(X_train)
    X_val = normalize_image(X_val)
    X_test = normalize_image(X_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val

def creation_model(input_shape) :
    """Fonction qui permet de créer notre modèle"""
    model_1 = Sequential()

    model_1.add(Conv2D(8, (3,3), input_shape = input_shape[1:]))
    model_1.add(Activation("relu"))
    model_1.add(MaxPool2D(pool_size=(2, 2)))
    model_1.add(Dropout(0.2))

    model_1.add(Conv2D(16, (3,3), input_shape = input_shape[1:]))
    model_1.add(Activation("relu"))
    model_1.add(MaxPool2D(pool_size=(2, 2)))
    model_1.add(Dropout(0.2))

    model_1.add(Conv2D(32, (3,3)))
    model_1.add(Activation("relu"))
    model_1.add(MaxPool2D(pool_size=(2, 2)))
    model_1.add(Dropout(0.2))

    model_1.add(Flatten())
    model_1.add(Dense(32))
    model_1.add(Activation("relu"))
    model_1.add(Dropout(0.5))
    model_1.add(Dense(1))
    model_1.add(Activation("softmax"))

    return model_1

def fit_model(model, epochs, batch_size, steps, X_train, X_val, y_train, y_val):
    """fonction qui renvoie l'entrainement de notre modèle"""
    import gc
    gc.collect()

    datagen = ImageDataGenerator( shear_range=0.2, zoom_range=0.2, horizontal_flip=True) #on ne rescale pas car on l'a déja fait avant (normalisation des données : /255)

    datagen.fit(X_train)
    # fits the model on batches with real-time data augmentation:
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch= steps, 
            epochs=epochs,
            validation_data = (X_val, y_val))
    return history

def creation_y(y, labels):
    y_train = np.zeros((len(y), len(labels)))
    for i in range(len(y)):
        y_train[i][y[i]-1] = 1
    return y_train