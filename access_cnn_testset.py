#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 17:08:45 2019

@author: noureldintawfek
"""


from contextlib import suppress
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
import warnings

from keras.applications import VGG16

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from distutils.dir_util import copy_tree

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from argparse import ArgumentParser

import glob

import shutil

import random

import datetime

import json

from sklearn.utils import class_weight


""" model Parameters """

img_width, img_height = 120, 120
channels_number = 3
epochs = 30
batch_size = 20
dateset_name = "Corina"
cnn_network = "VGG16"
unforzen_layers_size=2


""""""

test_dir = 'flow/test'
model_folder = test_dir
train_datagen = ImageDataGenerator(rescale=1./255)
model_dir= "my_model.h5"
predict_dir= "perdections"
from keras.models import load_model
model = load_model(model_dir)
# assess how well the modul is using test data set
test_generator = train_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

filesnames = test_generator.filenames

correct_classes = test_generator.classes

predicts = model.predict_generator(
        test_generator, 
        steps=test_generator.samples/test_generator.batch_size,
        verbose=1)

predicted_classes = np.argmax(predicts,axis=1)


classes_names = list(test_generator.class_indices.keys())
# load data about correct_imgs , false_img and confusion matrix in files
correct_imgs=[]
false_imgs=[]
for i in range(len(correct_classes)):
    correct_class=correct_classes[i]
    predicted_class=predicted_classes[i]
    filename=filesnames[i]
    perdict=predicts[i]
    # get index of the 3 largest elements
    most_perdicted=list(reversed(sorted(range(len(perdict)), key=lambda x: perdict[x])[-3:]))
    if(correct_class==predicted_class):
        correct_imgs.append({
                "file_name":filename,
                "correct_class":classes_names[correct_class],
                "correct_class_confidence":perdict[correct_class],
                "1st_pred_class":classes_names[most_perdicted[0]],
                "1st_pred_class_confidence":perdict[most_perdicted[0]],
                "2nd_pred_class":classes_names[most_perdicted[1]],
                "2nd_pred_class_confidence":perdict[most_perdicted[1]],
                "3rd_pred_class":classes_names[most_perdicted[2]],
                "3rd_pred_class_confidence":perdict[most_perdicted[2]],
                })
        
    else:
        false_imgs.append({
            "file_name":filename,
            "correct_class":classes_names[correct_class],
            "correct_class_confidence":perdict[correct_class],
            "1st_pred_class":classes_names[most_perdicted[0]],
            "1st_pred_class_confidence":perdict[most_perdicted[0]],
            "2nd_pred_class":classes_names[most_perdicted[1]],
            "2nd_pred_class_confidence":perdict[most_perdicted[1]],
            "3rd_pred_class":classes_names[most_perdicted[2]],
            "3rd_pred_class_confidence":perdict[most_perdicted[2]],
            })
    movePath=predict_dir+"/"+classes_names[most_perdicted[0]]
    if(not os.path.isdir(movePath)): 
        os.mkdir(movePath)
    shutil.copy(test_dir+"/"+filename, movePath)
    print(filename +" to " + movePath)
    
import pandas as pd
correct_imgs_json=pd.Series(correct_imgs).to_json(orient='values')

correct_imgs_file = open(predict_dir+"/"+'correct_imgs', "w")
correct_imgs_file.write(json.dumps(correct_imgs_json))
correct_imgs_file.close()



false_imgs_json=pd.Series(false_imgs).to_json(orient='values')

false_imgs_file = open(predict_dir+"/"+'false_imgs', "w")
false_imgs_file.write(json.dumps(false_imgs_json))
false_imgs_file.close()

test_accuracy= round((len(correct_imgs)/len(filesnames))*100,2)



from plot_confusion_matrix import plot_confusion_matrix


        
plot_confusion_matrix(test_generator.classes, predicted_classes, classes=classes_names,
                      title='Confusion matrix, without normalization, test_acc = '+str(test_accuracy)+"%")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12,forward=True)

plt.savefig(predict_dir+"/"+'confusion_matrix.png')

plot_confusion_matrix(test_generator.classes, predicted_classes, classes=classes_names,
                      normalize=True,
                      title='Confusion matrix, without normalization, test_acc = '+str(test_accuracy)+"%",)

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12,forward=True)

plt.savefig(predict_dir+"/"+'normalised_confusion_matrix.png')