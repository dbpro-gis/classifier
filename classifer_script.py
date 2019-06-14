#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:56:28 2019

@author: noureldintawfek
"""


from contextlib import suppress
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
import warnings

from tensorflow.keras.applications import VGG16


from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from distutils.dir_util import copy_tree

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


from argparse import ArgumentParser

import glob

import shutil

import random

import datetime

import json



# parsing  arguments given to the script
parser = ArgumentParser()
parser.add_argument("--dataPath",
                    help="path of unclassified data", action='store' ,required = False)

args = parser.parse_args()

# creating data subfolder in the folder containing the script and copy all the data to it

orginalDataFolder = "originalData"

if(not os.path.isdir(orginalDataFolder)): 
    os.mkdir(orginalDataFolder)

dataDirectory = orginalDataFolder

"""
the folder must have this structure
./folder/
        class1/
            image.png/
            image.png/
            ...
        class2/
            image.png/
            image.png/
            ...
        class3/
            image.png/
            image.png/
            ...
"""      

# if a dataPath arg is given , all data in this path will be copied to data folder
if(args.dataPath!=None):
    shutil.rmtree(orginalDataFolder)
    os.mkdir(orginalDataFolder)
    fromDirectory = args.dataPath
    copy_tree(fromDirectory, dataDirectory)


# this variable will contain all class names
classes_names = os.listdir(dataDirectory)    
classes_num = len(classes_names)

"""
Creating another directory with this structure :
./flow/
        train/
            class1/
            class2/
            ...
        validate/
            class1/
            class2/
            ...
        test/
            class1/
            class2/
            ...
"""

if(os.path.isdir("flow")): 
    shutil.rmtree("flow")
    
os.mkdir("flow")
    
subfolder_names=["train","validate","test"]

for subfolder_name in subfolder_names: 
    os.mkdir("flow/"+subfolder_name)
    for class_name in classes_names: 
        os.mkdir("flow/"+subfolder_name+"/"+class_name)

test_percentage = 10
validate_percentage = 10


""" create 3 randomized lists: train_images, validate_images and test_images of  images name for each class ,  """

all_images={}
train_images={}
validate_images={}
test_images={}
print("creating randomized lists for testing , validating and testing  for each class ")
for class_name in classes_names: 
    all_images[class_name] = [f for f in glob.glob(orginalDataFolder+"/"+class_name+"/*")]
    class_sample_size = len(all_images[class_name])
    test_sample_size=int((test_percentage/100)*class_sample_size)
    validate_sample_size=int((validate_percentage/100)*class_sample_size)
    # get randomized list of images names for testing
    test_images[class_name]=[]
    for i in range(0,test_sample_size):
        randomized_image=all_images[class_name].pop(random.randrange(len(all_images[class_name])))
        test_images[class_name].append(randomized_image)
    # get randomized list of images names for validating
    validate_images[class_name]=[]
    for i in range(0,validate_sample_size):
        randomized_image=all_images[class_name].pop(random.randrange(len(all_images[class_name])))
        validate_images[class_name].append(randomized_image)
    train_images[class_name]= all_images[class_name]
    
print("sample size :\n")
for class_name in classes_names: 
    x= len(train_images[class_name])
    y= len(test_images[class_name])
    z= len(validate_images[class_name])
    num= x + y + z
    print("class_name="+class_name+" ,all_images_size="+str(num)+" ,train_images_size="+str(x)+" ,test_images_size="+str(y)+" ,validate_images_size="+str(z)+"\n")


all_images={ "train": train_images , "validate": validate_images, "test" : test_images }

print("load randomized files in flow directory")
for subfolder_name in subfolder_names:
    for class_name in classes_names: 
        for randomized_image in all_images[subfolder_name][class_name]:
            shutil.copy(randomized_image, "flow/"+subfolder_name+"/"+class_name)
            
            
# creating model

""" model Parameters """

img_width, img_height = 64, 64
channels_number = 3
epochs = 30
batch_size = 20
dateset_name = "Eurosat"
cnn_network = "VGG16"
batchsize = 100
unforzen_layers_size=4


""""""


train_dir = 'flow/train'
validation_dir = 'flow/validate'
test_dir = 'flow/test'

train_samples_size = sum( len(train_images[class_name]) for class_name in classes_names)
test_samples_size = sum( len(test_images[class_name]) for class_name in classes_names)
validate_samples_size = sum( len(validate_images[class_name]) for class_name in classes_names)

    
    
# Create the model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, channels_number))


# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-unforzen_layers_size]:
    layer.trainable = False
    

model = Sequential()
 
# Add the vgg16
model.add(vgg_conv)
 
# Add the classifier
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes_num, activation='softmax'))

train_datagen = ImageDataGenerator(rescale=1./255)
 
validation_datagen = ImageDataGenerator(rescale=1./255)


 
# Change the batchsize according to your system RAM

 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
 
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer= optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# saving model , its metadata and history

creation_date = datetime.datetime.now()

model_folder=dateset_name+"_"+cnn_network+"_"+str(creation_date)
os.mkdir(model_folder)
model.save(model_folder+"/"+'my_model.h5')

model_metadata = {
    "img_width" : img_width,
    "img_height" : img_height,
    "channels_number" : channels_number,
    "epochs" : epochs,
    "batch_size" : batch_size,
    "dateset_name" : dateset_name,
    "cnn_network" : cnn_network,
    "test_percentage" : test_percentage,
    "validate_percentage" : validate_percentage,
    "batchsize" : batchsize,
    "unforzen_layers_size":unforzen_layers_size
}



metadata_file = open(model_folder+"/"+'metadata.json', "w")
metadata_file.write(json.dumps(model_metadata))
metadata_file.close()





# ploting 
figs=[]
figs.append(plt.figure(0))
figs[0].clear()
x = np.arange(1, epochs+1)
plt.plot(x, history.history['loss'])
plt.plot(x, history.history['val_loss'])
plt.grid(linestyle='--')
plt.title('Training loss,Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='best')
plt.show()

plt.savefig(model_folder+"/"+'training_loss_validation_lose.png')


figs.append(plt.figure(1))
figs[1].clear()
x = np.arange(1, epochs+1)
plt.plot(x, history.history['accuracy'])
plt.plot(x, history.history['val_accuracy'])
plt.grid(linestyle='--')
plt.title('Training accuracy,Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='best')
plt.show()

plt.savefig(model_folder+"/"+'training_acc_validation_acc.png')


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
    
import pandas as pd
correct_imgs_json=pd.Series(correct_imgs).to_json(orient='values')

correct_imgs_file = open(model_folder+"/"+'correct_imgs', "w")
correct_imgs_file.write(json.dumps(correct_imgs_json))
correct_imgs_file.close()



false_imgs_json=pd.Series(false_imgs).to_json(orient='values')

false_imgs_file = open(model_folder+"/"+'false_imgs', "w")
false_imgs_file.write(json.dumps(false_imgs_json))
false_imgs_file.close()

test_accuracy= round((len(correct_imgs)/len(filesnames))*100,2)



from plot_confusion_matrix import plot_confusion_matrix


        
plot_confusion_matrix(test_generator.classes, predicted_classes, classes=classes_names,
                      title='Confusion matrix, without normalization, test_acc = '+str(test_accuracy)+"%")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12,forward=True)

plt.savefig(model_folder+"/"+'confusion_matrix.png')

plot_confusion_matrix(test_generator.classes, predicted_classes, classes=classes_names,
                      normalize=True,
                      title='Confusion matrix, without normalization, test_acc = '+str(test_accuracy)+"%")

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 12,forward=True)

plt.savefig(model_folder+"/"+'normalised_confusion_matrix.png')


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())




