# Land-cover image classification

## Conventional models

`conventional_classifier.py` compares different classifiers against each other
and can use the best-performing classifier to generate predictions on a second
dataset.

```
$ python3 conventional_classifier.py --predict --level l2 <dataset_dir> <output_dir>
```

## CNN-based classification

- classier script can be applied for classification of images. It uses keras neural-network library for python. the used CNN for classification is VGG16 , but the code can be manuelly adjusted to use any CNN Architecture.

this script do the following:

- receives an an argument path of the folder that contain the data , the folder must be in the following form:

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


- the data will be loaded to "orignalData" folder, in same folder that contain the script
- three randomized sets of images { Train , Validate , Test } of each class will be created and will be loaded in flow folder , the size of each set can be modified using  test_percentage, validate_percentage variables
the flow folder will have this structure :
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
- the model will be trained on trained set and will be saved in new folder with its metadata as json file. Two plots will be created, one compares validate_loss to train_loss , and the other compare validate_accuracy with train_accuracy
- json file containing all correct labled and mislabed images will be created 
- test data will be evaluated , confusion matrixes will be created and saved 

model variables can be modified in code except model Architecture : 


img_width, img_height = 64, 64
channels_number = 3
epochs = 30
batch_size = 20
dateset_name = "Eurosat"
cnn_network = "VGG16" 
batchsize = 100
unforzen_layers_size=4fier
