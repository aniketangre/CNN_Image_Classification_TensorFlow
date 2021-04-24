#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os 
import scipy.ndimage
import scipy.misc as smc
import cv2
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
        
def rand_batch(num, data, labels):
    """ This functions returns num random samples and labels from a data set
        num    = required number of samples as an output
        data   = dat set from which random samples to be collected   
        labels = labels for the given data set                
    """
    # 1D array for indices
    idx = np.arange(0, len(data))
    
    # Shuffling indices
    np.random.shuffle(idx)
    idx_new = idx[:num]
    
    # Storing batch
    data_new = [data[i] for i in idx_new]
    labels_new = [labels[i] for i in idx_new]
    
    return np.asarray(data_new), np.asarray(labels_new)


def load_data(train_dir, test_dir, n_train_img, n_test_img, n_classes, img_pixels, img_size, imgcolor=True):
    """ This functon takes data set directories, load the data and process it for CNN model
        train_dir    = path of the training data set
        test_dir     = path of the testing data set
        n_train_img  = number of training examples 
        n_test_img   = number of test images
        n_classes    = number of classes for identification
        image_pixels = number of pixels in the image
    """
    # Array for storing training data
    train_X = np.zeros((n_train_img, img_pixels))
    train_y = np.zeros((n_train_img, n_classes))
    
    # Array for storing testing data
    test_X = np.zeros((n_test_img, img_pixels))
    test_y = np.zeros((n_test_img, n_classes))
    
    # Label encoding
    classes = ['bus', 'car']
    bus = [1,0]
    car = [0,1]
    
    train_idx = 0
    test_idx  = 0
    
    # Loop to search folders in train_dir 
    for folder in os.listdir(train_dir):
        # Loop over all the images in the folder 
        for file in os.listdir(train_dir + "/" + folder):
            # Read image
            img = plt.imread(train_dir + "/" + folder + "/" + file, True)
            
            # Convert images to gray from rgb
            if imgcolor == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize and flatten the image data 
            img = np.resize(img, img_size)
            img = img.flatten()
            
            # Store the 1D array with pixel data from each image
            train_X[train_idx] = img
            
            # Store the labels
            class_idx = classes.index(folder)
            train_y[train_idx] = [bus, car][class_idx]
            
            # Increment of index position
            train_idx = train_idx + 1
            
        print("Images from", folder, "are loaded")  
    
    # Loop to search folders in test_dir     
    for folder in os.listdir(test_dir):
        # Loop over all the images in the folder
        for file in os.listdir(test_dir + "/" + folder):
            # Read image
            img = plt.imread(test_dir + "/" + folder + "/" + file, True) 
            
            # Convert images to gray from rgb
            if imgcolor == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize and flatten the image data
            img = np.resize(img, img_size)
            img = img.flatten()
            
            # Store the 1D array with pixel data from each image
            test_X[test_idx] = img
            
            # Store the labels
            class_idx = classes.index(folder)
            test_y[test_idx] = [bus, car][class_idx]
            
            # Increment of index position
            test_idx = test_idx + 1
            
        print("Images from", folder, "are loaded")
         
    return train_X, train_y, test_X, test_y


def input_data(train_dir, test_dir):
    """ This functions reads the directory and returns number of images and sizes
        train_dir    = path of the training data set
        test_dir     = path of the testing data set
        img_size     = user selected image size for models
    """
    n_train_img = 0
    for file in os.listdir(train_dir):
        n_train_img += len(os.listdir(train_dir + "/" + file))
    
    n_test_img = 0
    for file in os.listdir(test_dir):
        n_test_img += len(os.listdir(test_dir + "/" + file))
        
    return n_train_img, n_test_img
