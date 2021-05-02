# Image classification using Convolutional Neural Networks

import numpy as np
import os 
import scipy.ndimage
import scipy.misc as smc
import cv2
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from helper_functions import *
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def main():
    
    # Data set paths 
    results_path = "result"
    train_dir = "Train_set"
    test_dir  = "Test_set"
    
    # Find out input parameters
    n_train_img, n_test_img = input_data(train_dir, test_dir)
    img_size   = (244,244)
    img_pixels = img_size[0]*img_size[1]
    n_classes  = 2
    
    # Load training and testing data
    train_X, train_y, test_X, test_y = load_data(train_dir, test_dir, n_train_img, n_test_img,
                                                 n_classes, img_pixels, img_size, imgcolor = True)
    
    # Define architecture
    input_nodes = img_pixels
    print("Number of input nodes  : ", input_nodes)
    hidden_nodes = 300
    print("Number of hidden nodes : ", hidden_nodes)
    n_epoch = 20
    print("Number of epochs       : ", n_epoch)
    batch_size = 64
    print("Size of batch          : ", batch_size)
    drop_rate = 0.5
    print("Drop rate              : ", drop_rate)
    
    # Define placeholders
    tf.disable_eager_execution()
    x = tf.placeholder('float', [None, img_pixels])
    
    # Reshape input to a tensor 
    input_layer = tf.reshape(x, [-1, img_size[0], img_size[1], 1])

    # Conv layer 1
    conv1 = tf.layers.conv2d(inputs = input_layer, filters = 16, kernel_size = [3, 3], 
                             strides = 1, padding = "same", activation = tf.nn.relu)
    print("Size of 1st convolution layer : ", conv1.shape)
    
    # batchN = tf.nn.batch_normalization()

    # Pool layer 1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)
    print("Size of 1st pooling layer     : ", pool1.shape)

    # Conv layer 2
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 32, kernel_size = [3, 3],
                             strides = 1, padding = "same", activation = tf.nn.relu)
    print("Size of 2nd convolution layer : ", conv2.shape)

    # Pool layer 2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
    print("Size of 2nd pooling layer     : ", pool2.shape)

    # Conv Layer 3 
    conv3 = tf.layers.conv2d(inputs = pool2, filters = 32, kernel_size = [3, 3],
                             strides = 1, padding = "same", activation = tf.nn.relu)
    print("Size of 3rd convolution layer : ", conv3.shape)

    # Pool layer 3
    pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size=[2, 2], strides = 2)
    print("Size of 3rd pooling layer     : ", pool3.shape)

    # Conv layer 4
    conv4 = tf.layers.conv2d(inputs = pool3, filters = 64, kernel_size = [3, 3],
                             strides = 1, padding = "same", activation = tf.nn.relu)
    print("Size of 4rth convolution layer: ", conv4.shape)

    # Pool layer 4
    pool4 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [2, 2], strides = 2)
    print("Size of 4rth pooling layer    : ", pool4.shape)
    
    # Fully connected layer 1
    pool4_flat = tf.reshape(pool4, [-1, pool4.shape[1]*pool4.shape[2]*64])
    fc = tf.layers.dense(inputs = pool4_flat, units = hidden_nodes, activation = tf.nn.relu)
    print("Size of fully connected layer : ", fc.shape)
    
    # Dropout layer
    dropout = tf.layers.dropout(inputs = fc, rate = drop_rate)
    print("Size of dropout layer         : ", dropout.shape)
    
    # Logits layer : input to the softmax for prediction
    predict = tf.layers.dense(inputs = dropout, units = n_classes)
    print("Size of logits layer          : ", predict.shape)

    # Placeholder for labels of the classes
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    # Loss function : with Softmax 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = y_))
    
    # Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # Create session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    
    # Trainig model 
    for epoch in range(n_epoch):
        epoch_loss = 0
        for i in range(int(n_train_img/batch_size)):
            epoch_X, epoch_y = rand_batch(batch_size, train_X, train_y)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_X, y_: epoch_y})
            epoch_loss += c
        
        pred_correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_correct, 'float'))
        
        # Print important values 
        print(epoch,'from', n_epoch, 'loss value:',epoch_loss)
        print('Accuracy:',accuracy.eval({x:test_X, y_:test_y})*100, "%\n------------------------------------------")

        # Save predictions and labels to a csv file for analysis
        prediction = tf.argmax(predict,1)
        p =  prediction.eval(feed_dict={x: test_X}, session=sess)
        np.savetxt(results_path + "/results.csv", np.r_['0,2', np.argmax(test_y, axis=1), p], delimiter = ",")

    # Save the trained model 
    saver.save(sess, results_path + "/trained_model")
    sess.close()