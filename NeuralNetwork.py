import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
import scipy.stats

affect_list = ["anger", "fear", "joy", "sadness"]
#affect_list = ["sadness"]

import os
cwd = os.getcwd()
print cwd
with open(os.path.join(cwd, 'dataset', 'task1', 'train_ainl', 'dataset_json.txt')) as data_file:
    train_data = json.load(data_file)
with open(os.path.join(cwd, 'dataset', 'task1', 'development_ainl', 'dataset_json.txt')) as data_file:
    development_data = json.load(data_file)

test_data = development_data





def calculate_pearson(pred):
    pred_str = str(pred)
    pred_real = str(str(pred_str.replace("\n", "").replace("        ", "  ").replace("       ", "  ").replace("      ", "  ").replace("     ", "  ").replace("    ", "  ").replace("   ", "  ").replace("[[", "").replace("]]", "")))
    pred_real_arr = pred_real.split("  ")
    pred_real_arr_float = map(lambda(x): float(x.replace(" ", "")), pred_real_arr)
    a, b = scipy.stats.pearsonr(pred_real_arr_float, y_test)
    return a



for affect in affect_list:
    print affect
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    exc_cnt_tr = 0
    exc_cnt_te = 0

    for idd in train_data[affect]:
        #print idd
        try:
           X_train.append(train_data[affect][idd]["ainl"])
           y_train.append(train_data[affect][idd]["magnitude"])
        except:
          exc_cnt_tr += 1
    print exc_cnt_tr

    for idd in test_data[affect]:
        #print idd
        try:
           X_test.append(test_data[affect][idd]["ainl"])
           y_test.append(test_data[affect][idd]["magnitude"])
        except:
          exc_cnt_te += 1
    print exc_cnt_te







    # Import TensorFlow
    import tensorflow as tf

    # Define a and b as placeholders
    a = tf.placeholder(dtype=tf.int8)
    b = tf.placeholder(dtype=tf.int8)

    # Define the addition
    c = tf.add(a, b)

    # Initialize the graph
    graph = tf.Session()

    # Run the graph
    graph.run(c, feed_dict={a: 5, b: 4})



    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()



    # Model architecture parameters
    n_features = 1 #500


    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])




    n_neurons_1 = 1616
    n_neurons_2 = 808
    n_neurons_3 = 404
    n_neurons_4 = 202
    n_neurons_5 = 101
    n_neurons_6 = 50
    n_neurons_7 = 25
    n_neurons_8 = 12
    n_neurons_9 = 6
    n_target = 1
    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_features, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    # Layer: Variables for hidden weights and biases
    W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
    bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))
    # Layer: Variables for hidden weights and biases
    W_hidden_6 = tf.Variable(weight_initializer([n_neurons_5, n_neurons_6]))
    bias_hidden_6 = tf.Variable(bias_initializer([n_neurons_6]))
    # Layer: Variables for hidden weights and biases
    W_hidden_7 = tf.Variable(weight_initializer([n_neurons_6, n_neurons_7]))
    bias_hidden_7 = tf.Variable(bias_initializer([n_neurons_7]))
    # Layer: Variables for hidden weights and biases
    W_hidden_8 = tf.Variable(weight_initializer([n_neurons_7, n_neurons_8]))
    bias_hidden_8 = tf.Variable(bias_initializer([n_neurons_8]))
    # Layer: Variables for hidden weights and biases
    W_hidden_9 = tf.Variable(weight_initializer([n_neurons_8, n_neurons_9]))
    bias_hidden_9 = tf.Variable(bias_initializer([n_neurons_9]))



    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_9, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))


    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))
    hidden_6 = tf.nn.relu(tf.add(tf.matmul(hidden_5, W_hidden_6), bias_hidden_6))
    hidden_7 = tf.nn.relu(tf.add(tf.matmul(hidden_6, W_hidden_7), bias_hidden_7))
    hidden_8 = tf.nn.relu(tf.add(tf.matmul(hidden_7, W_hidden_8), bias_hidden_8))
    hidden_9 = tf.nn.relu(tf.add(tf.matmul(hidden_8, W_hidden_9), bias_hidden_9))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_9, W_out), bias_out))


    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)






    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())


    # Number of epochs and batch size
    epochs = 50
    batch_size = 250

    for e in range(epochs):
        
        # Shuffle training data
        shuffle_indices = np.random.permutation(len(y_train))
        #X_train = X_train[shuffle_indices]
        #y_train = y_train[shuffle_indices]
        X_tmp = []
        Y_tmp = []
        for i in shuffle_indices:
            X_tmp.append(X_train[i])
            Y_tmp.append(y_train[i])
        X_train = X_tmp
        y_train = Y_tmp

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
            
        #pred = net.run(out, feed_dict={X: X_test})
        #print "iteration=", e, " ; pearson:", calculate_pearson(pred)


    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    #print(mse_final)
    pred = net.run(out, feed_dict={X: X_test})

    print " TOTAL for ", affect, "  pearson: ", calculate_pearson(pred)
