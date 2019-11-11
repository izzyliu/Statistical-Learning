#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:52:45 2018

Q1&2

@author: VvanHuang
"""
import tensorflow as tf
import numpy as np
from DNNclassifier import DNNClassifier
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

he_init = tf.variance_scaling_initializer()

#???
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None,
        activation=tf.nn.elu, initializer=he_init):
    with tf.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, activation=activation,
                                     kernel_initializer=initializer,
                                     name="hidden%d" % (layer + 1))
        return inputs
    
n_inputs = 28 * 28 
n_outputs = 5

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

dnn_outputs = dnn(X)

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
Y_proba = tf.nn.softmax(logits, name="Y_proba")

#########################################################
print("1.2.2 training")
learning_rate = 0.005

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# load original data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

#train on digits 0 to 4
X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]

n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

#
#with tf.Session() as sess:
#    init.run()
#    plt.figure("8.2")
#    plt.title("losses vs number of epochs")
#    plt.xlabel("number of epochs")
#    plt.ylabel("losses")
#    val_losses = []
#    train_losses = []
#    for epoch in range(n_epochs):
#        rnd_idx = np.random.permutation(len(X_train1))
#        for rnd_indices in np.array_split(rnd_idx, len(X_train1) // batch_size):
#            X_batch, y_batch = X_train1[rnd_indices], y_train1[rnd_indices]
#            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid1, y: y_valid1})
#        if loss_val < best_loss:
#            save_path = saver.save(sess, "./my_mnist_model_0_to_4.ckpt")
#            best_loss = loss_val
#            checks_without_progress = 0
#        else:
#            checks_without_progress += 1
##            print('add 1!')
#            if checks_without_progress > max_checks_without_progress:
#                print("Early stopping!")
#                break
#        train_losses.append(sess.run(loss, feed_dict={X: X_batch, y: y_batch}))
#        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
#            epoch, loss_val, best_loss, acc_val * 100))
#        val_losses.append(loss_val)
#    print(train_losses)
#    
#    plt.plot(range(len(val_losses)), val_losses, label="validation loss")
#    plt.plot(range(len(train_losses)), train_losses, label="training loss")
#    plt.grid()
#    plt.legend()
#    plt.show()
#        
#
#with tf.Session() as sess:
#    saver.restore(sess, "./my_mnist_model_0_to_4.ckpt")
#    acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
#    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    
##############################################
print("1.2.3 hyperparameters")


def leaky_relu(alpha=0.1):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu

print('Set1)')
dnn_clf1 = DNNClassifier(n_neurons=50, learning_rate=0.005, batch_size = 20, activation=tf.nn.relu, random_state=42)

dnn_clf1.fit(X_train1, y_train1, n_epochs=1000, max_checks_without_progress = 20, X_valid=X_valid1, y_valid=y_valid1)

y_pred1 = dnn_clf1.predict(X_test1)
accuracy_score1 = accuracy_score(y_test1, y_pred1)
print("The accuracy score is %s"%accuracy_score1)

##
#print('Set2)')
#dnn_clf2 = DNNClassifier(n_neurons=140, learning_rate=0.1, batch_size = 500, activation=leaky_relu(alpha=0.1), random_state=42)
#
#dnn_clf2.fit(X_train1, y_train1, n_epochs=1000, max_checks_without_progress = 30, X_valid=X_valid1, y_valid=y_valid1)
#
#y_pred2 = dnn_clf2.predict(X_test1)
#accuracy_score2 = accuracy_score(y_test1, y_pred2)
#print("The accuracy score is %s"%accuracy_score2)

    
#################################################################################
#print("1.2.4 Batch normalization")
#
#dnn_clf = DNNClassifier(learning_rate=0.005, batch_size = 20, activation=tf.nn.elu, random_state=42)
#momentum_list = [0.85, 0.9, 0.95, 0.99]
#acc_score_list = []
#for momentum in momentum_list:
#    print("Momentum=%s"%momentum)
#    dnn_clf_bn = DNNClassifier(learning_rate=0.005, batch_size = 20, activation=tf.nn.elu, random_state=42, 
#                           batch_norm_momentum=momentum)
#    dnn_clf_bn.fit(X_train1, y_train1, n_epochs=1000, max_checks_without_progress = 20, X_valid=X_valid1, y_valid=y_valid1)
#    y_pred = dnn_clf_bn.predict(X_test1)
#    acc_score_list.append(accuracy_score(y_test1, y_pred))
#print("The best accuracy score is: %s"%max(acc_score_list))

###################################################################################
print("1.2.5 add dropout")
#dnn_clf = DNNClassifier(learning_rate=0.005, batch_size = 20, activation=tf.nn.elu, random_state=42)
dropout_rates = [0.1, 0.3]
for rate in dropout_rates:
    print("dropout rate=%s"%rate)
    dnn_clf_dropout = DNNClassifier(learning_rate=0.005, batch_size = 20, activation=tf.nn.elu, random_state=42,
                                dropout_rate=rate)
    dnn_clf_dropout.fit(X_train1, y_train1, n_epochs=1000,max_checks_without_progress = 20, X_valid=X_valid1, y_valid=y_valid1)
    y_pred = dnn_clf_dropout.predict(X_test1)
    acc_score = accuracy_score(y_test1, y_pred)
    print("The accuracy score is %s"%acc_score)
    
    
    
    
    
    
    
    
    
    
    