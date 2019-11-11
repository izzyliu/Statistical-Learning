#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:13:44 2018

@author: VvanHuang
"""

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import matplotlib
import matplotlib.pyplot as plt
#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
#X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
#y_train = y_train.astype(np.int32)
#y_test = y_test.astype(np.int32)
#X_valid, X_train = X_train[:5000], X_train[5000:]
#y_valid, y_train = y_train[:5000], y_train[5000:]

# He et al. initialization from https://arxiv.org/abs/1502.01852
he_init = tf.contrib.layers.variance_scaling_initializer()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:20:05 2018

@author: VvanHuang
"""

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, n_neurons=100, optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=None, batch_size=None, activation=None, initializer=he_init,
                 batch_norm_momentum=None, dropout_rate=None, random_state=None):
        """Initialize the DNNClassifier by simply storing all the hyperparameters."""
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        """Build the hidden layers, with support for batch normalization and dropout."""
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(inputs, self.n_neurons,
                                     kernel_initializer=self.initializer,
                                     name="hidden%d" % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum,
                                                       training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        """Build the same model as earlier"""
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name='training')
        else:
            self._training = None

        dnn_outputs = self._dnn(X)

        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=1000, max_checks_without_progress = None, X_valid=None, y_valid=None):
        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""
        self.close_session()
        plt.figure(1)
        plt.title("losses vs number of epoches")
        plt.xlabel("number of epoches")
        plt.ylabel("losses")
        # infer n_inputs and n_outputs from the training set.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)
        
        # Translate the labels vector to a vector of sorted class indices, containing
        # integers from 0 to n_outputs - 1.
        # For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the sorted class
        # labels (self.classes_) will be equal to [5, 6, 7, 8, 9], and the labels vector
        # will be translated to [3, 3, 4, 0, 2, 1, 1, 1]
        self.class_to_index_ = {label: index
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # needed in case of early stopping
#        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None
        val_losses = []
        train_losses = []
        # Now train the model!
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy],
                                                 feed_dict={self._X: X_valid,
                                                            self._y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_val, best_loss, acc_val * 100))
                    val_losses.append(loss_val)
                    
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
                    train_losses.append(sess.run(self._loss, feed_dict={self._X: X_batch, self._y: y_batch}))
                else:
                    loss_train, acc_train = sess.run([self._loss, self._accuracy],
                                                     feed_dict={self._X: X_batch,
                                                                self._y: y_batch})
                    train_losses.append(loss_train)
         
                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_train, acc_train * 100))
            plt.plot(range(len(val_losses)), val_losses, label="validation loss")
            plt.plot(range(len(train_losses)), train_losses, label="training loss")
            plt.grid()
            plt.legend()
            plt.show()
            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)
# This class inherits from Sklearn's BaseEstimator and ClassifierMixin 
#class DNNClassifier(BaseEstimator, ClassifierMixin):
#	def __init__(self, n_hidden_layers=5, n_neurons=100, 
#              optimizer_class=tf.train.AdamOptimizer, learning_rate=0.01, 
#              batch_size=20, activation=tf.nn.elu, initializer=he_init, 
#              batch_norm_momentum=None, dropout_rate=None, 
#              max_checks_without_progress=20,show_progress=10,
#              tensorboard_logdir=None, random_state=None):
#		
#		# Initialize the class with sensible default hyperparameters
#		self.n_hidden_layers = n_hidden_layers
#		self.n_neurons = n_neurons
#		self.optimizer_class = optimizer_class
#		self.learning_rate = learning_rate
#		self.batch_size = batch_size
#		self.activation = activation
#		self.initializer = initializer
#		self.batch_norm_momentum = batch_norm_momentum
#		self.dropout_rate = dropout_rate
#		self.max_checks_without_progress = max_checks_without_progress
#		self.show_progress = show_progress
#		self.random_state = random_state
#		self.tensorboard_logdir = tensorboard_logdir
#		self._session = None #Instance variables preceded by _ are private members
#		
#	def _dnn(self, inputs):
#		'''This method builds the hidden layers and
#		 Provides for implementation of batch normalization and dropout'''
#
#		for layer in range(self.n_hidden_layers):
#
#			# Apply dropout if specified
#			if self.dropout_rate:
#				inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, 							training=self._training)
#			# Create the hidden layer
#			inputs = tf.layers.dense(inputs, self.n_neurons, 
#									 activation=self.activation, 
#									 kernel_initializer=self.initializer, 
#									 name = "hidden{}".format(layer+1))
#
#			# Apply batch normalization if specified
#			if self.batch_norm_momentum:
#				inputs = tf.layers.batch_normalization(inputs,momentum=										self.batch_norm_momentum,
#													training=self._training)
#				
#			# Apply activation function
#			inputs = self.activation(inputs, name="hidden{}_out".format(layer+1))
#		return inputs
#		
#	def _construct_graph(self, n_inputs, n_outputs):
#		'''This method builds the complete Tensorflow computation graph
#			n_inputs: number of features 
#			n_outputs: number of classes
#		'''
#
#		if self.random_state:
#			tf.set_random_seed(self.random_state)
#			np.random.seed(self.random_state)
#		 
#		# Placeholders for training data, labels are class exclusive integers
#		X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
#		y = tf.placeholder(tf.int32, shape=[None], name="y")
#		
#		# Create a training placeholder 
#		if self.batch_norm_momentum or self.dropout_rate:
#			self._training = tf.placeholder_with_default(False, shape=[], name="training")
#		else:
#			self._training = None
#		
#		# Output after hidden layers 
#		pre_output = self._dnn(X)
#		
#		# Outputs from output layer
#		logits = tf.layers.dense(pre_output, n_outputs, kernel_initializer=he_init, name="logits")
#		probabilities = tf.nn.softmax(logits, name="probabilities")
#		
#		''' Cost function is cross entropy and loss is average cross entropy. Sparse softmax must be used because shape of logits is [None, n_classes] and shape of labels is [None]'''
#		xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, 													 logits=logits)
#		loss = tf.reduce_mean(xentropy, name="loss")
#		
#		'''Optimizer and training operation. The control dependency is necessary for implementing batch normalization. The training operation must be dependent on the batch normalization.'''
#
#		optimizer = self.optimizer_class(learning_rate=self.learning_rate)
#		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#		with tf.control_dependencies(update_ops):
#			training_op = optimizer.minimize(loss)
#		
#		# Metrics for evaluation
#		correct = tf.nn.in_top_k(logits, y, 1)    
#		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name="accuracy")
#
#		# Initializer and saver 
#		init = tf.global_variables_initializer()
#		saver = tf.train.Saver()
#	
#		if self.tensorboard_logdir:
#			now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
#			tb_logdir = self.tensorboard_logdir + "/run-{}".format(now)
#			cost_summary = tf.summary.scalar("validation_loss", loss)
#			acc_summary = tf.summary.scalar("validation_accuracy", accuracy)
#			merged_summary = tf.summary.merge_all()
#			file_writer = tf.summary.FileWriter(tb_logdir, 													tf.get_default_graph())
#			
#			self._merged_summary = merged_summary
#			self._file_writer = file_writer
#		
#		self._X, self._y = X, y
#		self._logits = logits
#		self._probabilities = probabilities
#		self._loss = loss
#		self._training_op = training_op
#		self._accuracy = accuracy
#		self._init, self._saver = init, saver
#		
#		
#	def close_session(self):
#		if self._session:
#			self._session.close()
#			
#	def _get_model_parameters(self):
#		# Retrieves the value of all the variables in the network 
#		with self._graph.as_default():
#			gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#		return {gvar.op.name: value for gvar, value in 
#				zip(gvars, self._session.run(gvars))}
#	
#	def _restore_model_parameters(self, model_params):
#		# Restores the value of all variables using tf assign operations
#		# First retrieve the list of all the graph variables
#		gvar_names = list(model_params.keys())
#		
#		# Then retrieve all the assignment operations in the graph
#		assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + 			 "/Assign") for gvar_name in gvar_names}
#
#		# Fetch the initialization values of the assignment operations
#		'''graph.get_operation_by_name(operation).inputs returns the input to the given operation; because these are all assignment operations, the second argument to inputs is the value assigned to the variable'''
#		init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op 			  in assign_ops.items()}
#		# Create a dictionary mapping initial values to values after training
#		feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
#		# Assign the trained value to all the variables in the graph
#		self._session.run(assign_ops, feed_dict=feed_dict)
#		
#	def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
#		# Method to train the model. Implements early stopping if validation data is provided 
#		
#		self.close_session()
#		n_inputs = X.shape[1] # Number of features
#		
#		# If labels are provided in one_hot form, convert to integer class labels
#		y = np.array(y)
#		y_valid = np.array(y_valid)
#		
#		if len(y.shape) == 2:
#			y = np.argmax(y, axis=1)
#	 
#		if len(y_valid.shape) == 2:
#			y_valid = np.argmax(y_valid, axis=1)
#
#		self.classes_ = np.unique(y)
#		n_outputs = len(self.classes_) # Number of classes
#	
#		# Tensorflow expects labels from 0 to n_classes - 1. 
#		self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
#		labels = [self.class_to_index_[label] for label in y]
#		y = np.array(labels, dtype=np.int32)
#		
#		self._graph = tf.Graph()
#			
#		# Build the computation graph with self as default graph
#		with self._graph.as_default():
#			self._construct_graph(n_inputs, n_outputs)
#			
#		# Early stopping parameters
#		checks_without_progress = 0 
#		best_loss = np.float("inf")
#		best_parameters = None
#		
#		self._session = tf.Session(graph=self._graph)
#
#		with self._session.as_default() as sess:
#			# Initialize all variables
#			self._init.run()
#			num_instances = X.shape[0] # Total number of training instances
#			for epoch in range(n_epochs):
#				rnd_idx = np.random.permutation(num_instances)
#				for rnd_indices in np.array_split(rnd_idx, num_instances // self.batch_size):
#					X_batch, y_batch = X[rnd_indices], y[rnd_indices]
#					feed_dict = {self._X: X_batch, self._y: y_batch}
#					if self._training is not None:
#						feed_dict[self._training] = True
#					train_acc, _ = sess.run([self._accuracy,self._training_op], 					feed_dict)
#
#				# Early stopping implementation
#				if X_valid is not None and y_valid is not None:
#					feed_dict_valid = {self._X: X_valid, self._y: y_valid}
#
#					# Write summary for tensorboard
#					if self.tensorboard_logdir:
#						val_acc, val_loss, summary = sess.run([self._accuracy, self._loss, self._merged_summary], feed_dict=feed_dict_valid)
#
#						self._file_writer.add_summary(summary, epoch)
#
#					else:
#						val_acc, val_loss = sess.run([self._accuracy, self._loss], feed_dict=feed_dict_valid)
#					
#					# Show training progress every show_progress epochs
#					if self.show_progress:
#						if epoch % self.show_progress == 0:
#							print("Epoch: {} Current training accuracy: {:.4f} Validation Accuracy: {:.4f} Validation Loss {:.6f}".format(
#								epoch+1, train_acc, val_acc, val_loss))
#
#					# Check to see if model is improving 
#					if val_loss < best_loss:
#						best_loss = val_loss
#						checks_without_progress = 0
#						best_parameters = self._get_model_parameters()
#					else:
#						checks_without_progress += 1
#
#					if checks_without_progress > self.max_checks_without_progress:
#						print("Stopping Early! Loss has not improved in {} epochs".format(
#										   self.max_checks_without_progress))
#						break
#			   
#				# No validation set provided
#				else:
#					if self.show_progress:
#						if epoch % self.show_progress == 0:
#							print("Epoch: {} Current training accuracy: {:.4f}".format(
#								epoch+1, train_acc))
#						
#			# In the case of early stopping, restore the best weight values
#			if best_parameters:
#				self._restore_model_parameters(best_parameters)
#				return self
#			
#	def predict_probabilities(self, X):
#		# Predict the probabilities of each class 
#		if not self._session:
#			raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
#		with self._session.as_default() as sess:
#			return self._probabilities.eval(feed_dict={self._X: X})
#
#	def predict(self, X):
#		# Predict the classes themselves and return with shape=(None,)
#		class_indices = np.argmax(self.predict_probabilities(X), axis=1)
#		predictions = np.array([[self.classes_[class_index]] for class_index in class_indices], dtype=np.int32)
#		return np.reshape(predictions, (-1,))
#		
#	def save(self, path):
#		# Save the model to provided path
#		self._saver.save(self._session, path)