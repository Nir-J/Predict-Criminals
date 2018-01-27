import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Parameters
training_epochs = 30
batch_size = 50
start_rate = 0.0001

# Network Parameters
n_hidden_1 = 300 # 1st layer number of neurons
n_hidden_2 = 200 # 2nd layer number of neurons
n_hidden_3 = 50 # 3rd layer number of neurons
n_hidden_4 = 10 # 4th layer number of neurons
n_features = 70 # Feature columns
n_output = 1 # Output clssification

########### Defining tensorflow computational graph ###########

# tf Graph input
# Features
X = tf.placeholder(tf.float32, [None, n_features])
# Labels
Y = tf.placeholder(tf.float32, [None, n_output])
# decay step for learning rate decay
decay_step = tf.placeholder(tf.int32)


# Create model
def deep_neural_network(x):

	
	layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.relu)
	
	layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu)

	layer_3 = tf.layers.dense(layer_2, n_hidden_3, activation=tf.nn.relu)
	
	layer_4 = tf.layers.dense(layer_3, n_hidden_4, activation=tf.nn.relu)

	out_layer = tf.layers.dense(layer_4, n_output, activation=tf.nn.sigmoid)
	return out_layer

# Construct model
logits = deep_neural_network(X)

# Define loss and optimizer
labels = Y
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    					logits=logits, labels=labels))
global_step = tf.Variable(0, trainable=False)

# Using a learning rate which has polynomial decay
starter_learning_rate = start_rate
end_learning_rate = 0.00005 # we will use a polynomial decay to reach this learning rate
decay_steps = decay_step
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5)
#learning_rate = start_rate
# Using adam optimizer to reduce loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

# Model for testing and validating
prediction = logits # Just returns prediction values

def duplicate(df):
	""" Duplicates criminal rows to improve guilty to not guilty ratio """
	
	# This returns a boolean array representing truth values for each row
	rows = df['Criminal'] == 1
	# Creating a dataframe with all the criminal records
	df_try = df[rows]
	df = df.append([df_try]*12)
	return df


def get_accuracy(df):
	"""Calculates accuracy and confusion matrix for input data"""
	
	# Getting back predictions and transforming them to binary values
	pred_values = prediction.eval(feed_dict={X: df.iloc[:, 0:-1]}) # Getting back the predictions
	f = lambda x: 0 if x < 0.5 else 1
	preds = np.fromiter((f(xi) for xi in pred_values), pred_values.dtype, count=len(pred_values))

	# Preparing Labels
	labels = df.iloc[:, -1].values

	# Compating predictions and labels to calculate accuracy
	correct_labels = tf.equal(preds, labels)
	label_accuracy = tf.reduce_mean(tf.cast(correct_labels, tf.float32))
	result = label_accuracy.eval()
	print("Accuracy: {:.2f}%".format(result*100))

	# Confusion matrix to help troubleshoot
	tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
	print("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))

	return result*100


def plot_graph(train, validate):
	""" Plots graph of training vs validate """
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	#This is just numer of epochs
	x = np.arange(len(train))
	# Major and minor for y axis for more detail
	y_major = np.arange(85, 95, 1)
	y_minor = np.arange(85, 95, 0.2)
	#For plotting two graphs
	y1 = np.array(train)
	y2 = np.array(validate)
	# Fixing x and y axis
	ax.set_xticks(x)
	ax.set_yticks(y_major)
	ax.set_yticks(y_minor, minor=True)
	# Setting grid lines
	ax.grid(which="both")
	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)
	# Labels
	plt.xlabel('Epochs', fontsize=12)
	plt.ylabel('Accuracy', fontsize=12)

	ax.plot(x, y1)
	ax.plot(x, y2)
	plt.show()

def prepare_result(df):
	""" Function to prepare CSV of the testing result """

	# Getting back predictions and transforming them to binary values
	pred_values = prediction.eval(feed_dict={X: df.iloc[:, 1:]})
	f = lambda x: 0 if x < 0.5 else 1
	preds = np.fromiter((f(xi) for xi in pred_values), pred_values.dtype, count=len(pred_values))

	result = pd.DataFrame()
	result['PERID'] = df['PERID']
	result['Criminal'] = preds
	result.set_index('PERID', inplace=True)
	result.to_csv('output.csv', float_format='%.f')



################# MAIN ######################

training = pd.read_csv("criminal_train.csv", index_col='PERID')
testing = pd.read_csv("criminal_test.csv")

# Normalising numerical columns
cols_to_normalise = list(training.columns.values)[:70]
training[cols_to_normalise] = training[cols_to_normalise].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
testing[cols_to_normalise] = testing[cols_to_normalise].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


def train_and_test_model(traindata, testdata):

	# Splitting the training data into train and validate sets
	training, validate = np.split(traindata.sample(frac=1), [int(.9*len(traindata))])
	# Picking a same size random subset of training to compare validation results with training
	train = training.iloc[ 0:8382, :]
	
	# Lists to save progress for plotting graph
	train_acc = []
	validate_acc = []

	with tf.Session() as sess:
		sess.run(init)

        # Training cycle
		for epoch in range(training_epochs):
			# Shuffling dataset before training
			df = training.sample(frac=1)
			avg_cost = 0.
			total_data = df.index.shape[0] 
			num_batches = total_data // batch_size + 1
			i = 0
            # Loop over all batches
			while i < total_data:
			    batch_x = df.iloc[i:i+batch_size, 0:-1].values
			    batch_y = df.iloc[i:i+batch_size, -1].values # Last column is labels
			    # Reshaping labels from (?, ) to (?, 1)
			    batch_y = batch_y.reshape(batch_y.shape[0], 1)
			    i += batch_size
			    # Run optimization op and cost op (to get loss value)
			    _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
			                                                    Y: batch_y,
			                                                    decay_step: num_batches * training_epochs})
			    # Compute average loss
			    avg_cost += c / num_batches
			# Display logs per epoch step
			print("Epoch: {:04} | Cost={:.9f}".format(epoch+1, avg_cost))
			print("Validation", end=" ")
			validate_acc.append(get_accuracy(validate))
			print("Training", end=" ")
			train_acc.append(get_accuracy(train))
			print()
		print("Training complete")
		prepare_result(testdata)
	plot_graph(train_acc, validate_acc)


# Training the model after shuffling the data.

training = duplicate(training)
train_and_test_model(training, testing)
