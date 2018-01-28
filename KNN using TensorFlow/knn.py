import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tflearn.data_utils as du
import numpy as np

k = 5

#Loading the MNIST Dataset
def load_mnist_data():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	train_X = mnist.train.images
	train_Y = mnist.train.labels
	test_X = mnist.test.images
	test_Y = mnist.test.labels
	return train_X, train_Y, test_X, test_Y


train_X, train_Y, test_X, test_Y = load_mnist_data()
dim_m = train_X.shape[0] 	# data points
dim_n = train_X.shape[1] 	# features
classes = train_Y.shape[1]	# target classes
dim_m_t = test_X.shape[0] 	# test data points

print(dim_m, "Training samples loaded")
print(dim_m_t, "Test samples loaded")

# Input Placeholders
X = tf.placeholder(shape = [None, dim_n], dtype = tf.float32)
Y = tf.placeholder(shape = [None, classes], dtype = tf.float32)
X_t = tf.placeholder(shape = [dim_n], dtype = tf.float32)

# Calculate distance^2 to all training instances
distance = tf.reduce_sum(tf.square(X - X_t), axis = 1)

# Get count of k nearest neighbours's classes
_, idxs = tf.nn.top_k(tf.negative(distance), k=k, sorted=False)
k_nereast_neighbours_count = tf.reduce_sum(tf.gather(Y, idxs), axis = 0)

pred = tf.argmax(k_nereast_neighbours)

accuracy = 0

# Tensorflow Session
init = tf.global_variables_initializer()
sess = tf.Session()

for i in range(dim_m_t):
	sess.run(init)
	predicted_label = sess.run(pred, feed_dict = {X:train_X, Y:train_Y, X_t:test_X[i, :]})
	actual_label = np.argmax(test_Y[i])
	if actual_label == predicted_label:
		accuracy += 1./dim_m_t

print("Accuracy:", accuracy*100)