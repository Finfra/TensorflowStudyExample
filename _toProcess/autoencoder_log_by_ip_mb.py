
# coding: utf-8

# In[6]:


""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np


# In[7]:


# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import pandas as pd

from sklearn.model_selection import train_test_split
df=pd.read_csv("./../../_step2_OD/_data/EncodedData/union3_by_ip.csv",header=None)

train, test = train_test_split(df, test_size=0.2)

train_X=train[range(18)].values
train_X_cnt=train_X.shape[0]

test_X=test[range(18)].values
test_X_cnt=test_X.shape[0]




# In[8]:


# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 18 # MNIST data input (img shape: 28*28)


# In[9]:


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
keep_prob = 0.7

def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_1_drop=tf.nn.dropout(layer_1,keep_prob)
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1_drop, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_2_drop=tf.nn.dropout(layer_2,keep_prob)
    return layer_2_drop


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# In[11]:


class MiniBatch(object) :
    """for MiniBatch"""
    def __init__(self,arr):
        self.arr=arr
        self.wm=0
        self.train_X_cnt=len(arr)
    def miniBatch(self,cnt) : 
        wm_old=self.wm
        wm_new=self.wm+cnt
        self.wm=wm_new
        if self.train_X_cnt<wm_new :
            m1=self.arr[range(wm_old,self.train_X_cnt)]
            wm_new = wm_new-self.train_X_cnt
            m2=self.arr[range(wm_new)]
            self.wm=wm_new
            return np.vstack((m1,m2))
        else : 
            return self.arr[range(wm_old,wm_new)]

mb=MiniBatch(train_X)


# In[ ]:


# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    saver = tf.train.Saver()

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
#         batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x= mb.miniBatch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
        save_path = saver.save(sess, "./ckpt/autoencoder_by_id.ckpt")

# In[ ]:


saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./ckpt/autoencoder_by_id.ckpt")
    sess.run(init)
    batch_x=miniBatch(test_X,i%(test_X_cnt-batch_size),batch_size)
    i=i+1
    g = sess.run(decoder_op, feed_dict={X: batch_x})
    print( g)

    

