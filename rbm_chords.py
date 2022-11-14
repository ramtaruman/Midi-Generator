
# ---------------------------------------------IMPORTS----------------------------------------
import os
import midi_manipulation
import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
# tqdm supports async, most of what is written these days.
from tqdm import tqdm
# -----------------------------------------------END OF IMPORTS-----------------------------------

# ------------------------------OPTIMIZATION BLOCK-----------------(all micro optimizations go in here)
# disables eager execution, this function is not necessary if you are using v2.
tf.compat.v1.disable_eager_execution()
# eager execution is enabled by default.

# Tensorflow pollutes standard error with gpu's memory allocation logs. To disable such error logging the bottom line is used.
# Use '3' as a parameter if you wanna disable everything, including info, warning and error. 2 Just disables infor and warning.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ---------------------------------END OF BLOCK----------------------------------------------------------


# -------------------------FILE HANDLING BLOCK---------------------------------------------------
def get_songs(path):
    # the glob module finds all pathnames matching a specified patter
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs


songs = get_songs('in')


print("{} songs processed".format(len(songs)))
# ----------------------------------END OF BLOCK-------------------------------------------------------

# -------------------------------------------------NODE WEIGHTS BLOCK--------------------------------------
# parameters for the model

lowest_note = midi_manipulation.lowerBound  # lowest note in the arrangement
highest_note = midi_manipulation.upperBound  # highest note in the arrangement
note_range = highest_note - lowest_note  # pitch range of the arrangement

num_timesteps = 15  # timesteps created one at a time, change at your own risk
n_visible = 2 * note_range * num_timesteps  # first rbm layer : visible
n_hidden = 50  # second rbm layer : hidden

num_epochs = 200  # cycles that are run, higher would probably tend to have more accurate results but more time

batch_size = 100  # number of samples, adjust according to the dataset
lr = tf.constant(0.005, tf.float32)  # recommended rate according to RBM paper


x = tf.compat.v1.placeholder(tf.float32, [None, n_visible], name="x")
# edge weights

W = tf.Variable(tf.random.normal([n_visible, n_hidden], 0.01), name="W")
# hidden layer
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))
# visible layer
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))
# I used a matrix, you can use whatever else
# ---------------------------BLOCK END-------------------------------------------------------------------------


# ---------------------------------------Help method---------------------------------------------------------------------

# 2do:Include sampling details in documentation
def sample(probs):
    # returns random vector from the given range
    return tf.floor(probs + tf.random.uniform(tf.shape(input=probs), 0, 1))
# -----------------------------------------end block---------------------------------------------------------------------


# -----------------------------Gibbs sampler------------------------------------------------------------------
def gibbs_sample(k):  # a simple function for gibbs chain

    def gibbs_step(count, k, xk):  # samples from distribution, definitions are W and bh and bv

        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
        xk = sample(
            tf.sigmoid(tf.matmul(hk, tf.transpose(a=W)) + bv))  # hidden to simple values
        return count + 1, k, xk

    ct = tf.constant(0)
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                   gibbs_step, [ct, tf.constant(k), x])

    x_sample = tf.stop_gradient(x_sample)
    return x_sample
    # above code is simply for stopping gradient leakage into the previous method of running gibbs chain

# -------------------------------------------------END OF BLOCK---------------------------------------------------------


x_sample = gibbs_sample(1)

h = sample(tf.sigmoid(tf.matmul(x, W) + bh))

h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

# method to update the values of W,bh,bv
size_bt = tf.cast(tf.shape(input=x)[0], tf.float32)
W_adder = tf.multiply(lr / size_bt,
                      tf.subtract(tf.matmul(tf.transpose(a=x), h), tf.matmul(tf.transpose(a=x_sample), h_sample)))
bv_adder = tf.multiply(
    lr / size_bt, tf.reduce_sum(input_tensor=tf.subtract(x, x_sample), axis=0, keepdims=True))
bh_adder = tf.multiply(
    lr / size_bt, tf.reduce_sum(input_tensor=tf.subtract(h, h_sample), axis=0, keepdims=True))
# tensor flow update routine
updt = [W.assign_add(W_adder), bv.assign_add(
    bv_adder), bh.assign_add(bh_adder)]

# few graph routines
with tf.compat.v1.Session() as sess:
    # init vars and train the model
    # ------------------------------------------------------------START OF MODEL-----------------------------------------------------------
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    # num_epochs is the number of cycles the data is being run through
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            # songs are restructured to fit the training methods
            song = np.array(song)
            song = song[:int(
                np.floor(song.shape[0] // num_timesteps) * num_timesteps)]
            song = np.reshape(
                song, [song.shape[0] // num_timesteps, song.shape[1] * num_timesteps])

            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x: tr_x})
# ----------------------------------------------------------END OF MODEL -------------------------------------------------------------
    sample = gibbs_sample(1).eval(session=sess, feed_dict={
        x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue

        S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        midi_manipulation.noteStateMatrixToMidi(
            S, "out/generated_chord_{}".format(i))
