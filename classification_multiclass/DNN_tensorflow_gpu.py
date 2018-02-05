import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf


start = time.time()

print('This program will distribute the tensorflow graphs on the available gpus')

print('Construction phase')

print(' - construct graph')
n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

with tf.device('/cpu:0'):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'), tf.device('/gpu:0'):
    hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'), tf.device('/gpu:1'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

print('Execution phase')

if not os.path.isdir("/tmp/data/mnist") or input('Redo training? [n]/Y\n') == 'Y':

    print(' - prepare tensorboard dir')
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = 'tf_logs'
    logdir = '{}/run-{}/'.format(root_logdir, now)

    print(' - prepare tensorboard summary writer')
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    print(' - load the data')
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/mnist')

    print(' - run the training and save tensorboard summary')
    n_epochs = 40
    batch_size = 50

    with tf.Session() as sess:
        init.run()
        n_batches = mnist.train.num_examples // batch_size
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

                if batch_index % 10 == 0:
                    summary_str = accuracy_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
            print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)

        save_path = saver.save(sess, './my_model_final_gpu.ckpt')

    print(' - close tensorboard file writer')
    file_writer.close()

print('Use the model')

with tf.Session() as sess:
    saver.restore(sess, './my_model_final_gpu.ckpt')

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/mnist')
    X_new_scaled = mnist.test.images[0:1]  # TODO: why does [0] not work?
    y_new_scaled = mnist.test.labels[0:1]  # TODO: why does [0] not work?

    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    print('Prediction:', y_pred, 'Truth:', y_new_scaled)

print('To start the tensorboard server: \'tensorboard --logdir tf_logs\'')

end = time.time()
print('Duration of program:', end - start)

