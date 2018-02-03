import tensorflow as tf


print('Place operation')
with tf.device('/gpu:0'):
    a = tf.Variable(3.0)
    b = tf.Variable(4.0)

c = a * b

print('Log placement')
config = tf.ConfigProto()
config.log_device_placement = True
sess = tf.Session(config=config)

print('Run session')
sess.run(tf.global_variables_initializer())
sess.run(c)

