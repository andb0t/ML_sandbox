import tensorflow as tf


print('Place operation')
with tf.device('/gpu:0'):
    a = tf.Variable(3.0)
    b = tf.Variable(4.0)

c = a * b

print('Configure session')
config = tf.ConfigProto()
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # only use half the RAM on each GPU
sess = tf.Session(config=config)

print('Run session')
sess.run(tf.global_variables_initializer())
sess.run(c)

