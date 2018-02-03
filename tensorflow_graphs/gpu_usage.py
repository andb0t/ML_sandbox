import tensorflow as tf


def variables_on_cpu(op):
    print('This op has type', op.type)
    if op.type == 'Variable':
        return '/cpu:0'
    else:
        return '/gpu:0'


print('Place operations')
use_dynamic_placement = False
if use_dynamic_placement:
    with tf.device(variables_on_cpu):
        a = tf.Variable(3.0)
        b = tf.Variable(4.0)    
        c = a * b
else:
    with tf.device('/gpu:0'):
        a = tf.Variable(3.0)
        b = tf.Variable(4.0)    
    c = a * b

with tf.device('/gpu:1'):
    print('Postpone evaluation of tensor d (in case of untimely high memory consumption)')
    with tf.control_dependencies([c]):
        d = a + b


print('Configure session')
config = tf.ConfigProto()
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # only use half the RAM on each GPU
sess = tf.Session(config=config)

print('Run session')
sess.run(tf.global_variables_initializer())
sess.run([c, d])

print(c, d)

