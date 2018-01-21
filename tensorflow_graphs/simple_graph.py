import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

explicitness = 2

if explicitness == 0:
    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    sess.close()
elif explicitness == 1:
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
elif explicitness == 2:
    init = tf.global_variables_initializer()  # prepare an init node
    with tf.Session() as sess:
        init.run()  # actually initialize all the variables
        result = f.eval()

print('This is the result:', result)
