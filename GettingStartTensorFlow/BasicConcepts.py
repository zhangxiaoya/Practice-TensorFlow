x = 1
y = x + 9
print(y)
import tensorflow as tf
x = tf.constant(1, name='x')
y = tf.Variable(x + 9, name='y')
print(y)