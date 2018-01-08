import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plt

filename = "packt.jpg"
input_image = mp_img.imread(filename)

x = tf.Variable(input_image, name='x')
model = tf.global_variables_initializer();
with tf.Session() as sess:
    x = tf.transpose(x, perm=[1, 0, 2])
    sess.run(model)
    result = sess.run(x)

plt.imshow(result)
plt.show()
