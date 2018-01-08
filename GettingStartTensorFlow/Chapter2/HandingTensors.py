import matplotlib.image as mp_image
import matplotlib.pyplot as plt

filename = "packt.jpg"
input_image = mp_image.imread(filename)

print("input dim = {}".format(input_image.ndim))
print("input.shape = {}".format(input_image.shape))

# plt.imshow(input_image)
# plt.show()

import tensorflow as tf

my_image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(my_image, [45, 0, 0], [16, -1, -1])
with tf.Session() as sess:
    result = sess.run(slice, feed_dict={my_image: input_image})
    print(result.shape)

plt.imshow(result)
plt.show()
