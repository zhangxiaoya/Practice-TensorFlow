import numpy as np
import tensorflow as tf

tensor_2d = np.array([(1, 2, 3, 4), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)])
print(tensor_2d)
print(tensor_2d[2][3])
print(tensor_2d[0:2, 0:2])

matrix1 = np.array([(2, 2, 2), (2, 2, 2), (2, 2, 2)], dtype='int32')
matrix2 = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype='int32')
print("Matrix 1 = ")
print(matrix1)
print("Matrix 2 = ")
print(matrix2)

matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)

matrix_product = tf.matmul(matrix1, matrix1)
matrix_sum = tf.add(matrix1, matrix2)

matrix3 = np.array([(2, 7, 2), (1, 4, 2), (9, 0, 2)], dtype='float64')
print("Matrix3")
print(matrix3)

matrix_det = tf.matrix_determinant(matrix3)

with tf.Session() as sess:
    print(sess.run(matrix1))
    print(sess.run(matrix2))
    print(sess.run(matrix_product))
    print(sess.run(matrix_sum))
    print(sess.run(matrix_det))
