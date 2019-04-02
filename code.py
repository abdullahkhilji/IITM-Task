#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


def customOps(n):
    """
    Define custom tensorflow operations
    """
    # Define placeholder
    # perform tf operations listed in list
    # Even if condition, matrix manipulaitons should be in Tensorflow

    matrix = tf.placeholder(dtype=tf.float32, shape=(n, n), name="matA")

    x = tf.constant(list(range(n)))
    c = lambda i, x: i != n/2
    b = lambda i, x: (matrix[i, i], matrix[i, n - i - 1] == matrix[i, n - i - 1], matrix[i, i])






    bottom_right_matrix = tf.matrix_band_part(matrix, -1, 0)
    bottom_right_transpose_matrix = tf.transpose(bottom_right_matrix)

    transpose_matrix = tf.transpose(matrix)
    vector_m = tf.reduce_max(matrix, reduction_indices=[1])

    reshaped = tf.broadcast_to(vector_m, [n, n])

    up_traingle = tf.matrix_band_part(reshaped, -1, 0)
    reverse = tf.reverse(up_traingle, [-n+1])


    softmax_mat = tf.nn.softmax(reverse)

    return bottom_right_matrix








if __name__ == '__main__':
    mat = np.asarray([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    mat1 = np.random.rand(10, 10)
    n = mat.shape[0]

    finalVal = customOps(n)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    outVal = sess.run(finalVal, feed_dict={tf.get_default_graph().get_operation_by_name('matA').outputs[0]: mat})
    print(outVal)
    sess.close()

