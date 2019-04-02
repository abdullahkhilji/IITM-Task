#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


def custom_ops(n):
    """
    Custom matrix operations:

    :param n: length of 2D square tensor type tf.float32
    :return: finalVal
    """

    # Step 1: Transpose the elements in the bottom-right triangle of A

    reverse_of_mat = tf.reverse(matA, [1])
    diagonal_of_mat = tf.linalg.band_part(reverse_of_mat, 0, 0)
    lower_triangular_with_diagonal = tf.linalg.band_part(reverse_of_mat, -1, 0)
    lower_triangular_without_diagonal = tf.subtract(lower_triangular_with_diagonal, diagonal_of_mat)
    transposed_lower_triangle = tf.transpose(lower_triangular_without_diagonal)
    required_lower_triangle = tf.reverse(transposed_lower_triangle, [0])
    reverse_of_mat_minus_required = tf.subtract(reverse_of_mat, lower_triangular_without_diagonal)
    mat_minus_required = tf.reverse(reverse_of_mat_minus_required, [1])
    step_one = tf.add(required_lower_triangle, mat_minus_required)




    # matrix = tf.placeholder(dtype=tf.float32, shape=(n, n), name="matA")

    # x = tf.constant(list(range(n)))
    # c = lambda i, x: i != n/2
    # b = lambda i, x: (matrix[i, i], matrix[i, n - i - 1] == matrix[i, n - i - 1], matrix[i, i])

    # bottom_right_matrix = tf.matrix_band_part(matrix, -1, 0)
    # bottom_right_transpose_matrix = tf.transpose(bottom_right_matrix)

    # transpose_matrix = tf.transpose(matrix)
    # vector_m = tf.reduce_max(matrix, reduction_indices=[1])
    #
    # reshaped = tf.broadcast_to(vector_m, [n, n])
    #
    # up_traingle = tf.matrix_band_part(reshaped, -1, 0)
    # reverse = tf.reverse(up_traingle, [-n+1])
    #
    #
    # softmax_mat = tf.nn.softmax(reverse)

    return step_one


if __name__ == '__main__':
    mat = np.asarray([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])
    n = mat.shape[0]
    matA = tf.placeholder(tf.float32, shape=(n, n))
    finalVal = custom_ops(n)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    outVal = sess.run(finalVal, feed_dict={matA: mat})
    print(outVal)
    sess.close()

