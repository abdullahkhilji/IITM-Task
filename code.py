#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


def custom_ops(n):
    """
    Custom matrix operations broken into 8 (+ 1 return value) steps:

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

    step_1 = tf.add(required_lower_triangle, mat_minus_required)

    # Step 2: Take the maximum value along the columns of A to get a vector m
    #         (i.e. for each column, pick a value that is the maximum among all rows)

    step_2 = m = tf.reduce_max(step_1, axis=0)

    # Step 3: Consider m to be of the form [m1, m2, ... , mn]. Create a new matrix B such that:

    broadcasted_m = tf.broadcast_to(m, [n, n])
    reverse_m = tf.reverse(broadcasted_m, [1])
    upper_traingular_reverse_m = tf.linalg.band_part(reverse_m, 0, -1)
    reverse_upper_triangular_reverse_m = tf.reverse(upper_traingular_reverse_m, [1])

    inf_m = tf.fill([n, n], float("inf"))
    lower_triangular_inf_m = tf.linalg.band_part(inf_m, -1, 0)
    right_lower_triangular_inf_m = tf.reverse(lower_triangular_inf_m, [0])

    with_nan = lower_triangular_without_diagonal_inf_m = tf.subtract(inf_m, right_lower_triangular_inf_m)
    replaced_zero = tf.where(tf.is_nan(with_nan), tf.zeros([n, n]), with_nan)

    to_softmax = tf.subtract(reverse_upper_triangular_reverse_m, replaced_zero)

    step_3 = B = tf.nn.softmax(to_softmax)

    # Step 4: Sum along the rows of B to obtain vector v1

    step_4 = v1 = tf.reduce_sum(B, 1)

    # Step 5: Sum along the columns of B to get another vector v2

    step_5 = v2 = tf.reduce_sum(B, 0)

    # Step 6: Concatenate the two vectors and take a softmax of this vector

    concat = tf.concat([v1, v2], 0)

    step_6 = v = tf.nn.softmax(concat)

    # Step 7: Get the index number in vector v with maximum value

    step_7 = index_number = tf.to_float(tf.argmax(v))

    # Step 8: If the index number is greater than n/3 store:
    #         finalVal = ||v1 − v2||2
    #         Otherwise, store:
    #         finalVal = ||v1 + v2||2

    n_by_3 = tf.constant(n/3)

    finalVal = tf.cond(index_number > n_by_3, lambda: tf.norm(v1 - v2), lambda: tf.norm(v1 + v2))

    # Step 9: Return the variable finalVal

    return finalVal


if __name__ == '__main__':
    mat = np.asarray([[1.0, 2.0, 3.0],
                      [0.4, 0.5, 0.6],
                      [0.7, 0.8, 0.9]])
    n = mat.shape[0]
    matA = tf.placeholder(tf.float32, shape=(n, n))
    finalVal = custom_ops(n)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    outVal = sess.run(finalVal, feed_dict={matA: mat})
    print(outVal)
    sess.close()

