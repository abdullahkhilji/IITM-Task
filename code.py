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
    pass


if __name__ == '__main__':
    mat = np.asarray([[0, 1],
                      [1, 0]])
    n = mat.shape[0]

    finalVal = customOps(n)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    outVal = sess.run(finalVal, feed_dict={"matA": mat})
    print(outVal)
    sess.close()
