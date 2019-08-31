import sys

sys.path.append('/Users/huanghaihun/Documents/keras-qqwwee')
import unittest
import tensorflow as tf

tf.enable_eager_execution()
import numpy as  np
from yolo3.network import Matching


class MyTestCase(unittest.TestCase):
    #
    # def test_attention(self):
    #     style = tf.ones((2, 2))
    #     inputs = tf.ones((2, 4, 4, 2))
    #     match = Matching(style, f_dim=2)
    #     r = match(inputs)
    #     print(tf.shape(r), tf.shape(inputs))

    def test_einsum(self):
        a = tf.ones((2, 3, 3, 2))
        b = tf.ones((1, 1, 3, 4))

        c = tf.einsum("...in,...ih->...nh", a, b)
        print(c)

    def test_vanilla_channel(self):
        q = tf.ones((2, 1, 1, 8))
        m = tf.ones((2, 4, 4, 8))
        w = tf.ones((16, 8))
        r = Matching.multi_head_vanilla_channel_attention(w, q=q, k=m, v=m, dim_per_head=2, num_head=4)
        print(tf.shape(r))

    def test_vanilla_position(self):
        q = tf.ones((2, 1, 1, 8))
        m = tf.ones((2, 4, 4, 8))
        w = tf.ones((16, 4))
        r = Matching.multi_head_vanilla_position_attention(w, q=q, k=m, v=m, dim_per_head=2, num_head=4)
        print(tf.shape(r))


if __name__ == '__main__':
    unittest.main()
