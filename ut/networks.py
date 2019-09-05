import sys

sys.path.append('/Users/huanghaihun/Documents/keras-qqwwee')
import unittest
import tensorflow as tf

tf.enable_eager_execution()
import numpy as  np
from yolo3.network import MatchingVanilla,Matching


class MyTestCase(unittest.TestCase):
    #
    # def test_attention(self):
    #     style = tf.ones((2, 2))
    #     inputs = tf.ones((2, 4, 4, 2))
    #     match = Matching(style, f_dim=2)
    #     r = match(inputs)
    #     print(tf.shape(r), tf.shape(inputs))

    def test_einsum(self):
        a = tf.ones(( 3, 2))
        b = tf.ones((3, 3, 3, 2))

        c = tf.einsum("nd,nabd -> nd", a, b)
        d = tf.einsum("nd,nabd -> nab", a, b)


        print(c)
        print(d)
    def test_vanilla_channel(self):
        dim_per_head = 4
        num_head = 8
        total_dim = dim_per_head * num_head
        q = tf.ones((2,  total_dim))
        m = tf.ones((2, 4, 4, total_dim))
        w = tf.ones((num_head, dim_per_head*2, dim_per_head))
        r = MatchingVanilla.multi_head_vanilla_channel_attention( q=q, k=m, v=m, dim_per_head=dim_per_head, num_head=num_head, w=w)

        print('r:', r[0,0,0,0], tf.shape(r))

    def test_vanilla_position(self):
        q = tf.ones((2, 1, 1, 8))
        m = tf.ones((2, 4, 4, 8))
        r = MatchingVanilla.multi_head_vanilla_position_attention( q=q, k=m, v=m, dim_per_head=2, num_head=4)
        print(tf.shape(r))



if __name__ == '__main__':
    unittest.main()
