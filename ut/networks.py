import sys

sys.path.append('/Users/huanghaihun/Documents/keras-qqwwee')
import unittest
import tensorflow as tf

tf.enable_eager_execution()
import numpy as  np


class MyTestCase(unittest.TestCase):

    def test_attention(self):
        from yolo3.network import Matching
        style = tf.ones((2, 2))
        inputs = tf.ones((2, 4, 4, 2))
        match = Matching(style, f_dim=2)
        r = match(inputs)
        print(tf.shape(r), tf.shape(inputs))


if __name__ == '__main__':
    unittest.main()
