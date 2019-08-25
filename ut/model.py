
import sys
sys.path.append('/Users/huanghaihun/Documents/keras-qqwwee')
import unittest
import tensorflow as tf
tf.enable_eager_execution()
from yolo3.model import _scatter_moving_avg
class MyTestCase(unittest.TestCase):
    def test_something(self):

        ref = tf.Variable(initial_value=tf.reshape(tf.range(4,dtype=tf.float32), (2,2)))
        update = tf.ones([1,2])
        index = tf.zeros([1],dtype=tf.int32)

        _scatter_moving_avg(ref, index ,update, 0)

        print(tf.reduce_sum(ref[0, ...] - update ))

    def test_segment(self):
        indice = tf.constant([5, 1, 0])
        c = tf.constant([[1.0, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]])
        r = tf.unsorted_segment_mean(c, indice, 10)
        r1 = tf.gather(r, indice)
        print(r1)

if __name__ == '__main__':
    unittest.main()
