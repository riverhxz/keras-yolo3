
import sys
sys.path.append('/Users/huanghaihun/Documents/keras-qqwwee')
import unittest
import tensorflow as tf
# tf.enable_eager_execution()
from yolo3.model import _scatter_moving_avg
class MyTestCase(unittest.TestCase):
    def test_something(self):

        ref = tf.Variable(initial_value=tf.zeros(1,dtype=tf.float32))
        update = tf.assign(ref, tf.ones(1,dtype=tf.float32))
        with tf.control_dependencies([update]):
            b = ref * 1

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            res = sess.run(b)

        print(res)


if __name__ == '__main__':
    unittest.main()
