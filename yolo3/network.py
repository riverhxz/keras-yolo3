from keras.engine import Layer
from keras.layers import Conv2D, initializers
import tensorflow as tf

from keras import backend as K

from yolo3.utils import compose


def shape_t(s):
    print(tf.shape(s))


class Matching(Layer):
    def __init__(self, style_coding, f_dim,
                 **kwargs):
        super(Matching, self).__init__(**kwargs)
        self.style_coding = style_coding
        self.attention_fun_weight = self.add_weight("attention_weight", (f_dim, f_dim),
                                                    initializer=initializers.he_normal())

    @staticmethod
    def _channel_wised_attention(weight, q, k, v):
        q1 = tf.matmul(q, weight)
        for _ in range(2):
            q1 = K.expand_dims(q1, 1)
        shape_t(q1)
        a = q1 * k
        a = tf.nn.softmax(a, -1)
        o = a * v
        return o

    def channel_wised_attention(self, q, k, v):
        return self._channel_wised_attention(self.attention_fun_weight, q, k, v)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        fout_dim = input_shape[-1]
        x = self.channel_wised_attention(self.style_coding, inputs, inputs)
        return x
