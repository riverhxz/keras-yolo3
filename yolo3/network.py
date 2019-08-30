from keras.engine import Layer
from keras.layers import Conv2D, initializers, Dense
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras import backend as K

from yolo3.utils import compose


def shape_t(s):
    print(tf.shape(s))


class Matching(Layer):
    def __init__(self, style_coding, f_dim, eps=1e-5,
                 **kwargs):
        super(Matching, self).__init__(**kwargs)
        self.style_coding = style_coding
        self.eps = eps

    @staticmethod
    def _channel_wised_attention(q, k, d):
        for _ in range(2):
            q = K.expand_dims(q, 1)
        q1 = tf.broadcast_to(q, tf.shape(k))
        a = tf.einsum("nwhc,nwhc->nc", k, q1) / tf.sqrt(d * 1.0)
        a = tf.nn.softmax(a, -1)

        for _ in range(2):
            a = K.expand_dims(a, 1)
        return a

    @staticmethod
    def _position_wised_attention(q, k, d):
        for _ in range(2):
            q = K.expand_dims(q, 1)
        q1 = tf.broadcast_to(q, tf.shape(k))
        shape_k = tf.shape(k)
        a = tf.einsum("nwhc,nwhc->nwh", k, q1) // tf.sqrt(d * 1.0)

        a = tf.reshape(a, [shape_k[0],shape_k[1]*shape_k[2]])
        a = tf.nn.softmax(a, 1)
        a = tf.reshape(a, [shape_k[0],shape_k[1],shape_k[2], 1])
        return a

    def apply_all_attentions(self, q, k, v, d):
        channel_attention = self._channel_wised_attention(q, k, d)

        position_attention = self._position_wised_attention(q, k, d)
        shape_t(position_attention)
        shape_t(channel_attention)
        return v * channel_attention * position_attention

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        fout_dim = input_shape[-1]
        x = Dense(fout_dim, use_bias=False, activation='relu')(self.style_coding)
        x = self.apply_all_attentions(x, inputs, inputs, fout_dim)
        return x
