from keras.engine import Layer
from keras.layers import Conv2D, initializers, Dense
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras import backend as K

from keras.layers.advanced_activations import LeakyReLU

from keras.regularizers import l2
from functools import wraps
# from yolo3.model import DarknetConv2D_BN_Leaky
from yolo3.utils import compose


def shape_t(s):
    print(tf.shape(s))


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}

    normalizer = kwargs.get('norm')
    no_bias_kwargs.update(kwargs)

    if normalizer is not None:
        no_bias_kwargs.pop("norm")
        module = compose(
            DarknetConv2D(*args, **no_bias_kwargs),
            normalizer(),
            LeakyReLU(alpha=0.1))
    else:
        module = compose(
            DarknetConv2D(*args, **no_bias_kwargs),
            LeakyReLU(alpha=0.1))

    return module


class Matching(Layer):
    def __init__(self, style_coding,
                 **kwargs):
        super(Matching, self).__init__(**kwargs)
        self.style_coding = style_coding

    @staticmethod
    def _channel_wised_attention(q, k, d):
        q1 = tf.broadcast_to(q, tf.shape(k))
        a = tf.einsum("nwhc,nwhc->nc", k, q1) / tf.sqrt(d * 1.0)
        a = tf.nn.softmax(a, -1)

        for _ in range(2):
            a = K.expand_dims(a, 1)
        return a

    @staticmethod
    def _position_wised_attention(q, k, d):

        q1 = tf.broadcast_to(q, tf.shape(k))
        shape_k = tf.shape(k)
        a = tf.einsum("nwhc,nwhc->nwh", k, q1) / tf.sqrt(d * 1.0)

        a = tf.reshape(a, [shape_k[0], shape_k[1] * shape_k[2]])
        a = tf.nn.softmax(a, 1)
        a = tf.reshape(a, [shape_k[0], shape_k[1], shape_k[2], 1])

        return a

    @staticmethod
    def _no_bias_leaky_dense(dim):
        return Dense(dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")

    def apply_all_attentions(self, q, k, v, d):
        for _ in range(2):
            q = K.expand_dims(q, 1)
        channel_attention = self._channel_wised_attention(q, k, d)

        position_attention = self._position_wised_attention(q, k, d)

        return v * channel_attention * position_attention

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        fout_dim = input_shape[-1]
        x = Dense(fout_dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")(self.style_coding)
        x = self.apply_all_attentions(x, inputs, inputs, fout_dim)
        x = DarknetConv2D_BN_Leaky(fout_dim, (1, 1))(x)
        return x

class MatchingVanilla(Layer):
    def __init__(self, style_coding
                 , dim_per_head
                 , num_head
                 ,**kwargs):
        super(MatchingVanilla, self).__init__(**kwargs)
        self.style_coding = style_coding
        self.dim_per_head = dim_per_head
        self.num_head = num_head
        init = initializers.get("he_normal")
        self.w_channel_vanilla_attention = self.add_weight("w_vanilla_attention",
                                                           [self.dim_per_head * 2, self.dim_per_head],initializer=init)
        self.w_positional_vanilla_attention = self.add_weight("w_vanilla_attention",
                                                           [self.dim_per_head * 2, self.num_head],initializer=init)

    @staticmethod
    def _channel_wised_attention(q, k, d):
        q1 = tf.broadcast_to(q, tf.shape(k))
        a = tf.einsum("nwhc,nwhc->nc", k, q1) / tf.sqrt(d * 1.0)
        a = tf.nn.softmax(a, -1)

        for _ in range(2):
            a = K.expand_dims(a, 1)
        return a

    @staticmethod
    def _position_wised_attention(q, k, d):

        q1 = tf.broadcast_to(q, tf.shape(k))
        shape_k = tf.shape(k)
        a = tf.einsum("nwhc,nwhc->nwh", k, q1) / tf.sqrt(d * 1.0)

        a = tf.reshape(a, [shape_k[0], shape_k[1] * shape_k[2]])
        a = tf.nn.softmax(a, 1)
        a = tf.reshape(a, [shape_k[0], shape_k[1], shape_k[2], 1])

        return a

    @staticmethod
    def _no_bias_leaky_dense(dim):
        return Dense(dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")

    @staticmethod
    def multi_head_vanilla_channel_attention(w, q, k, v, dim_per_head, num_head):
        dim_total = dim_per_head * num_head

        q = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(q)
        k = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(k)
        v = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(v)

        shape_q, shape_k, shape_v = tf.shape(q), tf.shape(k), tf.shape(v)

        q = tf.broadcast_to(q, shape_k)

        a = tf.concat([q, k], axis=3)
        for _ in range(3):
            w = tf.expand_dims(w, 0)
        w = tf.broadcast_to(w, [shape_k[0], shape_k[1], shape_k[2], dim_total * 2, dim_total])

        a = tf.einsum("awhd,awhdc->ac", a, w) / tf.sqrt(dim_per_head * 1.0)
        # for tf 1.14
        # a = tf.einsum("...whd,...whdc->...c", a, w) / tf.sqrt(dim_per_head * 1.0)
        a = tf.reshape(a, [shape_k[0], 1, 1, dim_per_head, num_head])
        a = tf.nn.softmax(a, 3)
        a = tf.reshape(a, [shape_k[0], 1, 1, dim_per_head * num_head])
        o = a * v
        return o

    @staticmethod
    def multi_head_vanilla_position_attention(w, q, k, v, dim_per_head, num_head):
        dim_total = dim_per_head * num_head

        q = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(q)
        k = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(k)
        v = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(v)

        shape_q, shape_k, shape_v = tf.shape(q), tf.shape(k), tf.shape(v)

        q = tf.broadcast_to(q, shape_k)

        # compute attention of position
        a = tf.concat([q, k], axis=3)
        for _ in range(3):
            w = tf.expand_dims(w, 0)
        w = tf.broadcast_to(w, [shape_k[0], shape_k[1], shape_k[2], dim_total * 2, num_head])

        a = tf.einsum("awhd,awhdc->awhc", a, w) / tf.sqrt(dim_per_head * 1.0)
        # a = tf.einsum("...whd,...whdc->...whc", a, w) / tf.sqrt(dim_per_head * 1.0)
        # normalize attention
        a = tf.reshape(a, [shape_k[0], shape_k[1] * shape_k[2], num_head])
        a = tf.nn.softmax(a, 1)
        # compute output
        a = tf.reshape(a, [shape_k[0], shape_k[1], shape_k[2], num_head, 1])
        o = a * tf.reshape(v, [shape_k[0], shape_k[1], shape_k[2], num_head, dim_per_head])
        o = tf.reshape(o, shape_v)

        return o

    def apply_all_attentions(self, q, k, v):
        for _ in range(2):
            q = K.expand_dims(q, 1)
        channel_attention = self.multi_head_vanilla_channel_attention(
            self.w_channel_vanilla_attention, q, k, v, self.dim_per_head, self.num_head)
        position_attention = self.multi_head_vanilla_position_attention(
            self.w_positional_vanilla_attention, q, k, v, self.dim_per_head, self.num_head)

        return tf.concat([channel_attention, position_attention], axis=3)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        fout_dim = input_shape[-1]
        x = Dense(fout_dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")(self.style_coding)
        x = self.apply_all_attentions(x, inputs, inputs)
        x = DarknetConv2D_BN_Leaky(fout_dim, (1, 1))(x)
        return x
