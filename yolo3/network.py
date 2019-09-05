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

#
# class Matching(Layer):
#     def __init__(self, style_coding, **kwargs):
#         super(Matching, self).__init__(**kwargs)
#         self.style_coding = style_coding
#
#     @staticmethod
#     def _channel_wised_attention(q, k, d):
#         # q = DarknetConv2D_BN_Leaky(d, (1, 1))(q)
#         # k = DarknetConv2D_BN_Leaky(d, (1, 1))(k)
#         q = tf.squeeze(q,[1,2])
#         a = tf.einsum("nwhc,nc->nc", k, q) / tf.sqrt(d * 1.0)
#         a = tf.nn.softmax(a, -1)
#         for _ in range(2):
#             a = K.expand_dims(a, 1)
#         return a
#
#     @staticmethod
#     def _position_wised_attention(q, k, d):
#
#         # q = DarknetConv2D_BN_Leaky(d, (1, 1))(q)
#         # k = DarknetConv2D_BN_Leaky(d, (1, 1))(k)
#         q = tf.squeeze(q, [1, 2])
#         shape_k = tf.shape(k)
#         a = tf.einsum("nwhc,nc->nwh", k, q) / tf.sqrt(d * 1.0)
#         a = tf.reshape(a, [shape_k[0], shape_k[1] * shape_k[2]])
#         a = tf.nn.softmax(a, 1)
#         a = tf.reshape(a, [shape_k[0], shape_k[1], shape_k[2], 1])
#
#         return a
#
#     @staticmethod
#     def _no_bias_leaky_dense(dim):
#         return compose([
#             Dense(dim, use_bias=False, kernel_initializer="he_normal"),
#             LeakyReLU()
#         ])
#
#     def apply_all_attentions(self, q, k, v, d):
#         for _ in range(2):
#             q = K.expand_dims(q, 1)
#
#         channel_attention = self._channel_wised_attention(q, k, d)
#
#         position_attention = self._position_wised_attention(q, k, d)
#
#         return v * channel_attention * position_attention
#
#     def call(self, inputs, training=None):
#         input_shape = K.int_shape(inputs)
#         fout_dim = input_shape[-1]
#         x = compose(
#             Dense(fout_dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")
#             ,Dense(2*fout_dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")
#             ,Dense(fout_dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")
#         )(self.style_coding )
#         x = self.apply_all_attentions(x, inputs, inputs, fout_dim)
#         # x = DarknetConv2D_BN_Leaky(fout_dim, (1, 1))(x)
#         return x

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
        # for _ in range(2):
        #     q = K.expand_dims(q, 1)
        # q1 = tf.broadcast_to(q, tf.shape(k))
        a = tf.einsum("nwhc,nc->nc", k, q) / tf.sqrt(d * 1.0)
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
        return v * channel_attention * position_attention

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        fout_dim = input_shape[-1]
        x = Dense(fout_dim, use_bias=False, activation='relu')(self.style_coding)
        x = self.apply_all_attentions(x, inputs, inputs, fout_dim)
        return x


#
# class Matching(Layer):
#     def __init__(self, style_coding, f_dim, eps=1e-5,
#                  **kwargs):
#         super(Matching, self).__init__(**kwargs)
#         self.style_coding = style_coding
#         self.eps = eps
#
#     @staticmethod
#     def _channel_wised_attention(q, k, d):
#         for _ in range(2):
#             q = K.expand_dims(q, 1)
#         q1 = tf.broadcast_to(q, tf.shape(k))
#         a = tf.einsum("nwhc,nwhc->nc", k, q1) / tf.sqrt(d * 1.0)
#         a = tf.nn.softmax(a, -1)
#
#         for _ in range(2):
#             a = K.expand_dims(a, 1)
#         return a
#
#     @staticmethod
#     def _position_wised_attention(q, k, d):
#         for _ in range(2):
#             q = K.expand_dims(q, 1)
#         q1 = tf.broadcast_to(q, tf.shape(k))
#         shape_k = tf.shape(k)
#         a = tf.einsum("nwhc,nwhc->nwh", k, q1) // tf.sqrt(d * 1.0)
#
#         a = tf.reshape(a, [shape_k[0],shape_k[1]*shape_k[2]])
#         a = tf.nn.softmax(a, 1)
#         a = tf.reshape(a, [shape_k[0],shape_k[1],shape_k[2], 1])
#         return a
#
#     def apply_all_attentions(self, q, k, v, d):
#         channel_attention = self._channel_wised_attention(q, k, d)
#
#         position_attention = self._position_wised_attention(q, k, d)
#         return v * channel_attention * position_attention
#
#     def call(self, inputs, training=None):
#         input_shape = K.int_shape(inputs)
#         fout_dim = input_shape[-1]
#         x = Dense(fout_dim, use_bias=False, activation='relu')(self.style_coding)
#         x = self.apply_all_attentions(x, inputs, inputs, fout_dim)
#         return x


class MatchingVanilla(Layer):
    def __init__(self, style_coding
                 , dim_per_head
                 , num_head
                 , **kwargs):
        super(MatchingVanilla, self).__init__(**kwargs)
        self.style_coding = style_coding
        self.dim_per_head = dim_per_head
        self.num_head = num_head
        init = initializers.get("he_normal")
        dim_total = self.dim_per_head * self.num_head
        self.w_channel_vanilla_attention = self.add_weight("w_vanilla_attention",
                                                           [self.num_head, dim_per_head * 2, dim_per_head],
                                                           initializer=init)
        # self.w_positional_vanilla_attention = self.add_weight("w_vanilla_attention",
        #                                                    [dim_total * 2, self.num_head],initializer=init)

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
    def multi_head_vanilla_channel_attention(q, k, v, dim_per_head, num_head, w):
        dim_total = dim_per_head * num_head

        q = no_bias_leaky_dense(dim_total)(q)
        k = DarknetConv2D(dim_total, (1, 1))(k)
        v = DarknetConv2D(dim_total, (1, 1))(v)

        shape_q, shape_k, shape_v = tf.shape(q), tf.shape(k), tf.shape(v)

        k = tf.reduce_sum(k, [1, 2])

        a = tf.concat([q, k], axis=1)
        a = tf.reshape(a, [shape_k[0], num_head, dim_per_head * 2])

        """
        b - batch_size
        n - number head
        c - input channel per head
        e - output channel per head
        """
        a = tf.einsum("bnc,nce->bne", a, w) / tf.sqrt(dim_per_head * 1.0)

        # a = tf.reshape(a, [shape_k[0], 1, 1, dim_per_head, num_head])
        # print("a",tf.shape(a))
        a = tf.nn.softmax(a, 2)
        a = tf.reshape(a, [shape_k[0], 1, 1, dim_per_head * num_head])
        o = a * v
        return o

    @staticmethod
    def multi_head_vanilla_position_attention(q, k, v, dim_per_head, num_head):
        dim_total = dim_per_head * num_head

        q = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(q)
        k = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(k)
        v = DarknetConv2D_BN_Leaky(dim_total, (1, 1))(v)

        shape_q, shape_k, shape_v = tf.shape(q), tf.shape(k), tf.shape(v)

        q = tf.broadcast_to(q, shape_k)

        # compute attention of position
        a = tf.concat([q, k], axis=3)
        a = no_bias_leaky_dense(num_head)(a)
        # normalize attention
        a = tf.reshape(a, [shape_k[0], shape_k[1] * shape_k[2], num_head])
        a = tf.nn.softmax(a, 1)
        # compute output
        a = tf.reshape(a, [shape_k[0], shape_k[1], shape_k[2], num_head, 1])
        o = a * tf.reshape(v, [shape_k[0], shape_k[1], shape_k[2], num_head, dim_per_head])
        o = tf.reshape(o, shape_v)

        return o

    def apply_all_attentions(self, q, k, v):
        qe = q
        for _ in range(2):
            qe = K.expand_dims(qe, 1)
        channel_attention = self.multi_head_vanilla_channel_attention(
            q, k, v, self.dim_per_head, self.num_head, self.w_channel_vanilla_attention)
        # position_attention = self.multi_head_vanilla_position_attention(
        #     qe, k, v, self.dim_per_head, self.num_head)

        return channel_attention
        # return tf.concat([channel_attention, position_attention], axis=3)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        fout_dim = input_shape[-1]
        x = Dense(fout_dim, use_bias=False, activation=LeakyReLU(), kernel_initializer="he_normal")(
            self.style_coding)
        x = self.apply_all_attentions(x, inputs, inputs)
        # x = DarknetConv2D_BN_Leaky(fout_dim, (1, 1))(x)
        return x


def no_bias_leaky_dense(dim):
    return compose(
        Dense(dim, use_bias=False, activation=None, kernel_initializer="he_normal")
        # ,LeakyReLU()
    )
