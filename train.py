"""
Retrain the YOLO model for your own dataset.
"""
from tqdm import *

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model
import tensorflow as tf

from yolo3 import model
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from datetime import datetime
from tqdm import *


def _main():
    from train import get_classes, get_anchors
    annotation_path = 'data/input.csv'
    log_dir = 'logs/000/'
    classes_path = 'model_data/stdogs_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    val_split = 0.5

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting

    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # save_interval = 10
    # num_val = 4
    # num_train = 4
    batch_size = 4

    train_data = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
    eval_data = data_generator_wrapper(lines[num_train:num_train + num_val], batch_size, input_shape, anchors,
                                       num_classes)

    body = model_body(input_shape, anchors, num_classes, None,
                      freeze_body=2,
                      weights_path='model_data/darknet53.weights.h5')

    num_epoch = 10
    optimizer = tf.keras.optimizers.Adam()
    tf.summary.experimental.set_step(0)

    def train_step(image, y1, y2, y3):

        with tf.GradientTape() as tape:
            outputs = body(image)
            loss = loss_wrapper(outputs, [y1, y2, y3], anchors, num_classes)
            grads = tape.gradient(loss, body.trainable_weights)
            optimizer.apply_gradients(zip(grads, body.trainable_variables))
            return loss

    def evaluate_step(image, y1, y2, y3):
        with tf.GradientTape() as tape:
            outputs = body(image)
            loss = loss_wrapper(outputs, [y1, y2, y3], anchors, num_classes)
            return loss

    for epoch in range(num_epoch):
        print("epoch:", epoch)
        with tqdm(train_data, total=num_train // batch_size) as tbar:
            for x in train_data.take(num_train // batch_size):
                image, y1, y2, y3 = x
                loss = train_step(image, y1, y2, y3)
                tbar.update(1)
                tbar.set_description("loss={:.3f}".format(loss))
        val_loss = 0
        for x in eval_data.take(num_val // batch_size):
            image, y1, y2, y3 = x
            val_loss += evaluate_step(image, y1, y2, y3)

        val_loss = val_loss / num_val
        print("val_loss:", val_loss)

        tf.saved_model.save(body, "model_data/tfmodel" )



def extends(true_class_probs):
    extend_true_class_probs = tf.concat(
        [true_class_probs, 1 - tf.reduce_sum(true_class_probs, axis=4, keepdims=True)], axis=4)
    return tf.argmax(extend_true_class_probs, axis=4)


def moveing_avg(variable, value, update_weight=0.05):
    return variable.assign(variable * (1 - update_weight) + update_weight * value)


def update(center, value, keys):
    z = extends(keys)
    ez = tf.cast(tf.expand_dims(z, 4), value[0].dtype)
    v_shape = tf.shape(value)
    reshape_value = tf.reshape(value, [*v_shape[:-1], 1, v_shape[-1]])
    group_bys = tf.reduce_mean(ez * reshape_value, axis=[0, 1, 2])
    return moveing_avg(center, group_bys)


def update_centers():
    gap = outputs[3:6]
    keys = [x[..., 5:7] for x in [y1, y2, y3]]
    for (center, v, k) in zip(
            model.model_global.centers
            , gap
            , keys
    ):
        update(center, v, k)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, update_callback, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5', input_tensor=None):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = input_tensor if input_tensor is not None else Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'update_callback': update_callback})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def model_body(input_shape, anchors, num_classes, update_callback, load_pretrained=True, freeze_body=0,
               weights_path='model_data/yolo_weights.h5'):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    return model_body


def loss_wrapper(outputs, pred, anchors, num_classes):
    return yolo_loss([*outputs, *pred], anchors=anchors, num_classes=num_classes, ignore_thresh=0.5, print_loss=False)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield (image_data, *y_true)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    generator = data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    return tf.data.Dataset.from_generator(
        lambda: generator
        , output_types=(tf.float32, tf.float32, tf.float32, tf.float32,)
        , output_shapes=(
            tf.TensorShape([None, 416, 416, 3])
            , tf.TensorShape([None, 13, 13, 3, 7])
            , tf.TensorShape((None, 26, 26, 3, 7))
            , tf.TensorShape((None, 52, 52, 3, 7))
        )
    )


if __name__ == '__main__':
    _main()
