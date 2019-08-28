"""
Retrain the YOLO model for your own dataset.
"""
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras_radam import RAdam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import initializers
from keras.engine import Layer, InputSpec
# from tensorflow.python.ops import gen_nccl_ops
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss, yolo_body_adain, needle_preprocess
from yolo3.utils import get_random_data

import horovod.keras as hvd

import os

hvd.init()
os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
K.set_session(tf.Session(config=config))

def eval():
    # annotation_path = 'data/holes.csv'
    # log_dir = 'logs/holes/'
    # classes_path = 'model_data/wood_board.txt'
    # anchors_path = 'model_data/wood_anchors.txt'
    # weight_path = 'logs/holes/trained_weights_final.h5'
    annotation_path = 'data/stdogs/stdogs_1.csv'
    log_dir = 'logs/stdogs/'
    classes_path = 'model_data/stdogs_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    weight_path = 'logs/stdogs/ep030-loss12.408-val_loss17.981.h5'

    epoch = 200

    epoch = epoch
    val_split = 0.1
    # class_names = get_classes(classes_path)
    from sdog_annotation import train_classes as class_names
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting

    model = create_model_adain(input_shape, anchors, num_classes, weights_path=weight_path)  # make sure you know what you freeze
    verbose = 1 if hvd.rank() == 0 else 0
    print("verbose:",verbose)
    logging = TensorBoard(log_dir=log_dir)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)


    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        optimizer = Adam(lr=1e-5 * hvd.size(), clipvalue=1e1)
        optimizer = hvd.DistributedOptimizer(optimizer)

        model.compile(optimizer=optimizer,
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32  # note that more GPU memory is required after unfreezing the body

        model.evaluate_generator(data_generator_wrapper(lines, batch_size, input_shape, anchors,
                                                                   num_classes,random=True),steps=1)


def test():
    # annotation_path = 'data/holes.csv'
    # log_dir = 'logs/holes/'
    # classes_path = 'model_data/wood_board.txt'
    # anchors_path = 'model_data/wood_anchors.txt'
    # weight_path = 'logs/holes/trained_weights_final.h5'
    annotation_path = 'data/stdogs/stdogs_debug.csv'
    log_dir = 'logs/stdogs/'
    classes_path = 'model_data/stdogs_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    weight_path = 'logs/stdogs/ep030-loss12.408-val_loss17.981.h5'

    epoch = 200

    epoch = epoch
    val_split = 0.1
    # class_names = get_classes(classes_path)
    from sdog_annotation import train_classes as class_names
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    num_anchors = len(anchors)
    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting

    model = create_model_eval(num_anchors, num_classes, weights_path=weight_path)  # make sure you know what you freeze

    verbose = 1 if hvd.rank() == 0 else 0

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    batch_size = 1
    generator = data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                       num_classes)
    data = generator.__next__()
    data = data[0][:4]
    pred = model.predict(data)

    data1 = [*data]
    data1[1] = np.zeros(data[1].shape)


def _main():
    # annotation_path = 'data/holes.csv'
    # log_dir = 'logs/holes/'
    # classes_path = 'model_data/wood_board.txt'
    # anchors_path = 'model_data/wood_anchors.txt'
    # weight_path = 'logs/holes/trained_weights_final.h5'
    annotation_path = 'data/stdogs/stdogs_1.csv'
    log_dir = 'logs/stdogs/'
    classes_path = 'model_data/stdogs_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    weight_path = None

    epoch = 200

    epoch = epoch
    val_split = 0.1
    # class_names = get_classes(classes_path)
    from sdog_annotation import train_classes as class_names
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting

    model = create_model_adain(input_shape, anchors, num_classes, weights_path=weight_path)  # make sure you know what you freeze
    verbose = 1 if hvd.rank() == 0 else 0
    print("verbose:",verbose)
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        optimizer = RAdam(lr=1e-5 * hvd.size(), clipvalue=1e1)
        optimizer = hvd.DistributedOptimizer(optimizer)

        model.compile(optimizer=optimizer,
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 2  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0)
            ,hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=verbose)
            , reduce_lr
        ]
        if hvd.rank() == 0:
            callbacks += [ checkpoint ,logging,  early_stopping]

        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size // hvd.size()),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            verbose=verbose,
                            validation_steps=max(1, num_val // batch_size // hvd.size()),

                            epochs=epoch,
                            initial_epoch=0,

                            callbacks=callbacks)
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


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

def create_model_eval(num_anchors, num_classes,
                       weights_path='logs/holes/ep132-loss114.564-val_loss122.172.h5', max_box_length=20,
                       needle_size=64):
    '''create the training model'''
    import tensorflow as tf
    # K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    class_input = Input(shape=[1], dtype=tf.int32)
    needle_embedding = needle_preprocess(max_box_length=max_box_length, image_size=needle_size)
    model_body = yolo_body_adain(image_input, needle_embedding.inputs, needle_embedding.output, class_input, num_anchors // 3, num_classes)
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return Model([*model_body.inputs, class_input], model_body.outputs)



def create_model_adain(input_shape, anchors, num_classes,
                       weights_path=None, max_box_length=20,
                       needle_size=64, deprected_num_classes=1):
    '''create the training model'''
    from keras.layers import Reshape
    import tensorflow as tf
    K.clear_session()  # get a new session

    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, deprected_num_classes + 5)) for l in range(3)]
    image_input = Input(shape=(None, None, 3))
    class_input = Input(shape=[1], dtype=tf.int32)

    needle_embedding = needle_preprocess(max_box_length=max_box_length, image_size=needle_size)
    model_body = yolo_body_adain(image_input, needle_embedding.inputs, needle_embedding.output, class_input, num_anchors // 3, num_classes)
    if weights_path is not None:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, deprected_num_classes))
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': deprected_num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([*model_body.inputs, class_input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, size , rank, random=True):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        box_image_data = []
        box_len_data = []
        class_picked_data = []
        for b in range(batch_size * size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            if b % size != rank:
                continue
            image, box, box_images, box_len, class_picked = get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            box_image_data.append(box_images)
            box_len_data.append(box_len)
            class_picked_data.append(class_picked)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        box_image_data = np.stack(box_image_data, 0)
        box_len_data = np.stack(box_len_data, 0)
        class_picked_data = np.stack(class_picked_data, 0 )
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, box_image_data, box_len_data, class_picked_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, random=True):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, hvd.size(), hvd.rank()
                          , random=random)


if __name__ == '__main__':
    _main()
    # eval()