
""" QuickNAT Network.
References:
    Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds. arXiv preprint arXiv:1801.04161.
Link to the paper: 
    https://arxiv.org/abs/1801.04161
"""

import tensorflow as tf
import numpy as np

def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:  max pooled output tensor
           ind:   argmax indices
           ksize: ksize is the same as for the pool
       Return:
           ret:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        # print(set_output_shape)
        return ret

def dense_block(input_dense, training):
    """
       Dense block consisting of three convolutional layers 
       Args:
           input_dense:  max pooled tensor
           training:   whether to return the output in training mode
       Return:
           output_feature_map
    """
    net = input_dense  # 256,256,1
    # dense_layer = tf.layers.dense(net,units = 64, activation=tf.nn.relu)
    conv1 = tf.layers.batch_normalization(net, training=training)
    conv1 = tf.nn.relu(conv1)  # 256,256,64
    feature_map1_conv1 = tf.layers.conv2d(net, 64, (5, 5), activation=None, padding='same')  # 256,256,64
    concat1 = tf.concat([feature_map1_conv1, input_dense], axis=3)  # 256,256,128

    conv2 = tf.layers.batch_normalization(concat1, training=training)
    conv2 = tf.nn.relu(conv2)  # 256,256,64
    feature_map2_conv2 = tf.layers.conv2d(conv2, 64, (5, 5), activation=None, padding='same')  # 256,256,64
    concat2 = tf.concat([feature_map2_conv2, feature_map1_conv1, input_dense], axis=3)  # 256,256,128

    conv_out = tf.layers.batch_normalization(concat2, training=training)
    conv_out = tf.nn.relu(conv_out)  # 256,256,64
    output_feature_map = tf.layers.conv2d(conv_out, 64, (1, 1), activation=None, padding='same')  # 256,256,64
    
    return output_feature_map  # 256,256,64

def quick_nat(X, training, num_classes=3):
    """
       QuickNAT Network 
       Args:
           X:  Input Image
           training: whether to return the output in training mode
           num_classes: number of classes
    """

    ## Encoder Layers
    net = X  #Input(256,256,1)

    dense_block1 = dense_block(net, training)  # 256,256,64
    pool1, argmax_1 = tf.nn.max_pool_with_argmax(dense_block1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')  # 128,128,64

    dense_block2 = dense_block(pool1, training)
    pool2, argmax_2 = tf.nn.max_pool_with_argmax(dense_block2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')  # 64,64,64
    
    dense_block3 = dense_block(pool2, training)
    pool3, argmax_3 = tf.nn.max_pool_with_argmax(dense_block3, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')  # 32,32,64

    dense_block4 = dense_block(pool3, training)
    pool4, argmax_4 = tf.nn.max_pool_with_argmax(dense_block4, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')  # 16,16,64
    
    ## Bottleneck Layer
    bottleneck_layer = tf.layers.conv2d(pool4, 64, (5, 5), activation=None, padding='same')
    bottleneck_layer = tf.layers.batch_normalization(bottleneck_layer, training=training)  # 16,16,64

    ## Decoder Layers
    up5 = tf.concat([unpool(bottleneck_layer, argmax_4), dense_block4], axis=3)  # 32,32,128
    dense_block5 = dense_block(up5, training)  # 32,32,64

    up6 = tf.concat([unpool(dense_block5, argmax_3), dense_block3], axis=3)  # 64,64,128
    dense_block6 = dense_block(up6, training)  # 32,32,64

    up7 = tf.concat([unpool(dense_block6, argmax_2), dense_block2], axis=3)  # 256,256,128
    dense_block7 = dense_block(up7, training)  # 32,32,64

    up8 = tf.concat([unpool(dense_block7, argmax_1), dense_block1], axis=3)  # 256,256,128
    dense_block8 = dense_block(up8, training)  # 32,32,64

    ## Classifier Block
    classifier = tf.layers.conv2d(dense_block8, num_classes, (1, 1), activation=None, padding='same')
    output = tf.nn.softmax(classifier)

    return output  # 512,512,3