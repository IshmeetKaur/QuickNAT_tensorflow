import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import operator
import random

smooth = 1e-7

def dice_coef_(y_pred, y_true):
    """Returns a (approx) IOU score
    IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, C)
        y_true (4-D array): (N, H, W, C)
    Returns:
        float: IOU score
    """
    H, W, num_classes = y_pred.get_shape().as_list()[1:]
    pred_flat = tf.reshape(y_pred, [-1, H * W, num_classes])  # since 3 is the number of classes
    true_flat = tf.reshape(y_true, [-1, H * W, num_classes])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth  # shape(batch,row,width,num_classes)
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat,
                                                                   axis=1) + smooth  # shape(batch,row,width,num_classes)
    return tf.reduce_mean(intersection / denominator)


def dice_coef_weighted_(y_pred, y_true):
    """Returns a (approx) IOU score
   IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, C)
        y_true (4-D array): (N, H, W, C)
    Returns:
        float: IOU score
    """
    H, W, num_classes = y_pred.get_shape().as_list()[1:]
    med_bal_factor = [1, 1, 1]
    intersection, denominator = 0, 0
    for i in range(num_classes):
        intersection += med_bal_factor[i] * tf.reduce_sum(y_pred[:, :, :, i] * y_true[:, :, :, i])
        denominator += med_bal_factor[i] * (tf.reduce_sum(y_pred[:, :, :, i] + y_true[:, :, :, i]))
    dice_coef_weighted = ((2. * intersection + smooth) / (denominator + smooth))
    # print(dice_coef_weighted.shape)
    return tf.reduce_mean(dice_coef_weighted)


def dice_coef_axis(y_pred, y_true, i):
    intersection, union = 0, 0
    med_bal_factor = [1, 1, 1, 1]  # For testing
    intersection += med_bal_factor[i] * (tf.reduce_sum(y_true[:, :, :, i] * y_pred[:, :, :, i]))
    union += med_bal_factor[i] * (tf.reduce_sum(y_true[:, :, :, i] + y_pred[:, :, :, i]))
    dice_coef = ((2. * intersection + smooth) / (union + smooth))
    return dice_coef


def dice_coef_0(y_pred, y_true):
    return dice_coef_axis(y_pred, y_true, 0)


def dice_coef_1(y_true, y_pred):
    return dice_coef_axis(y_pred, y_true, 1)


def dice_coef_2(y_true, y_pred):
    return dice_coef_axis(y_pred, y_true, 2)


def compute_cross_entropy_mean(y_pred, y_true):
    # https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
    total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(tf.nn.softmax(y_pred)), [1]))
    return total_loss


def weighted_cross_entropy(y_pred, y_true, num_classes):
    '''
    @@@ From https://github.com/kwotsin/TensorFlow-ENet/blob/master/train_enet.py#L103
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want.
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.
    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------
    INPUTS:
    - y_pred(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - y_true(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''
    # your class weights
    if num_classes == 3:
        class_weights = [1,1,1] #[0.2, 1.2, 2.2]  # [0.014,0.3,3.04]#
    if num_classes == 2:
        class_weights = [1,1] #[0.2, 1.2]  # [0.3,3.04]#
    weights = y_true * class_weights
    sample_weights = tf.reduce_sum(tf.multiply(y_true, class_weights), 3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=y_pred, weights=sample_weights)
    return loss


def weighted_cross_entropy_plus_dice(y_pred, y_true, num_classes=3):
    loss = weighted_cross_entropy(y_pred, y_true, num_classes) + (-dice_coef_(y_pred, y_true))
    return loss


def make_train_op(y_pred, y_true, learning, momentum, nesterov, num_classes,decay_rate):
    """
        Returns a training operation
    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)
        learning: learning_rate
        momentum: scalar value 
        nesterov: if True use Nesterov Momentum.
        num_classes: number of classes
        decay_rate: rate of learning rate decay
    Returns:
        train_op: minimize operation
    """
    decay = False
    cross_entropy_loss = compute_cross_entropy_mean(y_pred, y_true)
    dice_loss = -dice_coef_(y_pred, y_true)
    weighted_cross_entropy_plus = weighted_cross_entropy_plus_dice(y_pred, y_true, num_classes)
    weighted_entropy = weighted_cross_entropy(y_pred, y_true, num_classes)
    global_step = tf.train.get_or_create_global_step()
    if decay:
        print("decay")
        # global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = learning
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   580, decay_rate, staircase=True)
        # Passing global_step to minimize() will increment it at each step.
        learning_step = (tf.train.GradientDescentOptimizer(learning_rate)
                         .minimize(weighted_cross_entropy_plus, global_step=global_step))
        return learning_step
    else:
        # optim = tf.train.AdamOptimizer()
        # optim = tf.train.GradientDescentOptimizer(learning_rate = learning)
        optim = tf.train.MomentumOptimizer(learning, momentum, use_nesterov=nesterov)
        return optim.minimize(weighted_cross_entropy_plus, global_step=global_step)
