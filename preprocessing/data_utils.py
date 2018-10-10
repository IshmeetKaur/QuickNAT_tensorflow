import tensorflow as tf
import h5py
import time
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from scipy import ndimage
import random

def read_dataset(filepath, file_for="train"):
    """
    Method to read a .mat format file
    Parameters
    ----------
    filepath: str
              path of  fle to be read
    file_for: str (train or test)
              file for train or test
    Returns
    ----------
    Dataset and Labels(for train)
    Dataset(for test)
    """
    if file_for == "train":
        with h5py.File(filepath, 'r') as file:
            dataset = np.array(file['imdb']['images']['data'][:])
            labels = np.array(file['imdb']['images']['label'])
        return dataset, labels
    if file_for == "validation":
        with h5py.File(filepath, 'r') as file:
            dataset = np.array(file['imdb']['images']['data'])
            labels = np.array(file['imdb']['images']['label'])
        return dataset, labels
    if file_for == "test":
        with h5py.File(filepath, 'r') as file:
            dataset = np.array(file['imdb']['images']['data'])
            labels = np.array(file['imdb']['images']['label'])
        return dataset, labels
    else:
        error_msg = "file_for should be for train or test only"
        return error_msg


def data_augmentation(batch_data, flip_lr=True, rotate=False, noise=True):
    if flip_lr:
        batch_data = _random_flip_leftright(batch_data)
    if rotate:
        batch_data = _random_rotation(batch_data, 10)
    if noise:
        batch_data = _random_blur(batch_data, 2)
    return batch_data


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def _random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            # Random angle
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle,
                                                          reshape=False)
    return batch


def _random_blur(batch, sigma_max):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            # Random sigma
            sigma = random.uniform(0., sigma_max)
            batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch


def split_dataset(labels, dataset="liver_spleen"):
    if dataset == "liver_spleen":
        label_liver_spleen = np.copy(np.array(labels))
        for label in label_liver_spleen:
            # indexed array
            for i in range(4, 12):
                label[0, :, :][label[0, :, :] == i] = 1
        return label_liver_spleen[:, 0, :, :]


    if dataset == "rlung_llung":
        label_rlung_llung = np.copy(np.array(labels))
        for label in label_rlung_llung:
            # indexed array
            for i in range(8, 12):
                label[0, :, :][label[0, :, :] == i] = 1
            for j in range(2, 6):
                label[0, :, :][label[0, :, :] == j] = 1
        return label_rlung_llung[:, 0, :, :]

    if dataset == "rkidney_lkidney":
        label_rkidney_lkidney = np.copy(np.array(labels))
        for label in label_rkidney_lkidney:
            # indexed array
            for i in range(10, 12):
                label[0, :, :][label[0, :, :] == i] = 1
            for j in range(2, 8):
                label[0, :, :][label[0, :, :] == j] = 1
        return label_rkidney_lkidney[:, 0, :, :]


    if dataset == "liver_kidney_lung":
        liver_kidney_lung = np.copy(np.array(labels))
        for label in liver_kidney_lung:
            for i in range(10, 12):
                label[0, :, :][label[0, :, :] == i] = 1
            for j in range(4, 6):
                label[0, :, :][label[0, :, :] == j] = 1
        return liver_kidney_lung[:, 0, :, :]


def one_hot_encode(label, num_classes):
    if (num_classes == 3):
        label[label == label.max()] = 3
        label[label == 6] = 2
        label[label == 8] = 2
        label_ohe = (np.arange(num_classes) == np.array(label)[..., None] - 1).astype(int)
    else:
        label_ohe = (np.arange(num_classes) == np.array(label)[..., None] - 1).astype(int)
    return label_ohe


def remove_back_pixels(train_data, train_labels):
    """
    train_labels: without one hot encoding
    """
    idx = []
    for i in range(0, train_labels.shape[0]):
        if (len(np.unique(train_labels[i])) != 1):
            idx.append(i)
    train_data = np.array([train_data[i] for i in idx])
    train_labels = np.array([train_labels[i] for i in idx])
    return train_data, train_labels


def centeredCrop(img, new_height, new_width):
    batch = []
    width = np.size(img[0], 1)
    height = np.size(img[0], 0)

    left = int(np.ceil((width - new_width) / 2.))
    top = int(np.ceil((height - new_height) / 2.))
    right = int(np.floor((width + new_width) / 2.))
    bottom = int(np.floor((height + new_height) / 2.))
    for i in range(len(img)):
        batch.append(img[i][top:bottom, left:right])
    return np.array(batch)


def normalize_data(train_data):
    normalized = []
    for img in train_data:
        normalized.append((img - img.min()) / (img.max() - img.min()))
    return np.array(normalized)


global_index = 0
list_index = 0
len_entries = 0


def data_generator(batch_size, data, labels, num_classes, data_aug=False, center_crop=True, test_data=False):
    global global_index
    global list_index
    global len_entries
    if list_index == len_entries:
        list_index = 0
        global_index = 0

    X = np.zeros(shape=(batch_size, 192, 192, 1), dtype=np.float32)
    Y = np.zeros(shape=(batch_size, 192, 192), dtype=np.float32)


    while True:
        batch_number = 1
        epoch_number = 1
        entries = 0
        batch_index = 0
        # Total number of samples
        data_num = (data.shape[0] // batch_size) * batch_size
        # create list of batches to shuffle the data
        entries_list = list(range(0, data_num))
        len_entries = len(entries_list)

        if not test_data:
            entries_list = list(range(0, data_num))
            print("entries list", entries_list)
            random.Random(4).shuffle(entries_list)
           
        for n, i in enumerate(entries_list[list_index:]):
            X[batch_index] = centeredCrop(data[i].reshape(1, 256, 256), 192, 192).reshape(192, 192, 1)
            Y[batch_index] = centeredCrop(labels[i].reshape(1, 256, 256), 192, 192).reshape(192, 192)
            batch_index += 1
            
            if batch_index == batch_size:
                batch_number += 1
                global_index += 1

                if list_index <= len(entries_list):
                    list_index = (batch_index * global_index)
                
                label_ohe = one_hot_encode(np.asarray(Y), num_classes)
                yield X, label_ohe
            entries += 1
        epoch_number += 1

