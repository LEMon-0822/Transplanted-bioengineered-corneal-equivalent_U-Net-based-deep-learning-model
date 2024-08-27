import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from util import *
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        self.to_tensor = ToTensor()

        lst_data = os.listdir(os.path.join(self.data_dir, 'Label'))

        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('PNG') | f.endswith('png') ]
        lst_order = np.ndarray((len(lst_data)), dtype=str)

        for i in range(len(lst_data)):
            lst_order[i] = lst_data[i].split('.')[0]
        self.file_name = lst_order
        self.order = lst_order.argsort()
        self.lst_data = np.asarray(lst_data)

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):

        input_name = self.lst_data[index].split('.')[0] + '.PNG'
        img_input = cv2.imread(os.path.join(self.data_dir, 'Input', input_name))
        img_label = cv2.imread(os.path.join(self.data_dir, 'Label', self.lst_data[index]))

        sz = img_label.shape
        img_label = img_label[:, :, 0]

        for i in range(sz[0]):
            for j in range(sz[1]):
                if img_label[i, j] == 255:
                    img_label[i, j] = 255
                else:
                    img_label[i, j] = 0


        img_input = cv2.resize(img_input, (512, 512))
        img_label = cv2.resize(img_label, (512, 512))

        if img_input.dtype == np.uint8:
            img_input = img_input/255.0
        if img_label.dtype == np.uint8:
            img_label = img_label/255.0



        if img_input.ndim == 2:
            img_input = img_input[:, :, np.newaxis]
        else:
            img_input = img_input[:, :, 0]
            img_input = img_input[:, :, np.newaxis]

        if img_label.ndim == 2:
            img_label = img_label[:, :, np.newaxis]
        else:
            img_label = img_label[:, :, 0]
            img_label = img_label[:, :, np.newaxis]

        # for i in range(sz[0]):
        #     for j in range(sz[1]):
        #         if img_label[i, j] > 0:
        #             img_label[i, j] = 1.0

        label = img_label

        data = {'input': img_input, 'label': label}

        if self.transform:
            data = self.transform(data)


        data = self.to_tensor(data)
        data['name'] = input_name
        return data


class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

# class RandomCrop(object):
#     def __init__(self, shape):
#         self.shape = shape
#
#     def __call__(self, data):
#         label, input = data['label'], data['input']
#
#         h, w = label.shape[:2]
#         new_h, new_w = self.shape
#
#         top = np.random.randint(0, h-new_h)
#         left = np.random.randint(0, w-new_w)
#
#         id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
#         id_x = np.arange(left, left + new_w, 1)
#
#         data['label'] = label[id_y, id_x]
#         data['input'] = input[id_y, id_x]
#
#         return data


class FullCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        label, input = data['label'], data['input']

        h, w = label.shape[:2]

        id_y = np.arange(0, 512, 1)[:, np.newaxis]
        range_x = 512
        mode = 0
        id_x = np.arange(900 - range_x, 900 + range_x, 1)

        data['label'] = label[id_y, id_x]
        data['input'] = input[id_y, id_x]

        return data

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        label = self.std * label + self.mean

        data['label'] = label
        data['input'] = input

        return data