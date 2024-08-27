import os
import random
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchvision import transforms, datasets
from torchsummary import summary
from model import *
from dataset import *
from util import *

import torchvision.transforms as transforms

class RandomGaussianNoise(object):
    def __init__(self, mean=0, std=0.3, shape=(512, 512, 1)):
        self.mean = mean
        self.std = std
        self.shape = shape

    def __call__(self, data):
        # label = np.array(data)

        label, input = data['label'], data['input']
        noise = np.random.rand(512, 512) * self.std + self.mean
        # print(noise.shape)

        # if np.random.rand() < 0.2:
        #     label = (label + noise) / torch.max (label + noise)

        input[:, :, 0] = input[:, :, 0] + noise


        data['input'] = input


        return data

class Rotation(object):
    def __init__(self, degree=15, scale=1, shape=(512, 512, 1)):
        self.degree = degree
        self.shape = shape
        self.scale = scale

    def __call__(self, data):
        # label = np.array(data)
        cX, cY = self.shape[0]//2, self.shape[1]//2
        label, input = data['label'], data['input']
        set_degree = np.random.randint(-self.degree, self.degree)
        M = cv2.getRotationMatrix2D((cX, cY), set_degree, self.scale)
        rotated_image = cv2.warpAffine(input, M, (self.shape[0], self.shape[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        rotated_label = cv2.warpAffine(label, M, (self.shape[0], self.shape[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # print(noise.shape)

        # if np.random.rand() < 0.2:
        #     label = (label + noise) / torch.max (label + noise)
        rotated_image = rotated_image[:, :, np.newaxis]
        rotated_label = rotated_label[:, :, np.newaxis]

        data['input'] = rotated_image
        data['label'] = rotated_label


        return data

class Distortion(object):
    def __init__(self, exp=20, scale=1, shape=(512, 512, 1)):
        self.exp = exp
        self.scale = scale
        self.rows = shape[0]
        self.cols = shape[1]

    def __call__(self, data):
        # label = np.array(data)
        mapy, mapx = np.indices((self.rows, self.cols), dtype=np.float32)

        mapx = 2 * mapx / (self.cols - 1) - 1
        mapy = 2 * mapy / (self.rows - 1) - 1

        label, input = data['label'], data['input']

        r, theta = cv2.cartToPolar(mapx, mapy)
        set_exp = np.random.randint(6, 14) / 10

        r[r < self.scale] = r[r < self.scale] ** set_exp
        mapx, mapy = cv2.polarToCart(r, theta)

        mapx = ((mapx + 1) * self.cols - 1) / 2
        mapy = ((mapy + 1) * self.rows - 1) / 2

        distorted_input = cv2.remap(input, mapx, mapy, cv2.INTER_LINEAR)
        distorted_label = cv2.remap(label, mapx, mapy, cv2.INTER_LINEAR)

        distorted_input = distorted_input[:, :, np.newaxis]
        distorted_label = distorted_label[:, :, np.newaxis]

        data['input'] = distorted_input
        data['label'] = distorted_label

        return data
def train(args):
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker
    mode = args.mode
    network = args.network
    out_channels = args.out_channels

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    mode = args.mode

    train_continue = args.train_continue

    learning_type = args.learning_type

    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')
    result_dir_test = os.path.join(result_dir, 'test')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, 'train'))

    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir_train, 'bmp'))
        os.makedirs(os.path.join(result_dir_val, 'bmp'))


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("CPU-GPU mode: %s" %device)
    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    ## Train Dataset
    if mode == 'train':
        # transform_train = transforms.Compose([FullCrop(shape=(ny, nx)), RandomFlip(), Normalization()])
        # transform_val = transforms.Compose([FullCrop(shape=(ny, nx)), RandomFlip(), Normalization()])
        transform_train = transforms.Compose([Normalization()])
        transform_val = transforms.Compose([ Normalization(), RandomGaussianNoise()])

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform_train, task=task ,opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_val, task=task ,opts=opts)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)

        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)

        num_batch_train = np.ceil(num_data_train / batch_size)
        num_batch_val = np.ceil(num_data_val / batch_size)

    # net = UNet().to(device)
    # net = ResUnet().to(device)
    net = AttU_Net().to(device)
    # net = R2AttU_Net().to(device)
    # net = NestedUNet().to(device)
    # net = R2U_Net().to(device)


    fn_loss = nn.BCEWithLogitsLoss().to(device)     ## unet
    # fn_loss = nn.BCELoss().to(device)



    optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.1)


    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    fn_class = lambda x: 1.0 * (x > 0.5)

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## Train network
    st_epoch = 0
    best_epoch = 0
    ## Train mode
    image_name = 'Confusion matrix_Noise result_review.txt'
    txt_dir = os.path.join("D:/Collagen sheet_Results_Doctor3/result_AttU-Net", image_name)

    if mode == 'train':
        _loss_best = 0
    f = open(txt_dir, 'w')
    for ckpt_idx in range(len(os.listdir(ckpt_dir))):

        if train_continue =="on":
            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim, index=ckpt_idx)

            TP = 0
            TN = 0
            FP = 0
            FN = 0

            if not os.path.exists(os.path.join(result_dir_test, 'bmp', str(ckpt_idx), 'output')):
                os.makedirs(os.path.join(result_dir_test, 'bmp', str(ckpt_idx), 'output'))
            with torch.no_grad():
                net.eval()
                loss_arr = []

                for batch, data in enumerate(loader_val, 1):
                    ## Forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)

                    output = net(input)



                    label = fn_tonumpy(label)
                    input = fn_tonumpy(input)
                    output = fn_tonumpy(output)

                    # cv2.imshow('test', input[0, :, :, 0])
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

                    input = fn_denorm(input, mean=0.5, std=0.5)


                    output = 1.0 * (output > 0.5)


                    input = np.clip(input, a_min=0, a_max=1)
                    output = np.clip(output, a_min=0, a_max=1)



                    tmp_same = (label == output)

                    TP += np.sum(label[tmp_same])


                    unique1, arr1 = np.unique((label == output), return_counts=True)

                    unique3, arr3 = np.unique((label < output), return_counts=True)
                    unique4, arr4 = np.unique((label > output), return_counts=True)


                    # TN += (arr1[1] - TP)
                    #
                    # if len(arr3) == 2:
                    #     FP += arr3[1]
                    # else:
                    #     FP += 0
                    #
                    # if len(arr4) == 2:
                    #     FN += arr4[1]
                    # else:
                    #     FN += 0






                    ## Save id
                    name = data['name']

                    if batch % 1 == 0:
                        cv2.imwrite(os.path.join(result_dir_test, 'bmp', 'Noise', 'label', name[0]), 255 * label[0])
                        cv2.imwrite(os.path.join(result_dir_test, 'bmp', 'Noise', 'input', name[0]), 255 * input[0])
                        cv2.imwrite(os.path.join(result_dir_test, 'bmp', 'Noise', 'output', name[0]), 255 * output[0])


            # ACC = (TP + TN) / (TP + TN + FP + FN)
            # SEN = (TP) / (TP + FN)
            # SPE = (TN) / (TN + FP)
            # DIC = (2 * TP) / (2 * TP + FN + FP)
            # JAI = (TP) / (TP + FN + FP)  ## Jaccard index


            # if _loss_best < JAI:
            #     _loss_best = JAI
            #     best_epoch = ckpt_idx
            # print("|ACC :%.4f  |SEN :%.4f  |SPE:%.4f  |DIC:%.4f  |JAI:%.4f  |Best epoch:%d|" % (ACC, SEN, SPE, DIC, JAI, best_epoch))
            # f.write("%.4f %.4f %.4f %.4f %.4f %d\n" % (ACC, SEN, SPE, DIC, JAI, best_epoch))
    f.close()






def test(args):
    print('test mode')
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker
    mode = args.mode
    network = args.network
    out_channels = args.out_channels

    ## Google colab directory
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    ## 실행 플래그
    mode = args.mode

    ## 연속 학습 플래그
    train_continue = args.train_continue

    ## Residual 플래그
    learning_type = args.learning_type

    ## 디렉토리 생성하기(결과를 텐서보드 -> png)
    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')
    result_dir_test = os.path.join(result_dir, 'test')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, 'train'))

    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir_train, 'bmp'))
        os.makedirs(os.path.join(result_dir_val, 'bmp'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("CPU-GPU mode: %s" % device)
    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    if mode == 'test':

        transform_test = transforms.Compose([Normalization()])
        # transform_test = transforms.Compose([Normalization()])

        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task ,opts=opts)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)         ## i5-7500 core : 4

        num_data_test = len(dataset_test)

        num_batch_test = np.ceil(num_data_test / batch_size)


    net = ResUnet().to(device)

    fn_loss = nn.BCEWithLogitsLoss().to(device)

    optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

    MAX_STEP = int(1e10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, MAX_STEP, eta_min=1e-5)

    ## Additional function setting
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  ## From tensor to numpy
    fn_denorm = lambda x, mean, std: (x * std) + mean

    ## Train mode
    if mode == 'test':
        ## Test는 무조건 로드 해야함 네트워크들
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        with torch.no_grad():
            net.eval()  ## 학습 : net.train(), validation, test : net.eval() --> validation도 어떤의미에선 test와 동일하기에
            loss_arr = []

            for batch, data in enumerate(loader_test, 1):
                ## Forward pass
                input = data['input'].to(device)
                print(input.shape)
                output = net(input)

                # loss = fn_loss(output, label)

                # loss_arr += [loss.item()]  ## Loss 배열에 loss 값 대입

                output = torch.sigmoid(output)
                print("TEST: BATCH %04d / %04d" %
                      (batch, num_batch_test))

                input = fn_tonumpy(input)
                output = fn_tonumpy(output)

                input = fn_denorm(input, mean=0.5, std=0.5)
                output = 1.0 * (output > 0.5)

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                input_dir = 'Input'
                output_dir = 'Output'
                name = data['name'][0]
                cv2.imwrite(os.path.join(result_dir_test, 'bmp', 'Input', name), 255 * input[0])
                cv2.imwrite(os.path.join(result_dir_test, 'bmp', 'Output', name), 255 * output[0])
