import os
import random
import argparse
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
        transform_val = transforms.Compose([Normalization()])

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform_train, task=task ,opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform_val, task=task ,opts=opts)
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


    fn_loss = nn.BCEWithLogitsLoss().to(device)
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
    if mode == 'train':
        _loss_best = 1
        if train_continue =="on":
            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

        for epoch in range(st_epoch + 1, num_epoch + 1):

            lr_scheduler.step()
            net.train()
            loss_arr = []

            for batch, data in enumerate(loader_train, 1):
                ## Forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                ## Backward pass (back propagation)
                optim.zero_grad()


                loss = fn_loss(output, label)
                loss.backward()

                optim.step()

                ## Calculate loss
                loss_arr += [loss.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))



                # output = torch.sigmoid(output)

                label = fn_tonumpy(label)
                input = fn_tonumpy(input)
                output = fn_tonumpy(output)

                input = fn_denorm(input, mean=0.5, std=0.5)
                output = 1.0 * (output > 0.5)

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)


                id = num_batch_train * (epoch - 1) + batch
                if batch % 10 == 0:
                    cv2.imwrite(os.path.join(result_dir_train, 'bmp', '%04d_label.bmp' % id), 255*label[0])
                    cv2.imwrite(os.path.join(result_dir_train, 'bmp', '%04d_input.bmp' % id), 255*input[0])
                    cv2.imwrite(os.path.join(result_dir_train, 'bmp', '%04d_output.bmp' % id), 255*output[0])

            writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

            # Network validation

            with torch.no_grad():
                net.eval()
                loss_arr = []

                for batch, data in enumerate(loader_val, 1):
                    ## Forward pass
                    label = data['label'].to(device)
                    input = data['input'].to(device)

                    output = net(input)

                    ## Calculate loss
                    loss = fn_loss(output, label)

                    loss_arr += [loss.item()]
                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                          (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                    label = fn_tonumpy(label)
                    input = fn_tonumpy(input)
                    output = fn_tonumpy(output)

                    input = fn_denorm(input, mean=0.5, std=0.5)
                    output = 1.0 * (output > 0.5)

                    input = np.clip(input, a_min=0, a_max=1)
                    output = np.clip(output, a_min=0, a_max=1)

                    ## Save id
                    id = num_batch_val * (epoch - 1) + batch
                    if batch % 10 == 0:
                        cv2.imwrite(os.path.join(result_dir_val, 'bmp', '%04d_label.bmp' % id), 255 * label[0])
                        cv2.imwrite(os.path.join(result_dir_val, 'bmp', '%04d_input.bmp' % id), 255 * input[0])
                        cv2.imwrite(os.path.join(result_dir_val, 'bmp', '%04d_output.bmp' % id), 255 * output[0])

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)


            # if epoch % 4 == 0:
            if _loss_best > np.mean(loss_arr):
                _loss_best = np.mean(loss_arr)
                best_epoch = epoch

            print(best_epoch)
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
        writer_train.close()
        writer_val.close()


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
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

        num_data_test = len(dataset_test)

        num_batch_test = np.ceil(num_data_test / batch_size)


    # net = UNet().to(device)
    # net = ResUnet().to(device)
    net = AttU_Net().to(device)
    # net = R2AttU_Net().to(device)
    # net = NestedUNet().to(device)
    # net = R2U_Net().to(device)

    fn_loss = nn.BCEWithLogitsLoss().to(device)

    optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

    MAX_STEP = int(1e10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, MAX_STEP, eta_min=1e-5)

    ## Additional function setting
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  ## From tensor to numpy
    fn_denorm = lambda x, mean, std: (x * std) + mean


    if mode == 'test':

        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_test, 1):
                ## Forward pass
                input = data['input'].to(device)

                output = net(input)

                # loss = fn_loss(output, label)

                # loss_arr += [loss.item()]

                # output = torch.sigmoid(output)
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
                # name = name.split('.')[0]
                # name = name + '.bmp'
                # print(name)
                cv2.imwrite(os.path.join(result_dir_test, 'bmp', 'input', name), 255 * input[0])
                cv2.imwrite(os.path.join(result_dir_test, 'bmp', 'output', name), 255 * output[0])
