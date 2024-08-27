import os
from scipy.stats import poisson
from skimage.transform import rescale, resize
import numpy as np

import torch
import torch.nn as nn

def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# def load(ckpt_dir, net, optim):
#     if not os.path.exists(ckpt_dir):
#         epoch = 0
#         return net, optim, epoch
#
#     ckpt_lst = os.listdir(ckpt_dir)
#     ckpt_lst.sort(key=lambda  f: int(''.join(filter(str.isdigit, f))))
#
#     dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
#
#     net.load_state_dict(dict_model['net'])
#     optim.load_state_dict(dict_model['optim'])
#     epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
#     # print(int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0]))
#     return net, optim, epoch

def load(ckpt_dir, net, optim, index=None):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda  f: int(''.join(filter(str.isdigit, f))))
    if index is None:
        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
        print(ckpt_lst[-1])
    else:

        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[index]))
        # print(ckpt_lst[index])

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    # print(epoch)
    # print(int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0]))
    return net, optim, epoch


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)
