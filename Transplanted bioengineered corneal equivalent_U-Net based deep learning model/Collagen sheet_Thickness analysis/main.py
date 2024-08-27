import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch
from torchvision import transforms, datasets

def ToTensor(arr):

        label = arr
        label = label.transpose((2, 0, 1)).astype(np.float32)

        label = torch.from_numpy(label)

        return label


Results_Path = 'D:/Collagen sheet_Results_Doctor3/result_ResU-Net/result/test/bmp'

count = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

label = 'label'
output = 'output'
save = 'post-processing'

image_lst = os.listdir(os.path.join(Results_Path, label))

Day1_img = [f for f in image_lst if f.startswith('2')]
Day4_img = [f for f in image_lst if f.startswith('13')]
Day7_img = [f for f in image_lst if f.startswith('14')]
Day14_img = [f for f in image_lst if f.startswith('15')]

x_label = [Day1_img, Day4_img, Day7_img, Day14_img]
x_value = ['Day1_img', 'Day4_img', 'Day7_img', 'Day14_img']
label_value = []
output_value = []
D1_output_value = []
D4_output_value = []
D7_output_value = []
D14_output_value = []
# for i in range(10):
#     if not os.path.exists(os.path.join(Results_Path, count[i], save)):
#         os.makedirs(os.path.join(Results_Path, count[i], save))
#     for _x in x_label:
#         for _img in _x:
#             print(_img)
#
#             Label_image = cv2.imread(os.path.join(Results_Path, label, _img))
#             Output_image = cv2.imread(os.path.join(Results_Path, count[i], output, _img))
#             img2 = np.zeros_like(Output_image)
#             Output_image = Output_image[:, :, 0]
#             cnt, labels = cv2.connectedComponents(Output_image)
#
#             if cnt > 1:
#                 idx = np.unique(labels, return_counts=True)[0][np.argsort(np.unique(labels, return_counts=True)[1])][-2]
#             else:
#                 idx = 2
#
#             img2[labels == idx] = [255, 255, 255]
#
#             cv2.imwrite(os.path.join(Results_Path,count[i], save, _img), img2)


# Count intensity
for i in range(10):
    for _x in x_label:
        Label_cnt = 0
        Output_cnt = 0
        for _img in _x:
            Label_image = cv2.imread(os.path.join(Results_Path, label, _img))
            Output_image = cv2.imread(os.path.join(Results_Path, count[i], save, _img))


            Label_image = Label_image[:, :, 0]
            Output_image = Output_image[:, :, 0]
            zeros_image = np.zeros_like(Label_image)


            Label_image = Label_image > 0
            Output_image = Output_image > 0
            zeros_image = zeros_image == 0

            L_unique, L_counts = np.unique(Label_image, return_counts=True)
            O_unique, O_counts = np.unique(Output_image, return_counts=True)

            same_region = 0
            new_image = Label_image[Output_image]

            same_region = np.sum(new_image)

            # for i in range(512):
            #     for j in range(512):
            #         if Label_image[i, j] == True and Output_image[i, j] == True:
            #             same_region = same_region + 1

            if len(L_counts) > 1:
                Label_cnt = Label_cnt + L_counts[1]
            else:
                Label_cnt = Label_cnt


            Output_cnt = same_region + Output_cnt


            # if len(O_counts) > 1:
            #     Output_cnt = Output_cnt + O_counts[1]
            # else:
            #     Output_cnt = Output_cnt


        label_value.append(Label_cnt)
        if _x == x_label[0]:
            D1_output_value.append(Output_cnt)
        elif _x == x_label[1]:
            D4_output_value.append(Output_cnt)
        elif _x == x_label[2]:
            D7_output_value.append(Output_cnt)
        else:
            D14_output_value.append(Output_cnt)

        print(label_value)
        print(D1_output_value)
        print(D4_output_value)
        print(D7_output_value)
        print(D14_output_value)

D1_output_value = np.asarray(D1_output_value)
D4_output_value = np.asarray(D4_output_value)
D7_output_value = np.asarray(D7_output_value)
D14_output_value = np.asarray(D14_output_value)

print(np.mean(D1_output_value)/ label_value[0]*100, np.std(D1_output_value)/ label_value[0]*100, np.std(D1_output_value)/ (label_value[0]*np.sqrt(10)))
print(np.mean(D4_output_value) / label_value[1]*100, np.std(D4_output_value)/ label_value[1]*100, np.std(D4_output_value)/ (label_value[1] *np.sqrt(10)))
print(np.mean(D7_output_value) / label_value[2]*100, np.std(D7_output_value)/ label_value[2]*100, np.std(D7_output_value)/ (label_value[2]*np.sqrt(10)))
print(np.mean(D14_output_value) / label_value[3]*100, np.std(D14_output_value)/ label_value[3]*100, np.std(D14_output_value)/ (label_value[3]*np.sqrt(10)))

print(np.max(D1_output_value)/ label_value[0] *100, np.min(D1_output_value)/ label_value[0] *100  )
print(np.max(D4_output_value)/ label_value[1] *100, np.min(D4_output_value)/ label_value[1] *100  )
print(np.max(D7_output_value)/ label_value[2] *100, np.min(D7_output_value)/ label_value[2] *100  )
print(np.max(D14_output_value)/ label_value[3] *100, np.min(D14_output_value)/ label_value[3] *100  )


# fn_loss = nn.BCEWithLogitsLoss()
# loss_arr = []
#
# for _x in x_label:
#     Label_cnt = 0
#     Output_cnt = 0
#     for _img in _x:
#         Label_image = cv2.imread(os.path.join(Results_Path, label, _img))
#         Output_image = cv2.imread(os.path.join(Results_Path, save, _img))
#
#         # tensor_label = ToTensor(np.asarray(Label_image))
#         # tensor_output = ToTensor(np.asarray(Output_image))
#         # loss = fn_loss(tensor_output, tensor_label)
#         # print(loss)
#
#
#         Label_image = Label_image[:, :, :]
#         Output_image = Output_image[:, :, :]
#         zeros_image = np.zeros_like(Label_image)
#
#
#         Label_image = Label_image > 0
#         Output_image = Output_image > 0
#         zeros_image = zeros_image == 0
#
#         L_unique, L_counts = np.unique(Label_image, return_counts=True)
#         O_unique, O_counts = np.unique(Output_image, return_counts=True)
#
#         same_region = 0
#         new_image = Label_image[Output_image]
#
#         same_region = np.sum(new_image)
#
#         # for i in range(512):
#         #     for j in range(512):
#         #         if Label_image[i, j] == True and Output_image[i, j] == True:
#         #             same_region = same_region + 1
#
#         if len(L_counts) > 1:
#             Label_cnt = Label_cnt + L_counts[1]
#         else:
#             Label_cnt = Label_cnt
#
#
#         Output_cnt = same_region + Output_cnt
#
#
#         # if len(O_counts) > 1:
#         #     Output_cnt = Output_cnt + O_counts[1]
#         # else:
#         #     Output_cnt = Output_cnt
#
#
#     label_value.append(Label_cnt)
#     output_value.append(Output_cnt)
#
#
# print(label_value, output_value)
#
#
#
# for _x in x_label:
#     for _img in _x:
#         print(_img)
#         Label_image = cv2.imread(os.path.join(Results_Path, label, _img))
#         Output_image = cv2.imread(os.path.join(Results_Path, output, _img))
#         img2 = np.zeros_like(Output_image)
#         Output_image = Output_image[:, :, 0]
#         cnt, labels = cv2.connectedComponents(Output_image)
#
#         if cnt > 1:
#             idx = np.unique(labels, return_counts=True)[0][np.argsort(np.unique(labels, return_counts=True)[1])][-2]
#         else:
#             idx = 2
#
#         img2[labels == idx] = [255, 255, 255]
#
#         cv2.imwrite(os.path.join(Results_Path, save, _img), img2)



