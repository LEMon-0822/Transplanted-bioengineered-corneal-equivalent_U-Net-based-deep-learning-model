import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('DOC3_15.bmp')


dst = np.ones_like(image)
dst = dst *255

dst_ths = np.ones_like(image)
dst_ths = dst_ths *255


mask = cv2.imread('Summation.bmp')
mask_ths = cv2.imread('Summation.bmp')

mask = [mask > 0]
mask_ths = [mask_ths > 10]

mask = np.asarray(mask)
mask_ths = np.asarray(mask_ths)

# mask = mask * 255
mask = mask[0, :, :, 0]
mask_ths = mask_ths[0, :, :, 0]


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if mask[i, j] == False:
            dst[i, j] = 0
        if mask_ths[i, j] == False:
            dst_ths[i, j] = 0

dst = dst[:, :, 0]
dst_ths = dst_ths[:, :, 0]

# print(np.sum(dst[:, :, 0]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
image_ths = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

image[:, :, 3] = dst
image_ths[:, :, 3] = dst_ths

cv2.imwrite('BGD_remove.PNG', image)
cv2.imwrite('BGD_remove_thresh.PNG', image_ths)
