import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

image_dir = "H:/Collagen sheet_dataset/test"
image_lst = os.listdir(image_dir)

image_lst = np.sort(image_lst)

enface_img = np.ndarray((500, 512), dtype='uint8')
cnt = 0
for _name in image_lst:
    img = cv2.imread(os.path.join(image_dir, _name))
    for i in range(img.shape[1]):
        enface_img[cnt, i] = np.average(img[:, i])

    cnt = cnt + 1

enface_img = (enface_img - np.min(enface_img) ) / (np.max(enface_img) - np.min(enface_img))
enface_img = enface_img * 255
# enface_img = np.asarray(enface_img, dtype='uint8')

gmin = np.min(enface_img)
gmax = np.max(enface_img)

alpha = 0
enface_img = np.clip(enface_img - 128 * alpha, 0, 255).astype(np.uint8)


cv2.imshow('asda', enface_img)
cv2.waitKey()
cv2.destroyAllWindows()