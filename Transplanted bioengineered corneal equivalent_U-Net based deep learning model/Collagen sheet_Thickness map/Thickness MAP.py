import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

def normalFunS(x, mu,sd):
    a=1/(sd*np.sqrt(2*np.pi))
    b=np.exp(-(x-mu)**2/(2*sd**2))
    return(a*b)

image_dir = "./Doctor_post-processing results/Doctor3_2_post-processing"
image_lst = os.listdir(image_dir)


x = 512
y = len(image_lst)
z = 512
cnt = 0
color_map = np.zeros((y, x))
image_lst = np.sort(image_lst)
for _name in image_lst:
    _image = cv2.imread(os.path.join(image_dir, _name))

    for _x in range(x):
        start_pos = 0
        end_pos = 0

        a_line = _image[:, _x]

        color_map[cnt, _x] = np.sum(a_line) / (255)
        # print(np.sum(a_line)/(255*3))
    cnt = cnt + 1
# print(np.min(color_map))
color_map = cv2.flip(color_map, flipCode=1)
# color_map = (color_map - np.min(color_map)) / (np.max(color_map) - np.min(color_map))
# color_map = color_map * 255
# color_map = color_map.astype(dtype='uint8')

# ROI_map = color_map[110:140, 170:210] / 3
# print(np.mean(ROI_map), np.std(ROI_map))
print(color_map.shape)
print(np.count_nonzero(color_map))
print(cnt)

# equ = cv2.equalizeHist(color_map)
# alpha = 0
# enface_img = np.clip(color_map - 128 * alpha, 0, 255).astype(np.uint8)

# cv2.imwrite('Day1.png', color_map)


# cv2.imshow('test', enface_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
