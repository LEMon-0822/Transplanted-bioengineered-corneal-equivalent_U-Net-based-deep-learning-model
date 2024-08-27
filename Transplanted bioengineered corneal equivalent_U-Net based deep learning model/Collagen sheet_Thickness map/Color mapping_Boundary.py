import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

image_dir = "H:/Collagen sheet_results (7 and 14 Day)/result/test/bmp/Output"
image_lst = os.listdir(image_dir)


x = 512
y = len(image_lst)
z = 512

color_map = np.zeros((y, x))
for _name in range(len(image_lst)):
    _image = cv2.imread(os.path.join(image_dir, image_lst[_name]))
    print(_name)
    for _x in range(x):
        start_pos = 0
        end_pos = 0

        a_line = _image[:, _x]

        for i in range(len(a_line)):
            if a_line[i, 0] > 0:
                start_pos = i
                break
        for i in range(len(a_line)):
            if a_line[511 - i, 0] > 0:
                end_pos = 512 - i
                break

        color_map[_name, _x] = end_pos - start_pos


color_map = color_map / np.max(color_map)
color_map = color_map * 255
color_map = color_map.astype(dtype='uint8')

# color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
# cv2.imshow('asda', color_map)
# cv2.waitKey()
cv2.imwrite('Boundary.bmp', color_map)