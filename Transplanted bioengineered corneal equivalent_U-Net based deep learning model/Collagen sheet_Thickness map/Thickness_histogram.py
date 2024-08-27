import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

folder_path = './Doctor3_15'
image_name = 'Summation.bmp'

image = cv2.imread(os.path.join(folder_path, image_name))

hist_list = []

hist_list.append(image[:, :, 0])

hist_list = np.asarray(hist_list)
hist_list = np.concatenate(hist_list)
hist_list = np.concatenate(hist_list)
# hist_list =
print(len(np.unique(hist_list)))

plt.hist(hist_list, bins=32, range=[1,255], width=8)
plt.show()