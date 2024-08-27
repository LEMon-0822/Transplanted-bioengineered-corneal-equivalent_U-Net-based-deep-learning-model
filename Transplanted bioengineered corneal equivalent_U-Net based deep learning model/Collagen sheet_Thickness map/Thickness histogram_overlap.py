import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def normalFunS(x, mu,sd):
    a=1/(sd*np.sqrt(2*np.pi))
    b=np.exp(-(x-mu)**2/(2*sd**2))
    return(a*b)



dir_list_label = ['./Doctor_label/Doctor3_2_label', './Doctor_label/Doctor3_13_label',
            './Doctor_label/Doctor3_14_label', './Doctor_label/Doctor3_15_label']


histogram_dataset_pre = []
folder_num = 0

for folder in dir_list_label:
    image_lst = os.listdir(folder)
    print(folder)
    x = 512
    y = len(image_lst)
    z = 512
    cnt = 0
    color_map = np.zeros((y, x))
    image_lst = np.sort(image_lst)

    for _name in image_lst:
        _image = cv2.imread(os.path.join(folder, _name))

        for _x in range(x):
            start_pos = 0
            end_pos = 0

            a_line = (_image[:, _x] > 250)


            color_map[cnt, _x] = np.sum(a_line) / 3


        cnt = cnt + 1
    print(folder_num)
    if folder_num == 0:
        histogram_dataset_pre = {str(folder_num):color_map}
    else:
        histogram_dataset_pre[str(folder_num)] = color_map
    folder_num = folder_num + 1

histogram_dataset_post = []
folder_num = 0


color_map = (color_map - np.min(color_map)) / (np.max(color_map) - np.min(color_map))
color_map = color_map * 255
color_map = color_map.astype(dtype='uint8')
color_map_3ch = np.ndarray((color_map.shape[0], color_map.shape[1], 3))

color_map_3ch[:, :, 0] = color_map
color_map_3ch[:, :, 1] = color_map
color_map_3ch[:, :, 2] = color_map

color_map_3ch = color_map_3ch.astype(dtype='uint8')
color_map = cv2.cvtColor(color_map_3ch, cv2.COLOR_RGB2HSV)

hist_label = cv2.calcHist([color_map], [0], None, [256], [0, 256])

hist_list_0 = np.concatenate(histogram_dataset_pre['0']).astype(int)
hist_list_1 = np.concatenate(histogram_dataset_pre['1']).astype(int)
hist_list_2 = np.concatenate(histogram_dataset_pre['2'])
hist_list_3 = np.concatenate(histogram_dataset_pre['3'])


hist_list_0 = np.delete(hist_list_0, np.where(hist_list_0==0))
hist_list_1 = np.delete(hist_list_1, np.where(hist_list_1==0))
hist_list_2 = np.delete(hist_list_2, np.where(hist_list_2==0))
hist_list_3 = np.delete(hist_list_3, np.where(hist_list_3==0))

unique0, counts0 = np.unique(hist_list_0, return_counts=True)
unique1, counts1 = np.unique(hist_list_1, return_counts=True)
unique2, counts2 = np.unique(hist_list_2, return_counts=True)
unique3, counts3 = np.unique(hist_list_3, return_counts=True)

mu_0 = np.mean(hist_list_0)
std_0 = np.std(hist_list_0)
mu_1 = np.mean(hist_list_1)
std_1 = np.std(hist_list_1)
mu_2 = np.mean(hist_list_2)
std_2 = np.std(hist_list_2)
mu_3 = np.mean(hist_list_3)
std_3 = np.std(hist_list_3)



plt.grid(True, axis='y', c='black', alpha=0.3)
plt.grid(True, axis='x', c='black', alpha=0.3)
plt.plot(unique0, counts0, color='black', label='Day 1', linewidth=2)
plt.plot(unique1, counts1, color='r', label='Day 4', linewidth=2)
plt.plot(unique2, counts2, color='g', label='Day 7', linewidth=2)
plt.plot(unique3, counts3, color='b', label='Day 14', linewidth=2)

plt.show()
