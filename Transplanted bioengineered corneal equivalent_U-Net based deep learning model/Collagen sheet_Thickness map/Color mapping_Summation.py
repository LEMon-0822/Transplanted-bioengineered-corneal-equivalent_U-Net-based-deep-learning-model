import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

def normalFunS(x, mu,sd):
    a=1/(sd*np.sqrt(2*np.pi))
    b=np.exp(-(x-mu)**2/(2*sd**2))
    return(a*b)

image_dir = "./Output"
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

        color_map[cnt, _x] = np.sum(a_line) / (255*3)
        # print(np.sum(a_line)/(255*3))
    cnt = cnt + 1


# color_map = (color_map - np.min(color_map)) / (np.max(color_map) - np.min(color_map))
# color_map = color_map * 255
# color_map = color_map.astype(dtype='uint8')

print(np.unique(color_map))
hist_list = np.concatenate(color_map)


hist_list = np.delete(hist_list, np.where(hist_list==0))

print(len(hist_list))


x = np.unique(hist_list).tolist()


mu = np.mean(hist_list)
std = np.std(hist_list)


plt.grid(True, axis='y', c='black', alpha=0.3)
plt.grid(True, axis='x', c='black', alpha=0.3)


n, bins, ig = plt.hist(hist_list, bins=len(np.unique(hist_list)), range=[1, len(np.unique(color_map))],
                       color=[0.3, 0.3, 0.3], width=0.6, edgecolor='black', linewidth=2)
plt.xlim([0, 50])
plt.ylim([0, 8000])
print(mu, std)
print(np.max(normalFunS(bins, mu, std)))
# plt.plot(bins, normalFunS(bins, mu, std), linewidth=2, color="r")

# plt.show()
plt.savefig('C:/PycharmProjects/Collagen sheet_Thickness map/Doctor3_15/Figure_3.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

# equ = cv2.equalizeHist(color_map)
# res = np.hstack((color_map, equ)) #stacking images side-by-side
#
# color_map[:, :] = 0
# color_map[450:, :] = 0
#
# plt.subplot(1, 2, 1)
# plt.hist(color_map.ravel(), 256, [0,256])
# plt.subplot(1, 2, 2)
# plt.hist(equ.ravel(), 256, [0,256])
# plt.show()
# color_map = cv2.applyColorMap(res, cv2.COLORMAP_JET)
#
# cv2.imshow('asda', color_map)
# cv2.waitKey()
# cv2.imwrite('Summation.bmp', color_map)