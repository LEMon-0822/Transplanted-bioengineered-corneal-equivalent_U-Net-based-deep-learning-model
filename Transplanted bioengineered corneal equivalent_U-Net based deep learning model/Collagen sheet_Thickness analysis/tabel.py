A   import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

Doc1_label = [1350947, 992900, 1093893, 913226]
Doc2_label = [1368969, 1031463, 1113161, 947217]
Doc3_label = [1319512, 964103, 1050256, 867333]
Doc_label = [Doc1_label, Doc2_label, Doc3_label]
Doc_label = np.asarray(Doc_label)


Doc1_att = [1269715, 908764, 1015706, 837442]
Doc2_att = [1272472, 929630, 1040879, 863033]
Doc3_att = [1238384, 886267, 990719, 805325]
Doc_att = [Doc1_att, Doc2_att, Doc3_att]
Doc_att = np.asarray(Doc_att)


Doc1_nested = [1262019, 898601, 1004706, 842560]
Doc2_nested = [1260195, 917555, 1034091, 855904]
Doc3_nested = [1238188, 873696, 980018, 798105]
Doc_nested = [Doc1_nested, Doc2_nested, Doc3_nested]
Doc_nested = np.asarray(Doc_nested)


Doc1_Res = [1264690, 902722, 1016161, 835117]
Doc2_Res = [1274697, 911244, 1042903, 864423]
Doc3_Res = [1234665, 855771, 987770, 799110]
Doc_Res = [Doc1_Res, Doc2_Res, Doc3_Res]
Doc_Res = np.asarray(Doc_Res)


Doc1_U = [1251889, 880818, 991295, 824631]
Doc2_U = [1254467, 909346, 1029568, 853394]
Doc3_U = [1227782, 881849, 984934, 802548]
Doc_U = [Doc1_U, Doc2_U, Doc3_U]
Doc_U = np.asarray(Doc_U)
print(Doc1_U, Doc2_U, Doc3_U)

day = 3
x_value = ['Doctor1', 'Doctor2', 'Doctor3']


# 그림 사이즈, 바 굵기 조정
fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.1

# 연도가 4개이므로 0, 1, 2, 3 위치를 기준으로 삼음
index = np.arange(3)

# 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
b1 = plt.bar(index, Doc_label[:, day], bar_width,  label='Label')

b2 = plt.bar(index + bar_width, Doc_att[:, day], bar_width, label='AttU_Net')

b3 = plt.bar(index + 2 * bar_width, Doc_nested[:, day], bar_width, label='NestedU_Net')

b4 = plt.bar(index + 3 * bar_width, Doc_Res[:, day], bar_width, label='ResU_Net')

b5 = plt.bar(index + 4 * bar_width, Doc_U[:, day], bar_width, label='U_Net')

# x축 위치를 정 가운데로 조정하고 x축의 텍스트를 year 정보와 매칭
plt.xticks(np.arange(0.2, 3 + 0.2, 1), x_value, size=20)

# x축, y축 이름 및 범례 설정
plt.title('Day 14', size = 40)
# plt.xlabel('Doctor', size = 13)
plt.ylabel('Number', size = 20)
plt.legend()
plt.show()