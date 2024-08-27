import matplotlib.pyplot as plt
import cv2


image = cv2.imread('BGD_remove_thresh.PNG')

cv2.imshow('test', image)
cv2.scaleAdd()