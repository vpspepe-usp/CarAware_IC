import numpy as np
import cv2
import time

x = np.load('ObjectRecognizer/array.npy')
while True:
    cv2.imshow("teste", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()