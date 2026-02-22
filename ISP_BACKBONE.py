import numpy as np
import matrix_add
import time
import DPC_module 
import BLC_module

import cv2

x = np.array(cv2.imread(r"C:\Users\jithu\Desktop\images.webp"))



x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
x[10][10] = 0
print(x[:5,:5])


width = x.shape[1]
length = x.shape[0]

x = DPC_module.DPC(x.flatten(),width,length, 10)
x = BLC_module.BLC(x,np.array([5,10,15,20]),width,length) 
                   

x=x.reshape((length, width))

print(x[:5,:5])


