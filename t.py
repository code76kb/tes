import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
import math

img = cv.imread("/home/satyaprakash/junk/img.jpg")

kernel = np.random.uniform(0,0.6,(3,3,3))
pool_kernel = np.zeros((2,2))
paddSize = ((kernel.shape[0]-1)/2) + 1
output = np.zeros_like(img)
paddedImg = np.zeros((img.shape[0]+paddSize,img.shape[1]+paddSize,img.shape[2]),dtype='uint8')
paddedImg [paddSize-1:img.shape[0]+paddSize-1 , paddSize-1:img.shape[1]+paddSize-1] = img
print "Padding :",paddSize
print "Kernel shape: ",kernel.shape
print "Pool Kernel Shape: ",pool_kernel.shape
print "img shape: ",img.shape
print "paddedImg shape:",paddedImg.shape
print "Output Shape: ",output.shape

startT = 0
stopT  = 0

def conv(img,kernel):
    startT = time.time()
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            output[x,y,0]= (kernel[:,:,0] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 0]).sum() # Red Channel
            output[x,y,1]= (kernel[:,:,1] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 1]).sum() # Green Channel
            output[x,y,2]= (kernel[:,:,2] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 2]).sum() # Blue Channel
    relued = output
    relued[relued<=0]=0 #Relu Activation
    stopT = time.time()
    return output,relued,startT,stopT

def pool(img,kernel):
    w = int(img.shape[0]/kernel.shape[0])
    h = int(img.shape[1]/kernel.shape[1])
    print w,h
    outputImg = np.zeros((w,h,3),dtype='uint8')
    for x in range(w):
        for y in range(h):
            Rx0 = x*kernel.shape[0]
            Rx1 = Rx0+kernel.shape[0]
            Ry0 = y*kernel.shape[1]
            Ry1 = Ry0+kernel.shape[1]
            outputImg[x,y,0] =  img[Rx0:Rx1, Ry0:Ry1,0].max() # Red-Channel
            outputImg[x,y,1] =  img[Rx0:Rx1, Ry0:Ry1,1].max() # Green-Channel
            outputImg[x,y,2] =  img[Rx0:Rx1, Ry0:Ry1,2].max() # Blue-Channel
    return outputImg

temp,relued,startT,stopT=conv(img,kernel)
sec = stopT-startT
print "Took :",sec," sec!!"
print "temp :",temp.shape
relu-R = relued[:,:,0]
relu-G = relued[:,:,1]
relu-B = relued[:,:,2]

norm-R = temp[:,:,0]
norm-G = temp[:,:,1]
norm-B = temp[:,:,2]
plt.plot(relu-R,relu-R)
plt.plot(relu-G,relu-G)
plt.plot(relu-B,relu-B)
plt.show()



# pool_temp = pool(temp,pool_kernel)
# print "Pool shape: ",pool_temp.shape

# cv.imshow('original',img)
# cv.imshow('conved',temp)
#
# cv.imshow('R-Channel',temp[:,:,0])
# cv.imshow('G-Channel',temp[:,:,1])
# cv.imshow('B-Channel',temp[:,:,2])
#
# cv.waitKey(0)
# cv.destroyAllWindows()
