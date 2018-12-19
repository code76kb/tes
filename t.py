import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time

# img = cv.imread("/media/patel/DATA/ML_init/tes/img.jpg")

startT = 0
stopT  = 0
# global output
# global paddedImg

# kernel = np.random.uniform(0,0.6,(3,3,3))
# pool_kernel = np.zeros((2,2))


def init_kernel(no_kernels,shape):
    variance = 0.6/shape[0]
    shape1 = (no_kernels,)+shape
    kernel = np.random.uniform(-variance,variance,shape1)
    paddSize = ((shape[0]-1)/2) + 1
    print "Kernel Shape : ",kernel.shape,", paddSize :",paddSize
    return kernel,paddSize

# def init_output(img,paddSize):
#     global output,paddedImg
#     output = np.zeros_like(img)
#     paddedImg = np.zeros((img.shape[0]+paddSize,img.shape[1]+paddSize,img.shape[2]),dtype='uint8')
#     paddedImg [paddSize-1:img.shape[0]+paddSize-1 , paddSize-1:img.shape[1]+paddSize-1] = img
#     print "PaddedImg Shape:",paddedImg.shape


def conv(img,kernel,paddSize):
    startT = time.time()

    output = np.zeros_like(img,dtype='float128')
    paddedImg = np.zeros((img.shape[0]+paddSize,img.shape[1]+paddSize,img.shape[2]),dtype='float128')
    paddedImg [paddSize-1:img.shape[0]+paddSize-1 , paddSize-1:img.shape[1]+paddSize-1] = img

    # print 'Conve : paddedImg shape :',paddedImg.shape
    # print 'Conve : kernel shape :',kernel.shape
    # print 'Conve : img shape :',img.shape

    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            # print "Conve :at x,y pos on img :",x,',',y
            output[x,y,0]= (kernel[:,:,0] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 0]).sum() # Red Channel
            if(img.shape[2] > 1):
                output[x,y,1]= (kernel[:,:,1] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 1]).sum() # Green Channel
            if(img.shape[2] > 2):
                output[x,y,2]= (kernel[:,:,2] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 2]).sum() # Blue Channel
    relued = output
    relued[relued<=0]=0 #Relu Activation
    stopT = time.time()
    sec = stopT - startT
    return output,relued,sec

# Calculate Conve output shape

def outPutShape_Cal(imgShape,kernelShape,paddSize):
    return ((imgShape - kernelShape) * 2 (paddSize - 1) ) + 1


# for error calculation
def conv2(img,kernel,paddSize,outputShape):
    startT = time.time()

    # Calculate output shape after conv
    imgShape = np.array(img.shape)
    kernelShape = np.array(kernel.shape)
    # outputShape = outPutShape_Cal(imgShape,kernelShape,paddSize)

    output = np.zeros(outputShape,dtype='float128')

    paddedImg = np.zeros((img.shape[0]+paddSize,img.shape[1]+paddSize,img.shape[2]),dtype='float128')
    if(paddSize != 0 ):
        paddedImg [paddSize-1:img.shape[0]+paddSize-1 , paddSize-1:img.shape[1]+paddSize-1] = img
    else:
        paddedImg = img

    # print "Conve2 : PaddedImg :",paddedImg
    # print "Conve2 : kernel :",kernel


    # print 'Conve2 : paddedImg shape :',paddedImg.shape
    # print 'Conve2 : kernel shape :',kernel.shape
    # print 'Conve2 : img shape :',img.shape
    # print 'Conve2 : output shape :',output.shape

    for x in range(0,output.shape[0]):
        for y in range(0,output.shape[1]):
            # print "Conve :at x,y pos on img :",x,',',y
            output[x,y,0]= (kernel[:,:,0] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 0]).sum() # Red Channel
            if(img.shape[2] > 1):
                output[x,y,1]= (kernel[:,:,1] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 1]).sum() # Green Channel
            if(img.shape[2] > 2):
                output[x,y,2]= (kernel[:,:,2] * paddedImg[x:x+kernel.shape[0], y:y+kernel.shape[1], 2]).sum() # Blue Channel
    # relued = output
    # relued[relued<=0]=0 #Relu Activation
    stopT = time.time()
    sec = stopT - startT
    return output,sec

def pool(img,kernel):
    w = int(img.shape[0]/kernel.shape[0])
    h = int(img.shape[1]/kernel.shape[1])
    d = int(img.shape[2])
    outputImg = np.zeros((w,h,d),dtype='float128')
    # print 'Pooled output shaped :',outputImg.shape
    for x in range(w):
        for y in range(h):
            Rx0 = x*kernel.shape[0]
            Rx1 = Rx0+kernel.shape[0]
            Ry0 = y*kernel.shape[1]
            Ry1 = Ry0+kernel.shape[1]
            outputImg[x,y,0] =  img[Rx0:Rx1, Ry0:Ry1,0].max() # Red-Channel
            if(img.shape[2]>1):
                outputImg[x,y,1] =  img[Rx0:Rx1, Ry0:Ry1,1].max() # Green-Channel
            if(img.shape[2]>2):
                outputImg[x,y,2] =  img[Rx0:Rx1, Ry0:Ry1,2].max() # Blue-Channel
    return outputImg




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
