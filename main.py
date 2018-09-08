import t
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
from mnist import MNIST

# Hyper parameter
no_kernels = 3
kernel_shape = (3,3,1)
pool_Shape = (2,2)
no_conv_layer = 2
batch_size = 10

# input DATA
mndata = MNIST('/media/patel/DATA/ML_init/DataSets/mnist')
img = cv.imread("/media/patel/DATA/ML_init/tes/img.jpg")

images,labels = mndata.load_training();
print 'images :',len(images)

# Initializing
kernels, paddSize = t.init_kernel(no_kernels,kernel_shape)
# t.init_output(img,paddSize)

# Pooling layer kernel
pool_kernel = np.zeros(pool_Shape)

# output Varaibles
all_layer_output = []
all_convoled = []

# To show oputput of convelution
def show(max_layer,max_kernel):
    for l in range(0,max_layer):
        for k in range(0,max_kernel):
            imgname = "Conved at "+str(l)+" by K:"+str(k)
            cv.imshow(str(imgname),all_layer_output[l][k])
    cv.waitKey(0)
    cv.destroyAllWindows()

# perform convelution and pooling opration
def layer_opration(img,kernel):
    output,relued,time = t.conv(img,kernel,paddSize)
    outputImg = t.pool(relued,pool_kernel)
    return outputImg

# Layer by layer conv+pool run
def run(img):
    print "no of layers: ",no_conv_layer," no kernals: ",no_kernels
    all_layer_output = []
    for i in range(0,no_conv_layer):
        startT = time.time()
        all_convoled = []
        for j in range(0,kernels.shape[0]):
            if(i==0):
                all_convoled.append(layer_opration(img,kernels[j,:,:,:]))
            else:
                img1 = all_layer_output[len(all_layer_output)-1][j]
                all_convoled.append(layer_opration(img1,kernels[j,:,:,:]))
        all_layer_output.append(all_convoled)
        print "Time at layer ",i," :",time.time()-startT
    return all_layer_output


# input
img = (np.array(images[0],dtype="uint8")).reshape((28,28,1))
# conved
all_layer_output = run(img)
# show
# show(no_conv_layer,no_kernels)
# Fully connected
final_size_x = img.shape[0]/(no_conv_layer * pool_Shape[0])
final_size_Y = img.shape[1]/(no_conv_layer * pool_Shape[1])
final_size_z = img.shape[2]
final_shape = (no_kernels,final_size_x,final_size_Y,final_size_z)
print "Final Shape: ",final_shape
fc = (np.array(all_layer_output)).reshape(final_shape)
print "FC reshaped Shape: ",fc.shape
