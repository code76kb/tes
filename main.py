import t
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
# from mnist import MNIST

# Hyper parameter
# Convelution Parrametar
no_kernels = 3
kernel_shape = (3,3,3)
pool_Shape = (2,2)
no_conv_layer = 2
# batch size
batch_size = 10
# Fully connected Parameter
no_hidden_layers = 2
no_output_nods = 10
no_hidden_nods_1 = 60
no_hidden_nods_2 = 40
weight_matrix_1 = 0
weight_matrix_2 = 0
weight_matrix_outPut = 0
bais_1 = 1
bais_2 = 1
bais_3 = 1

# input DATA

img = cv.imread("/home/satyaprakash/tes/img.jpg")

# images,labels = mndata.load_training();
# print 'images :',len(images)

# Initializing
kernels, paddSize = t.init_kernel(no_kernels,kernel_shape)
# Initializing Weight Matrixes
weight_matrix_1 = np.random.uniform(-1,1,(147,no_hidden_nods_1))
weight_matrix_2 = np.random.uniform(-1,1,(no_hidden_nods_1,no_hidden_nods_2))
weight_matrix_outPut = np.random.uniform(-1,1,(no_hidden_nods_2,no_output_nods))
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
    print "Image Shape: ",img.shape
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

# returns flattern images data
def flattern_data(all_layer_output):
    outputImgs = all_layer_output[len(all_layer_output)-1]
    fc = []
    for i in range(len(outputImgs)):
        # print "flatten :",(outputImgs[i]).flatten()
        fc.extend((outputImgs[i]).flatten())
    return np.array(fc).reshape(1,len(fc))

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# input
# img = (np.array(images[0],dtype="uint8")).reshape((28,28,1))
# conved
all_layer_output = run(img)
# show
show(no_conv_layer,no_kernels)

#fullConnected
# fc = flattern_data(all_layer_output)
# nueral net
# hidden_layer_1_out = sigmoid(np.dot(fc, weight_matrix_1)) + bais_1
# hidden_layer_2_out = sigmoid(np.dot(hidden_layer_1_out, weight_matrix_2)) + bais_2
# final_output = sigmoid(np.dot(hidden_layer_2_out, weight_matrix_outPut))

print "final :",final_output
print "Arg max: ",final_output.argmax()," max: ",final_output.max()
