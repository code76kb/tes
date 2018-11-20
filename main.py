import t
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
from mnist import MNIST

# Hyper parameter
# Convelution Parrametar
no_kernels = 3
kernel_shape = (3,3,1)
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
bais_1 = 0
bais_2 = 0
bais_3 = 0

# input DATA
mndata = MNIST('/media/patel/DATA/ML_init/DataSets/mnist')
img = cv.imread("/media/patel/DATA/ML_init/tes/img.jpg")

images,labels = mndata.load_training();
print 'images :',len(images)

# Initializing
kernels, paddSize = t.init_kernel(no_kernels,kernel_shape)
# Initializing Weight Matrixes
weight_matrix_1 = np.random.uniform(-1,1,(147,no_hidden_nods_1))
weight_matrix_2 = np.random.uniform(-1,1,(no_hidden_nods_1,no_hidden_nods_2))
weight_matrix_outPut = np.random.uniform(-1,1,(no_hidden_nods_2,no_output_nods))

bais_1 = np.random.uniform(-1,1,(147,1))
bais_2 = np.random.uniform(-1,1,(60,1))
bais_3 = np.random.uniform(-1,1,(40,1))
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
def conv(img):
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

# returns flattern images data
def flattern_data(all_layer_output):
    outputImgs = all_layer_output[len(all_layer_output)-1]
    fc = []
    for i in range(len(outputImgs)):
        # print "flatten :",(outputImgs[i]).flatten()
        fc.extend((outputImgs[i]).flatten())
    return np.array(fc).reshape(1,len(fc))

# Sigmoid Activation
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Calculate Cost
def softmax_cost(out,label):
    eout = np.exp(out,dtype=np.float)
    probs = eout/eout.sum()
    label = label.reshape((1,10))
    p = (label*probs).sum()
    print "Label shape:",label.shape,"probs shape:",probs.shape
    cost = -np.log(p)
    print  "P: ",p
    print "Probs :",probs
    print "cost :",cost
    return cost,props



# Prdict
def predict(img,label):

    # input
    img = (np.array(img,dtype="uint8")).reshape((28,28,1))
    # label vector
    labeles = np.zeros((10,1))
    labeles[label-1,0] = 1

    # conved
    all_layer_output = conv(img)
    # show
    # show(no_conv_layer,no_kernels)
    #fullConnected
    fc = flattern_data(all_layer_output)
    # nueral net
    hidden_layer_1_out = sigmoid(np.dot(fc, weight_matrix_1)) + bais_1
    hidden_layer_2_out = sigmoid(np.dot(hidden_layer_1_out, weight_matrix_2)) + bais_2
    final_output = sigmoid(np.dot(hidden_layer_2_out, weight_matrix_outPut))

    print "final :",final_output
    print "Arg max: ",final_output.argmax()," max: ",final_output.max()
    print "Label: ",label
    softmax_cost(final_output,labeles)

predict(images[0],labels[0])
