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
mndata = MNIST('/mnt/66C2AAD8C2AAABAD/ML_init/DataSets/mnist')
img = cv.imread("/mnt/66C2AAD8C2AAABAD/ML_init/tes/img.jpg")

images,labels = mndata.load_training();
print 'images :',len(images)

# Initializing
kernels_0, paddSize = t.init_kernel(no_kernels,kernel_shape)
kernels_1, paddSize_1 = t.init_kernel(no_kernels,kernel_shape)
# Initializing Weight Matrixes
weight_matrix_1 = np.random.uniform(-1,1,(no_hidden_nods_1,147))
weight_matrix_2 = np.random.uniform(-1,1,(no_hidden_nods_2,no_hidden_nods_1))
weight_matrix_outPut = np.random.uniform(-1,1,(no_output_nods,no_hidden_nods_2,))

bais_1 = np.random.uniform(-1,1,(147,1))
bais_2 = np.random.uniform(-1,1,(60,1))
bais_3 = np.random.uniform(-1,1,(40,1))
# Pooling layer kernel
pool_kernel = np.zeros(pool_Shape)

# output Varaibles
all_layer_output_Convoled = []
all_convolved = []
all_layer_output = []
all_pooled = []

# To show oputput of convelution
def show(max_layer,max_kernel,all_layer_output):
    for l in range(0,max_layer):
        for k in range(0,max_kernel):
            imgname = "Conved at "+str(l)+" by K:"+str(k)
            cv.imshow(str(imgname),all_layer_output[l][k])
    cv.waitKey(0)
    cv.destroyAllWindows()

# perform convelution and pooling opration
def layer_opration(img,kernel):
    print ' layer opr img ::',img.shape
    print ' layer opr kernel ::',kernel.shape
    output,relued,time = t.conv(img,kernel,paddSize)
    outputImg = t.pool(relued,pool_kernel)
    return outputImg,relued

# Layer by layer conv+pool run
def conv(img):
    print "no of layers: ",no_conv_layer," no kernals: ",no_kernels
    all_layer_output = []
    all_layer_output_Convoled = []
    for i in range(0,no_conv_layer):
        startT = time.time()
        all_pooled = []
        all_convolved = []

        if (i == 0):
            kernels = kernels_0
        if(i == 1):
            kernels = kernels_1

        for j in range(0,kernels.shape[0]):
            if(i==0):
                outputImg, relued = layer_opration(img,kernels[j,:,:,:])
                all_pooled.append(outputImg)
                all_convolved.append(relued)
            else:
                img1 = all_layer_output[len(all_layer_output)-1][j]
                outputImg, relued= layer_opration(img1,kernels[j,:,:,:])
                all_pooled.append(outputImg)
                all_convolved.append(relued)

        all_layer_output.append(all_pooled)
        all_layer_output_Convoled.append(all_convolved)

        print "Time at layer ",i," :",time.time()-startT
    return all_layer_output, all_layer_output_Convoled

# returns flattern images data
def flattern_data(all_layer_output):
    outputImgs = all_layer_output[len(all_layer_output)-1]
    fc = []
    for i in range(len(outputImgs)):
        # print "flatten :",(outputImgs[i]).flatten()
        fc.extend((outputImgs[i]).flatten())
    return np.array(fc).reshape(len(fc),1)

# Sigmoid Activation
def sigmoid(x):
    return  1 / (1 + np.exp(-x))

# sigmoid prime
def sigmoidPrime(x):
    return (np.exp(-x)/ ((1+np.exp(-x))**2) )


# Calculate Cost
def softmax_cost(out,label):
    eout = np.exp(out,dtype=np.float)
    probs = eout/eout.sum()
    label = label.reshape((10,1))
    p = (label*probs).sum()
    print "Label shape:",label.shape,"probs shape:",probs.shape
    cost = -np.log(p)
    entropy = -1 * label * (np.log(p))
    print "Entropy :",entropy
    print  "P: ",p
    print "Probs :",probs
    print "cost :",cost
    return cost,probs

# pooled delta error mapping
def pool_error_map(conved, pooled_delta):
    # print 'Error pool :',pooled_delta
    conved = np.array(conved)
    mapShape = conved.shape
    error_pool_shape = pooled_delta.shape
    # conved = conved.reshape(conved.shape[0],conved.shape[1],conved.shape[2])
    conved_error_map = np.zeros_like(conved)
    print "error map shape :",conved_error_map.shape
    print 'conved data shape :',conved.shape
    print 'pooled_delta shape :',pooled_delta.shape
    w = int(conved.shape[1] / pooled_delta.shape[1] )
    h = int (conved.shape[2] / pooled_delta.shape[2])
    conved = conved.flatten()
    pooled_delta = pooled_delta.flatten()
    conved_error_map = conved_error_map.flatten()
    i = 0
    step = error_pool_shape[1] * error_pool_shape[2]
    print 'step size :',step
    print 'Conved len : ',len(conved)

    while (i < len(conved)):
        tmp = conved[i:i+step]
        maxPos = tmp.argmax();
        corrs_pool_pos = maxPos % step
        print 'corrs_pool_pos :',corrs_pool_pos," error:",pooled_delta[corrs_pool_pos]
        conved_error_map[maxPos] = pooled_delta[corrs_pool_pos]
        i = i + step

    # print 'conved_error_map :',conved_error_map
    conved_error_map = conved_error_map.reshape(mapShape)
    return conved_error_map



# Prdict
def predict(img,label):
    global weight_matrix_1,weight_matrix_2,weight_matrix_outPut,kernels_0,kernels_1
    # input
    img = (np.array(img,dtype="uint8")).reshape((28,28,1))
    # label vector
    labeles = np.zeros((10,1))
    labeles[label-1,0] = 1

    # conved
    all_layer_output, all_layer_output_Convoled = conv(img)
    print 'shape of alloutput :',np.array(all_layer_output).shape
    print "shape of all convolved output:",np.array(all_layer_output_Convoled).shape
    # show
    # show(no_conv_layer,no_kernels,all_layer_output)
    #fullConnected
    fc = flattern_data(all_layer_output)
    print "fullConnected shape:",fc.shape
    print "Weight 1shape :",weight_matrix_1.shape


    # nueral net
    hidden_layer_1_out = sigmoid(np.dot(weight_matrix_1,fc))
    # print 'hidden_layer_1 :', hidden_layer_1_out.shape
    # print "weight_matrix_2 :",weight_matrix_2.shape
    hidden_layer_2_out = sigmoid(np.dot( weight_matrix_2, hidden_layer_1_out))
    final_output = sigmoid(np.dot(weight_matrix_outPut, hidden_layer_2_out))
    #
    print "final :",final_output
    print "Arg max: ",final_output.argmax()," max: ",final_output.max()
    print "Label: ",label

    cost,probs = softmax_cost(final_output,labeles)
    # back prop
    # Final out put layer
    delta3 =   np.multiply( (probs - labeles.reshape((10,1)) ),sigmoidPrime(np.dot(weight_matrix_outPut, hidden_layer_2_out)) )
    dedw3 = np.dot( delta3, hidden_layer_2_out.T)
    # Hidden layer 2
    delta2  =  np.dot(weight_matrix_outPut.T, delta3) * sigmoidPrime(np.dot( weight_matrix_2, hidden_layer_1_out))
    dedw2   =  np.dot(delta2, hidden_layer_1_out.T)
    # print 'dedw2 shape :',dedw2.shape
    # Hidden layer 1
    delta1 =  np.dot(weight_matrix_2.T , delta2) * sigmoidPrime( np.dot(weight_matrix_1,fc) )
    dedw1  =  np.dot(delta1, fc.T)
    # First layer Fully connected
    delta0 = np.dot(weight_matrix_1.T, delta1) * fc

    # print "dedw1 shape :",dedw1.shape
    # print "W1 shape :",weight_matrix_1.shape
    # print "delta1 shape :",delta1.shape
    # print 'delta0 shape :',delta0.shape
    # print "delta0 :",delta0

    # update the weight matrix in ANN
    weight_matrix_1 = weight_matrix_1 - dedw1
    weight_matrix_2 = weight_matrix_2 - dedw2
    weight_matrix_outPut = weight_matrix_outPut - dedw3

    # into Conve convnet

    # arrang delta0 into matrix
    deltaOutPutImg = delta0.reshape((no_kernels,7,7,1))
    # print "delta img shape :", deltaOutPutImg.shape
    # map delta0
    dedx = 0
    i = no_conv_layer-1


    # mappedError = pool_error_map(all_layer_output_Convoled[1],deltaOutPutImg)
    # inputX = np.array(all_layer_output_Convoled[1])[0,:,:,:]
    # error = mappedError[0,:,:,:]
    # print 'mapped error shape :',error.shape
    # print 'input x shape :',inputX.shape
    #
    # dedw,time = t.conv2(inputX,error,paddSize)
    # print 'dedw at layer :','i'," and kernel # ;", 'j' ," shape :",dedw

    layer_delta_error = []
    while (i>=0):
        layer_kernal_gradiant = []
        if(i == no_conv_layer-1):
            mappedError = pool_error_map(all_layer_output_Convoled[i],deltaOutPutImg)
            inputX = np.array(all_layer_output_Convoled[i])
            # print 'mapped error shape :',mappedError.shape
            # print 'input x shape :',inputX.shape
            for j in range(0,mappedError.shape[0]):
                # dedw = np.multiply(all_layer_output_Convoled[i] , mappedError)
                # print 'dedw at layer :',i," shape :",dedw.shape
                outputShape = (3,3,1)
                dedw,time = t.conv2(inputX[j,:,:,:],mappedError[j,:,:,:],paddSize,outputShape)
                layer_kernal_gradiant.extend(dedw)
                # layer delta error
                kernel =  kernels_0[j,:,:,:]
                kernel = np.rot90(kernel,2,(1,2)) # 180 dgre kernel rotetion
                # print "kernels_0 shape :",kernel.shape
                delta_error,time = t.conv2(mappedError[j,:,:,:],kernel,2,(14,14,1))
                layer_delta_error.extend(delta_error)

                # print 'dedw at layer :',i," and kernel # ;", j ," shape :",dedw.shape
                # print 'delta_error at layer :',i," and kernel # ;", j ," shape :",delta_error

            # Update the kernel weights
            # print "gradiant len :",np.array(layer_kernal_gradiant).reshape(3,3,3,1)
            kernels_0 = kernels_0 - np.array(layer_kernal_gradiant).reshape(3,3,3,1)
            # print "After update kernel shape : ",kernels_0.shape

        else:
            deltaOutPutImg = np.array(layer_delta_error).reshape((3,14,14,1))
            mappedError = pool_error_map(all_layer_output_Convoled[i],deltaOutPutImg)
            inputX = np.array(all_layer_output_Convoled[i])
            # print 'deltaOutPutImg :',deltaOutPutImg.shape
            # print 'mapped error shape :',mappedError.shape
            # print 'input x shape :',inputX.shape
            for j in range(0,mappedError.shape[0]):
                outputShape = (3,3,1)
                dedw,time = t.conv2(inputX[j,:,:,:],mappedError[j,:,:,:],paddSize,outputShape)
                layer_kernal_gradiant.extend(dedw)
                # #layer delta error
                # kernel =  kernels_0[j,:,:,:]
                # kernel = np.rot90(kernel,2,(1,2)) # 180 dgre kernel rotetion
                # print "kernels_0 shape :",kernel.shape
                # delta_error,time = t.conv2(mappedError[j,:,:,:],kernel,paddSize,(14,14,1))
                # layer_delta_error.extend(delta_error)

                # print 'dedw at layer :',i," and kernel # ;", j ," shape :",dedw
                # print 'delta_error at layer :',i," and kernel # ;", j ," shape :",delta_error.shape

            # Update the kernel weights
            # print "gradiant len :",len(layer_kernal_gradiant)
            kernels_1 = kernels_1 - np.array(layer_kernal_gradiant).reshape(3,3,3,1)
            # print "After update kernel at layer ",i," kernel is : ",np.array(layer_kernal_gradiant).reshape(3,3,3,1)


        i = i-1


def train():
    i = 0;
    till = len(images)/2
    print "Total data len:",len(images)," till :",till
    while(i < till):
        print " \n \n For Ideration I is : ",i
        predict(images[i],labels[i])
        i = i+1

# predict(images[0],labels[0])
train()
