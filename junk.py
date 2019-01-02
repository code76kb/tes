import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

k1 = np.random.randint(10,size = (2,2))

# print "k1 :",k1
# k1.dump('k1.dat')
#
# k0 = np.load("k1.dat")
# print "\n k0 :",k0
# print  "\n k0 len :",len(k0)

k = np.load("kernels_0.dat")
# cv2.imshow("k0",k[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(np.random.random((50,50)));
# plt.colorbar()
# plt.show()
k0 = k[0]
print "\n k0;",k

plt.imshow(k0.reshape(3,3))
plt.show()
