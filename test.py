import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
img = np.zeros((512,512,1), np.uint8)
img28 = np.zeros((28,28,1),np.uint8)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if(drawing):
            cv2.circle(img,(x,y),10,(255,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # cv2.circle(img,(x,y),5,(255,255,255),-1)


def get():
    global img,img28
    img = np.zeros((512,512,1), np.uint8)
    img28 = np.zeros((28,28,1),np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    # cv2.destroyAllWindows()
    while(1):
        img28 = cv2.resize(img,(28,28),interpolation = cv2.INTER_AREA)
        cv2.imshow('image',img)
        cv2.imshow('img 28',img28)

        k = cv2.waitKey(1) & 0xFF
        if(k != 255):
            print ('\n key press :',k)

        if k == 27:
            cv2.destroyAllWindows()
            break

        if k == 13: # enter key
            cv2.destroyAllWindows()
            return img28

        if k == 99: # clear key "c"
            img = np.zeros((512,512,1), np.uint8)
            img28 = np.zeros((28,28,1),np.uint8)

# get()
