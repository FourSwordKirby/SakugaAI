import cv2
import numpy as np

def EdgeDetect(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    canny = cv2.Canny(img, 50, 240)

    cv2.imshow('Original', img)
    cv2.imshow('Sobel horizontal', sobel_horizontal)
    cv2.imshow('Sobel vertical', sobel_vertical)
    cv2.imshow('Laplace', laplacian)
    #Canny seems to work the best straight up
    cv2.imshow('Canny', canny)
    cv2.waitKey(0)

def OpticalFlow():
    cap = cv2.VideoCapture("test2.avi")
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next
    cap.release()
    cv2.destroyAllWindows()

OpticalFlow()
#EdgeDetect('test.png')
#EdgeDetect('test2.png')
# EdgeDetect('test3.png')
# EdgeDetect('test4.png')
# EdgeDetect('test5.png')