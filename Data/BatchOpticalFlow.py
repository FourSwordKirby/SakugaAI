import cv2
import numpy as np

f1 = cv2.imread('f1.png')
prvs = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
print(prvs.shape)
hsv = np.zeros_like(f1)
hsv[...,1] = 255
print(hsv.shape)

f3 = cv2.imread('f3.png')
nxt = cv2.cvtColor(f3,cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
print(ang.shape)
print(hsv[...,0].shape)

hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

print(rgb)
cv2.imshow('frame2', rgb)
k = cv2.waitKey()
if k == ord('s'):
    cv2.imwrite('opticalfb.png',prev)
    cv2.imwrite('opticalhsv.png',prev + rgb)
