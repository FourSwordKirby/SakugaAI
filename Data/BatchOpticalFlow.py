import cv2
import numpy as np
from PIL import Image
import pdb
f1 = cv2.imread('f1.png')
prvs = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(f1)
hsv[...,1] = 255

f3 = cv2.imread('f3.png')
nxt = cv2.cvtColor(f3,cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res



cap = cv2.VideoCapture("test.avi")

ret, frame1 = cap.read()
img = Image.fromarray(np.asarray(cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY), dtype=np.uint8))
img.save('frame1.png')

ret, frame2 = cap.read()
img = Image.fromarray(np.asarray(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY), dtype=np.uint8))
img.save('frame2.png')

cap.read()
cap.read()
ret, frame3 = cap.read()
img = Image.fromarray(np.asarray(cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY), dtype=np.uint8))
img.save('frame3.png')

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
nxt = cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
img = Image.fromarray(np.asarray(warp_flow(nxt, flow), dtype=np.uint8))
img.save('frameinterp.png')
img = Image.fromarray(np.asarray(draw_flow(nxt, flow), dtype=np.uint8))
img.save('frameflow.png')

# cv2.imshow('frame1', frame1)
# cv2.imshow('frame3', frame3)
# cv2.imshow('flow', draw_flow(nxt, flow))
# cv2.imshow('interp', warp_flow(nxt, flow))

cap.release()
cv2.destroyAllWindows()

