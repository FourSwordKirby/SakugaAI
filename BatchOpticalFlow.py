import cv2
import numpy as np
from PIL import Image
import pdb
import os

flow_adjust = 0.3

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T * flow_adjust
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow * flow_adjust
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

# frame1 = cv2.imread('f1.png')
# frame3 = cv2.imread('f3.png')

# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# nxt = cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)
# flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# img = Image.fromarray(np.asarray(warp_flow(nxt, flow), dtype=np.uint8))
# img.save('frameinterp.png')
# img = Image.fromarray(np.asarray(draw_flow(nxt, flow), dtype=np.uint8))
# img.save('frameflow.png')

# def EdgeDetect(filename):
#     img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#     rows, cols = img.shape

#     canny = cv2.Canny(img, 50, 240)

#     # cv2.imshow('Canny', canny)
#     return Image.fromarray(canny)


# frame1 = cv2.imread('edge1.png')
# frame3 = cv2.imread('edge3.png')
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# nxt = cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)
# flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# img = Image.fromarray(np.asarray(warp_flow(nxt, flow), dtype=np.uint8))
# img.save('edge_frameinterp.png')
# img = Image.fromarray(np.asarray(draw_flow(nxt, flow), dtype=np.uint8))
# img.save('edge_frameflow.png')

a = 1/0
# cv2.imshow('frame1', frame1)
# cv2.imshow('frame3', frame3)
# cv2.imshow('flow', draw_flow(nxt, flow))
# cv2.imshow('interp', warp_flow(nxt, flow))

# frame1 = cv2.imread('edge_1.png')
# frame3 = cv2.imread('edge_3.png')
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# nxt = cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)

# flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# img = Image.fromarray(np.asarray(warp_flow(nxt, flow), dtype=np.uint8))
# img.save('frameinterp_edge.png')
# img = Image.fromarray(np.asarray(draw_flow(nxt, flow), dtype=np.uint8))
# img.save('frameflow_edge.png')


root= os.getcwd() + "/Data"

for item in os.listdir(root):
    episode_dir = os.path.join(root, item)
    if os.path.isdir(episode_dir):
        for group in os.listdir(episode_dir):
            group_dir = os.path.join(episode_dir, group)
            if os.path.isdir(group_dir):
                before_frame = None
                end_frame = None
                before_edge_frame = None
                end_edge_frame = None

                for filename in os.listdir(group_dir):
                    if "before" in filename:
                        if "edge" in filename:
                            before_edge_frame = cv2.imread(group_dir + "/" + filename)
                        else:
                            before_frame = cv2.imread(group_dir + "/" + filename)
                    elif "end" in filename:
                        if "edge" in filename:
                            end_edge_frame = cv2.imread(group_dir + "/" + filename)
                        else:
                            end_frame = cv2.imread(group_dir + "/" + filename)
                before_frame = cv2.cvtColor(before_frame,cv2.COLOR_BGR2GRAY)
                end_frame = cv2.cvtColor(end_frame,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(before_frame,end_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                img = Image.fromarray(np.asarray(warp_flow(end_frame, flow), dtype=np.uint8))
                img.save(group_dir + "/" + 'normal_flow.png')

                before_edge_frame = cv2.cvtColor(before_edge_frame,cv2.COLOR_BGR2GRAY)
                end_edge_frame = cv2.cvtColor(end_edge_frame,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(before_edge_frame,end_edge_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                img = Image.fromarray(np.asarray(warp_flow(end_edge_frame, flow), dtype=np.uint8))
                img.save(group_dir + "/" + 'edge_flow.png')
