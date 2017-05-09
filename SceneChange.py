# Third-Party Library Imports
import cv2
import numpy
from PIL import Image

import json
from pprint import pprint
import glob, os, sys
import shutil

class SceneDetector(object):
    """Base SceneDetector class to implement a scene detection algorithm."""
    def __init__(self):
        pass

    def process_frame(self, frame_num, frame_img, frame_metrics, scene_list):
        """Computes/stores metrics and detects any scene changes.
        Prototype method, no actual detection.
        """
        return

    def post_process(self, scene_list):
        pass

class ContentDetector(SceneDetector):
    """Detects fast cuts using changes in colour and intensity between frames.
    Since the difference between frames is used, unlike the ThresholdDetector,
    only fast cuts are detected with this method.  To detect slow fades between
    content scenes still using HSV information, use the DissolveDetector.
    """

    def __init__(self, threshold = 30.0, min_scene_len = 15):
        super(ContentDetector, self).__init__()
        self.threshold = threshold
        self.min_scene_len = min_scene_len  # minimum length of any given scene, in frames

    def process_frame(self, frame1, frame2):
        # Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        # of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).
    
        if frame1 is not None:
            # Change in average of HSV (hsv), (h)ue only, (s)aturation only, (l)uminance only.
            delta_hsv_avg, delta_h, delta_s, delta_v = 0.0, 0.0, 0.0, 0.0

            num_pixels = frame2.shape[0] * frame2.shape[1]
            last_hsv = cv2.split(cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV))
            curr_hsv = cv2.split(cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV))
            
            delta_hsv = [-1, -1, -1]
            for i in range(3):
                num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
                curr_hsv[i] = curr_hsv[i].astype(numpy.int32)
                last_hsv[i] = last_hsv[i].astype(numpy.int32)
                delta_hsv[i] = numpy.sum(numpy.abs(curr_hsv[i] - last_hsv[i])) / float(num_pixels)
            delta_hsv.append(sum(delta_hsv) / 3.0)
            delta_h, delta_s, delta_v, delta_hsv_avg = delta_hsv

            # frame_metrics[frame_num]['delta_hsv_avg'] = delta_hsv_avg
            # frame_metrics[frame_num]['delta_hue'] = delta_h
            # frame_metrics[frame_num]['delta_sat'] = delta_s
            # frame_metrics[frame_num]['delta_lum'] = delta_v

            return delta_hsv_avg

    def post_process(self, scene_list):
        """Not used for ContentDetector, as cuts are written as they are found."""
        return

detector = ContentDetector()

similar_dirs = []
bad_dirs = []
diff_stats = []

root= os.getcwd() + "/Data"

for episode in os.listdir(root):
    episode_dir = os.path.join(root, episode)
    if os.path.isdir(episode_dir):
        i = 0
        for group in os.listdir(episode_dir):
            group_dir = os.path.join(episode_dir, group)

            if os.path.isdir(group_dir):
                i+=1
                if(i % 200 == 0):
                    print(i)

                frames = []
                for filename in os.listdir(group_dir):
                    if "edge" in filename or "flow" in filename:
                        continue
                    frame = cv2.imread(group_dir + "/" + filename)
                    frames.append(frame)

                assert(len(frames) == 3)

                # print(detector.process_frame(frames[0], frames[1]))
                # print(detector.process_frame(frames[1], frames[2]))
                # print(detector.process_frame(frames[0], frames[2]))

                diff_stats.append(detector.process_frame(frames[0], frames[1]))
                diff_stats.append(detector.process_frame(frames[1], frames[2]))
                diff_stats.append(detector.process_frame(frames[0], frames[2]))

                #very basic, if one frame of the 3 frames is very different from another
                #we assume a cut happened
                if(detector.process_frame(frames[0], frames[1]) > 30.0 or
                    detector.process_frame(frames[1], frames[2]) > 30.0 or
                    detector.process_frame(frames[0], frames[2]) > 30.0):
                    bad_dirs.append((group_dir, episode))

                if(detector.process_frame(frames[0], frames[1]) <= 0.1 and
                    detector.process_frame(frames[1], frames[2]) <= 0.1 and
                    detector.process_frame(frames[0], frames[2]) <= 0.1):
                    similar_dirs.append((group_dir, episode))
        break

bad_loc = os.getcwd() + "/Bad_Data"
for bad_dir in bad_dirs:
    dir_name = bad_dir[0]
    episode_name = bad_dir[1]
    if not os.path.exists(bad_loc + "/" + episode_name):
        os.makedirs(bad_loc + "/" + episode_name)
    shutil.move(dir_name, bad_loc + "/" + episode_name)

    
sim_loc = os.getcwd() + "/Similar_Data"
for sim_dir in similar_dirs:
    dir_name = sim_dir[0]
    episode_name = sim_dir[1]
    if not os.path.exists(sim_loc + "/" + episode_name):
        os.makedirs(sim_loc + "/" + episode_name)
    shutil.move(dir_name, sim_loc + "/" + episode_name)

print(len(bad_dirs))
print(numpy.mean(diff_stats))
print(numpy.std(diff_stats))

print(len(similar_dirs))