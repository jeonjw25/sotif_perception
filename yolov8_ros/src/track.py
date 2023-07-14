#!/usr/bin/env python3

import time, sys, os
import torch
from ros import rosbag
import roslib, rospy
from rospy import Time
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from cam_calibration.msg import gt_5
# from trackers.multi_tracker_zoo import create_tracker
from evaluator import Evaluator
from homography import calRelativeVal
from classify_traffic_light import classify_traffic_light
from utils import calculate_curvature
import math
import message_filters

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov8_tracking" 


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

@torch.no_grad()

class Yolov8_Tracking:
    global gt_list
    global pred_list
    gt_list = [] 
    pred_list = []
    def __init__(self):
        self.device = 'cuda:' + str(rospy.get_param("~device",""))
        self.num = 0
        self.num2 = 0
        # Initialize weights 
        weights = rospy.get_param("~weights")
        
         # Setting inference size
        self.img_size_w = rospy.get_param("~inference_size_w")
        self.img_size_h = rospy.get_param("~inference_size_h")
        self.img_size = [self.img_size_h, self.img_size_w]
        self.model = YOLO(weights)
        self.cls_dict = {0: "car", 1: "truck", 2: "traffic_light", 3: "pedestrian", 4: "stop_sign", 5: "cyclist"}
        self.flag = True
        # Initialize CV_Bridge
        self.bridge = CvBridge()
         # tracker
        self.prev_frames = None
        self.prev_time = Time(0)
        self.prev_distances = {}

        self.seq = 0

        # Initialize subscriber to Image/CompressedImage topic
        self.image_sub = rospy.Subscriber(
                rospy.get_param("~input"), Image, self.callback, queue_size=100)

        self.gt_sub = rospy.Subscriber(
                rospy.get_param("~gt"), gt_5, self.callback_gt, queue_size=100)

        # self.image_sub = message_filters.Subscriber(
        #         rospy.get_param("~input"), Image)
        
        # self.gt_sub = message_filters.Subscriber(
        #                 rospy.get_param("~gt"), gt_5)

        # ats = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.gt_sub], 
        #                                                   queue_size=100, slop=10, allow_headerless=True)
        # ats.registerCallback(self.callback)
        
        # rospy.sleep(3)
        
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=10)
        

        print("callback")

    def callback_gt(self, data):
        if self.flag:
            self.gt = data
            
            print("gt_num: ", self.num2)
            self.num2 += 1
        # self.flag = False
    
    def callback(self, image):
        """adapted from yolov5/detect.py"""
        
        if self.flag:
            # self.gt = gt_5
            # self.num2 += 1
            # print("gt_num: ", self.num2)
            self.seq = image.header.seq
            
            curr_time = image.header.stamp
            detected_boxes = []
            im = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width,  -1)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            
            preds = self.model.track(source=im)
            # im = RGB, im0 = BGR
            # im, im0 = self.preprocess(im)
            im0 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cnt = 0
            cnt2 = 0

            bev, lp, rp, lane_quality = calculate_curvature(im0)
            detected = []
            prev_frames = self.prev_frames
            curr_frames = im0
            
            curr_time = curr_time.to_sec()
            objects = preds[0].boxes
            total_obj_num = len(preds[0].boxes.cls)
            
            if total_obj_num > 0:
                distances = self.prev_distances
                for i in range(total_obj_num):
                    cls = int(objects.cls[i])
                    xyxy = objects.xyxy[i]
                    conf = float(objects.conf[i])
                    id = objects.id[i]
                    dist_x, dist_y = 0, 0

                    # relative distance
                    if cls in [0, 1, 3, 5]:
                        dist_calculator = calRelativeVal(img=im0, bbox = xyxy)

                        dist_x, dist_y = dist_calculator.calculate_3d_coord()
                        
                        curr_dist = round(math.sqrt((dist_x)**2 + (dist_y)**2),2)
                        
                        if (cls, id) not in distances or distances[(cls, id)] is None:
                            distances[(cls,id)] = []
                            distances[(cls,id)].append(dist_x)
                            distances[(cls,id)].append(dist_y)
                            distances[(cls,id)].append(curr_dist)
                            distances[(cls,id)].append(curr_time)
                            # cv2.putText(im0, str(curr_dist)+'m',
                            #                 (int(xyxy[0]), int(xyxy[1] + 15 if xyxy[1] <20 else xyxy[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)                   
                        else:
                            prev_dist = distances[(cls,id)]
                            if curr_time - prev_dist[3] < 2 :
                                speed_x = round((dist_x - prev_dist[0]) / (curr_time-prev_dist[3]),2)
                                speed_y = round((dist_y - prev_dist[1]) / (curr_time-prev_dist[3]),2)
                                speed = round((curr_dist - prev_dist[2]) / (curr_time-prev_dist[3]),2)
                            #     cv2.putText(im0, str(speed)+'m/s',
                            #                 (int(xyxy[2]), int(xyxy[1] + 35 if xyxy[1] <20 else xyxy[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)  
                            # cv2.putText(im0, str(curr_dist)+'m',
                            #                 (int(xyxy[0]), int(xyxy[1] + 15 if xyxy[1] <20 else xyxy[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                            distances[(cls,id)] = [dist_x, dist_y, curr_dist, curr_time]
                    
                    # traffic light
                    elif cls == 2:
                            print(classify_traffic_light(img_bgr = im, box = xyxy)) 
                    
                    map(int, xyxy)
                    im = cv2.rectangle(im, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 3)
                    im = cv2.putText(im, f"id {id}_" + self.cls_dict[cls], (int(xyxy[0])+13, int(xyxy[1])+13), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2)
                    
                    detected.append([self.num, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), self.cls_dict[cls], conf, dist_x, dist_y])
            # print(detected)    
            pred_list.append(detected)
            

            gts = []
            # print(self.gt.obj_1)
            gts.append(list(self.gt.obj_1))
            gts.append(list(self.gt.obj_2))
            gts.append(list(self.gt.obj_3))
            gts.append(list(self.gt.obj_4))
            gts.append(list(self.gt.obj_5))
            gts.append(list(self.gt.obj_6))
            gts.append(list(self.gt.obj_7))
            gts.append(list(self.gt.obj_8))
            gts.append(list(self.gt.obj_9))
            gts.append(list(self.gt.obj_10))
            gts.append(list(self.gt.obj_11))
            gts.append(list(self.gt.obj_12))
            gts.append(list(self.gt.obj_13))
            gts.append(list(self.gt.obj_14))
            gts.append(list(self.gt.obj_15))
            gts.append(list(self.gt.obj_16))
            gts.append(list(self.gt.obj_17))
            gts.append(list(self.gt.obj_18))
            gts.append(list(self.gt.obj_19))
            gts.append(list(self.gt.obj_20))
            gts_ = []
            global tmp
            tmp = gts
            for g in gts:
                if g[0] != -1:
                    cnt2 += 1
                    xmin = int(g[1] - (g[3]/2))
                    xmax = int(g[1] + (g[3]/2))
                    ymin = int(g[2] - (g[4]/2))
                    ymax = int(g[2] + (g[4]/2))
                    

                    bbox = [xmin, ymin, xmax, ymax]
                    gt = [self.num, xmin, ymin, xmax, ymax, g[0], 1, round(g[5], 2), round(g[6],2)]
                    
                    # im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
                else:
                    gt = []
                gts_.append(gt)
            
            gt_list.append(gts_)
            # if self.num >= 1:
                # for gt in gt_list[self.num - 1]:
                    # print(gt)
                    # for t in gt:
                    #     if t[0] != -1:
                    #         cnt2 += 1
                    #         xmin = int(t[1] - (t[3]/2))
                    #         xmax = int(t[1] + (t[3]/2))
                    #         ymin = int(t[2] - (t[4]/2))
                    #         ymax = int(t[2] + (t[4]/2))
                    #         bbox = [xmin, ymin, xmax, ymax]
                    #         im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        cv2.imshow("im", im)
        cv2.waitKey(1)
        img_filename = "/root/catkin_ws/src/yolov8_ros/result/result_image_{:d}.png".format(self.num)
        cv2.imwrite(img_filename, im)  
        print("img_num: ", self.num)
        self.num += 1           
        self.flag = True

if __name__ == "__main__":

    rospy.init_node("yolov8", anonymous=True)
    
    detector = Yolov8_Tracking()
    
    rospy.spin()
    # print(len(gt_list), gt_list)
    # print(len(pred_list), pred_list)

    evaluator = Evaluator(preds=pred_list, gts=gt_list)
