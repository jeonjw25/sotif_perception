#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
import math
from rostopic import get_topic_type
from rospy import Time

from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from cam_calibration.msg import gt_5

from evaluator import Evaluator

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

# import from yolov5 submodules
from ultralytics import YOLO
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from yolov8.ultralytics.yolo.data.augment import LetterBox

from trackers.multi_tracker_zoo import create_tracker

# import from oedr
from homography import calRelativeVal
from classify_traffic_light import classify_traffic_light
from utils import calculate_curvature
from ultralytics import YOLO

@torch.no_grad()



class Yolov8_Tracking:

    global gt_list
    global pred_list
    gt_list = [] 
    pred_list = []
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.max_det = rospy.get_param("~maximum_detections")
        self.device = select_device(str(rospy.get_param("~device","")))
        self.classes = rospy.get_param("~classes", None)
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.view_image = rospy.get_param("~view_image")
        self.line_thickness = rospy.get_param("~line_thickness")
        self.half = rospy.get_param("~half", False)
        self.num = 0
        self.num2 = 0
        # Initialize weights 
        weights = rospy.get_param("~weights")
        
        # Half
        self.half = rospy.get_param("~half", False)
          
        
        # Initialize device
        self.model = AutoBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), fp16=bool(rospy.get_param("~data")))
        # self.model = YOLO(weights)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )
        
        # Setting inference size
        self.img_size_w = rospy.get_param("~inference_size_w")
        self.img_size_h = rospy.get_param("~inference_size_h")
        self.img_size = [self.img_size_h, self.img_size_w]
        self.img_size = check_imgsz(self.img_size, stride=self.stride)

        # Initialize subscriber to Image/CompressedImage topic
        self.image_sub = rospy.Subscriber(
                rospy.get_param("~input"), Image, self.callback, queue_size=100)

       
        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        )
        self.gt_sub = rospy.Subscriber(
                rospy.get_param("~gt"), gt_5, self.callback_gt, queue_size=100)
        
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=10
            )
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()
        
        # tracker
        self.prev_frames = None
        self.prev_time = Time(0)
        self.prev_distances = {}
        print(self.device)
        tracker_type = rospy.get_param("~tracker_type")
        tracking_config = rospy.get_param("~tracking_config")
        reid_weights = rospy.get_param("~reid_weights")
        self.tracker = create_tracker(tracker_type=tracker_type, 
                                      tracker_config=tracking_config,
                                      reid_weights=reid_weights,
                                      device=self.device, 
                                      half=self.half)
        self._image_counter = 0

        self.seq = 0
        self.flag = True
    
    def callback_gt(self, data):
        if self.flag:
            self.gt = data
            
            print("gt_num: ", self.num2)
            self.num2 += 1
        self.flag = False

    def callback(self, data):
        """adapted from yolov5/detect.py"""
        if not self.flag:
        
            self.seq = data.header.seq
            
            curr_time = data.header.stamp
            detected_boxes = []
            im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width,  -1)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            
            # im = RGB, im0 = BGR
            im, im0 = self.preprocess(im)
            cnt = 0
            cnt2 = 0
            
            
            bev, lp, rp, lane_quality = calculate_curvature(im0)
            
            im0_copy = im0.copy()

            tracker_list = []
            outputs = [None]
            tracker = self.tracker
                    
                

            # Run inference
            im = torch.from_numpy(im).to(self.device) 
            im = im.half() if self.half else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
                
            pred = self.model(im, augment=False, visualize=False)
            
          
            # seg
            # masks = []
            p = non_max_suppression(
                pred[0], self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32
            )
            # proto = pred[1][-1] if len(pred[1]) == 3 else pred[1]
            
            
            ### To-do move pred to CPU and fill BoundingBox messages
            
            # Process predictions 
            det = p[0]
            
            prev_frames = self.prev_frames
            curr_frames = im0
            
            
            curr_time = curr_time.to_sec()
            
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            
            # if prev_frames is not None:
            #     annotator = Annotator(prev_frames, line_width=self.line_thickness, example=str(self.names))
        
            
            if hasattr(tracker, 'tracker') and hasattr(tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:  # camera motion compensation
                    tracker.camera_update(prev_frames, curr_frames)
                    
            
            
            detected = []
            if det is not None and len(det):
                shape = im0.shape
                # scale bbox first the crop masks
                #masks.append(process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                
                outputs = tracker.update(det.cpu(),im0)
                
                # retina_masks = False
                
                if len(outputs) > 0:
                    # annotator.masks(
                    #             masks[0],
                    #             colors=[colors(x, True) for x in det[:, 5]],
                    #             # im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                    #             # 255 if retina_masks else im[0]
                    #             im_gpu=im[0]
                    #         )
                    distances = self.prev_distances
                    
                    for i, (output) in enumerate(outputs):
                        cnt += 1
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        
                            
                        c = int(cls)
                        if self.names[c] == 'traffic_light':
                            print(self.names[c])
                            self.names[c] = classify_traffic_light(img_bgr = im0_copy, box = output) 
                            percentage = round(percentage, 2)
                            id = int(id)
                            label = f'{id} {self.names[c]} {conf:.2f}'
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)
                            
                            self.names[c] = 'traffic_light'

                        else:
                            id = int(id)
                            label = f'{id} {self.names[c]} {conf:.2f}'
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)
                            
                        # bounding_box = BoundingBox()
                        
                        # # Fill in bounding box message
                        # bounding_box.Class = self.names[c]
                        # bounding_box.probability = conf 
                        # bounding_box.xmin = int(bbox[0])
                        # bounding_box.ymin = int(bbox[1])
                        # bounding_box.xmax = int(bbox[2])
                        # bounding_box.ymax = int(bbox[3])
                        dist_x = 0
                        dist_y = 0
                        

                        if self.names[c] in ['car', 'truck', 'cyclist', 'person']:
                            dist_calculator = calRelativeVal(img=im0, bbox = bbox)

                            dist_x, dist_y = dist_calculator.calculate_3d_coord()
                            
                            curr_dist = round(math.sqrt((dist_x)**2 + (dist_y)**2),2)
                            
                            if (c, id) not in distances or distances[(c, id)] is None:
                                distances[(c,id)] = []
                                distances[(c,id)].append(dist_x)
                                distances[(c,id)].append(dist_y)
                                distances[(c,id)].append(curr_dist)
                                distances[(c,id)].append(curr_time)
                                cv2.putText(im0, str(curr_dist)+'m',
                                                (int(bbox[0]), int(bbox[1] + 15 if bbox[1] <20 else bbox[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)                   
                            else:
                                prev_dist = distances[(c,id)]
                                if curr_time - prev_dist[3] < 2 :
                                    speed_x = round((dist_x - prev_dist[0]) / (curr_time-prev_dist[3]),2)
                                    speed_y = round((dist_y - prev_dist[1]) / (curr_time-prev_dist[3]),2)
                                    speed = round((curr_dist - prev_dist[2]) / (curr_time-prev_dist[3]),2)
                                    cv2.putText(im0, str(speed)+'m/s',
                                                (int(bbox[2]), int(bbox[1] + 35 if bbox[1] <20 else bbox[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)  
                                cv2.putText(im0, str(curr_dist)+'m',
                                                (int(bbox[0]), int(bbox[1] + 15 if bbox[1] <20 else bbox[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                                distances[(c,id)] = [dist_x, dist_y, curr_dist, curr_time]
                        pred_ = [self.num, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), self.names[c], conf, dist_x, dist_y]      
                        detected.append(pred_)

                    self.prev_distances = distances        

                        
            
                        

                        # bounding_boxes.bounding_boxes.append(bounding_box)
                    #     ### POPULATE THE DETECTION MESSAGE HERE
                    
                else:
                    pass

            
                # Stream results
                im0 = annotator.result()
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
            for g in gts:
                if g[0] != -1:
                    cnt2 += 1
                    xmin = int(g[1] - (g[3]/2))
                    xmax = int(g[1] + (g[3]/2))
                    ymin = int(g[2] - (g[4]/2))
                    ymax = int(g[2] + (g[4]/2))
                    

                    bbox = [xmin, ymin, xmax, ymax]
                    gt = [self.num, xmin, ymin, xmax, ymax, g[0], 1, round(g[5], 2), round(g[6],2)]
                    if prev_frames is not None: 
                        cv2.rectangle(im0, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
                else:
                    gt = []
                gts_.append(gt)
            gt_list.append(gts_)
            
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
            
            print("img_num: ", self.num)
            self.num += 1
            
            # Publish & visualize images
            if prev_frames is not None:
                if self.view_image:
                    img_filename = "/root/catkin_ws/src/yolov8_ros/result/result_image_{:03d}.png".format(self._image_counter)
                    bev_filename = "/root/catkin_ws/src/yolov8_ros/result/bev/bev_image_{:03d}.png".format(self._image_counter)
                    cv2.imwrite(img_filename, im0)
                    cv2.imwrite(bev_filename, bev)
                    self._image_counter += 1
                    
                    
            if self.publish_image:
                self.image_pub.publish(np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1))
            
            self.prev_frames = curr_frames
            self.prev_time = curr_time
            self.flag = True

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([LetterBox(self.img_size, False, stride=self.stride)(image=img0)])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5", anonymous=True)
    
    detector = Yolov8_Tracking()
    
    rospy.spin()
    # print(len(gt_list), gt_list)
    # print(len(pred_list), pred_list)

    evaluator = Evaluator(preds=pred_list, gts=gt_list)
