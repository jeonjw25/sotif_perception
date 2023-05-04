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
from rostopic import get_topic_type

from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes


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


@torch.no_grad()
class Yolov8_Tracking:
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
        
        # Initialize weights 
        weights = rospy.get_param("~weights")

        # Half
        self.half = rospy.get_param("~half", False)
                
        
        # Initialize device
        self.model = AutoBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), fp16=bool(rospy.get_param("~data")))
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
        self.img_size = [self.img_size_w, self.img_size_h]
        self.img_size = check_imgsz(self.img_size, stride=self.stride)
        
        
        # Initialize subscriber to Image/CompressedImage topic
        self.image_sub = rospy.Subscriber(
                rospy.get_param("~input"), Image, self.callback, queue_size=1)


        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        )
        
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
        tracker_type = rospy.get_param("~tracker_type")
        tracking_config = rospy.get_param("~tracking_config")
        reid_weights = rospy.get_param("~reid_weights")
        self.tracker = create_tracker(tracker_type=tracker_type, 
                                      tracker_config=tracking_config,
                                      reid_weights=reid_weights,
                                      device=self.device, 
                                      half=self.half)
        self._image_counter = 0

    def callback(self, data):
        """adapted from yolov5/detect.py"""
        # print(data.header)
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width,  -1)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        # im = RGB, im0 = BGR
        im, im0 = self.preprocess(im)
        bev, lp, rp, lane_quality = calculate_curvature(im0)
        

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
        masks = []
        p = non_max_suppression(
            pred[0], self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32
        )
        proto = pred[1][-1] if len(pred[1]) == 3 else pred[1]
        
        
        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = p[0]
        
        
        # a = det[:,5]
        # a = torch.tensor(a, device=self.device, dtype=torch.float32) / 255.0  # shape(n,3)
        # print(a.shape)
        # a = a[:, None, None]  # shape(n,1,1,3)
        # print(a.shape)
        # bounding_boxes = BoundingBoxes()
        # bounding_boxes.header = data.header
        # bounding_boxes.image_header = data.header
        
        curr_frames = im0
        prev_frames = self.prev_frames
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        
        
        
        if hasattr(tracker, 'tracker') and hasattr(tracker, 'camera_update'):
            if prev_frames is not None and curr_frames is not None:  # camera motion compensation
                tracker.camera_update(prev_frames, curr_frames)
                
            self.prev_frames = curr_frames
        
        if det is not None and len(det):
            shape = im0.shape
            # scale bbox first the crop masks
            masks.append(process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
            
            outputs = tracker.update(det.cpu(),im0)
            
            # retina_masks = False
            if len(outputs) > 0:
                annotator.masks(
                            masks[0],
                            colors=[colors(x, True) for x in det[:, 5]],
                            # im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            # 255 if retina_masks else im[0]
                            im_gpu=im[0]
                        )
                for i, (output) in enumerate(outputs):
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]
                        
                    c = int(cls)
                    if self.names[c] == 'traffic light':
                        self.names[c] = classify_traffic_light(img_rgb = im0, box = output) 
                    
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
                    
                    if bbox[1] < 20:
                        text_pos_y = bbox[1] + 30
                    else:
                        text_pos_y = bbox[1] - 10
                        
                    
                    
                    if self.names[c] in ['car', 'truck', 'bus', 'person']:
                        dist_calculator = calRelativeVal(img=im0, bbox = bbox)
                        x, y, z = dist_calculator.calculate_3d_coord()
                        cv2.putText(im0, str(round(x/z, 2))+'m',
                            (int(bbox[0]), int(text_pos_y)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                       

                    # bounding_boxes.bounding_boxes.append(bounding_box)

                #     # Annotate the image
                #     if self.publish_image or self.view_image:  # Add bbox to image
                #         # integer class
                #         label = f"{id}{self.names[c]} {conf:.2f}"
                #         annotator.box_label(xyxy, label, color=colors(c, True))       

                    
                #     ### POPULATE THE DETECTION MESSAGE HERE
            else:
                pass
            # Stream results
            im0 = annotator.result()
            
            

        # Publish prediction
        # self.pred_pub.publish(bounding_boxes)
        
        cv2.imshow("im0", im0)
        cv2.waitKey(0)
        # Publish & visualize images
        if self.view_image:
            img_filename = "/root/catkin_ws/src/yolov8_ros/result/result_image_{:03d}.png".format(self._image_counter)
            bev_filename = "/root/catkin_ws/src/yolov8_ros/result/bev/bev_image_{:03d}.png".format(self._image_counter)
            
            #cv2.imwrite(bev_filename, bev)
            self._image_counter += 1
        
            
            # cv2.imshow(str(0), im0)
            # cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            self.image_pub.publish(np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1))
        

    
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

    
    # def make_prev(self, img):
    #     prev_frames = img
        
    #     return prev_frames


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5", anonymous=True)
    detector = Yolov8_Tracking()
    
    rospy.spin()