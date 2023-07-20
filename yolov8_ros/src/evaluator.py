
#  [[frame_id, xmin, ymin, xmax, ymax, cls, conf, rd_x, rd_y]
#   [frame_id, xmin, ymin, xmax, ymax, cls, conf, rd_x, rd_y]
#                          .
#                          .                                ]

import torch

from collections import Counter
import numpy as np
import os

class Evaluator():
    def __init__(self, preds, gts, bag_name, iters):
        self.preds = preds
        self.gts = gts
        self.cls_dict = {0: "car", 1: "truck", 2: "cyclist", 3: "pedestrian", 4: "stop_sign", 5: "traffic_light"}
        self.rd_x_e = []
        self.rd_y_e = []
        self.classes = []
        self.iters = iters
        self.bag_name = bag_name
        self.dir_path = f'/root/catkin_ws/src/yolov8_ros/result'
        self.file_name = f'{self.bag_name}_result.txt'
        self.file_name2 = f'{self.bag_name}_rdx_rdy_result.txt'
        self.pred_boxes, self.gt_boxes = self.preprocess(self.preds, self.gts) 
        # print(self.pred_boxes)
        # print(self.gt_boxes)
           
        self.mAP = self.calculate_mAP(self.pred_boxes, self.gt_boxes)
        # print(len(self.rd_x_e))
        
        self.rdx_mse, self.rdy_mse = self.rel_dist_MSE(self.rd_x_e, self.rd_y_e)
        
        print("mAP: ", self.mAP)
        print("rdx_MAE: ", self.rdx_mse, "m")
        print("rdy_MAE: ", self.rdy_mse, "m")

        file_path = os.path.join(self.dir_path, self.file_name)
        file_path2 = os.path.join(self.dir_path, self.file_name2)

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        with open(file_path, 'a') as file:
            file.write(f'{self.iters}th: ')
            file.write(f'mAP: {self.mAP}  ')
            file.write(f'rd_x_MAE: {self.rdx_mse}  ')
            file.write(f'rd_y_MAE: {self.rdy_mse} \n')
            #file.write("\n")
        
        with open(file_path2, 'a') as file:
            for i in self.pred_boxes:
                file.write(f'{i}\n')
            file.write('\n')
            


    def preprocess(self, preds, gts):
        gt_boxes = []
        pred_boxes = []
        gts = gts[1:]
        preds = preds[:-1]
        
        for gt in gts:
            for g in gt:
                if g:
                    if g[0] > 0:
                        g[0] = g[0] - 1
                        g[5] = self.cls_dict[g[5]]
                        if g[5] not in self.classes:
                            self.classes.append(g[5])
                        gt_boxes.append(g)

        for pred in preds:
            if pred:
                for p in pred:
                    pred_boxes.append(p)

        return pred_boxes, gt_boxes

    def rel_dist_MSE(self, rdx_e, rdy_e):
        rdx_mse = sum(rdx_e) / len(rdx_e)
        rdy_mse = sum(rdy_e) / len(rdy_e)

        return rdx_mse, rdy_mse
    
    def intersection_over_union(self, box1, box2):
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou
    
    def calculate_mAP(self, pred_boxes, gt_boxes, iou_threshold = 0.4):
        AP = [] # array containing AP of each class
        epsilon = 1e-6 # ???

        # classes = ['car', 'truck', 'person', 'traffic light', 'traffic sign', 'bicycle']
        # classes = ['car', 'traffic_light']
        for c in self.classes:
            detections = []
            ground_truths = []

            for box in pred_boxes:
                if box[5] == c:
                    detections.append(box)

            for box in gt_boxes:
                if box[5] == c:
                    ground_truths.append(box)
            
            amount_bboxes = Counter(gt[0] for gt in ground_truths)
            #print(amount_bboxes)
            # amount_bboxes = {0:3, 1:5}

            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)
                # print(amount_bboxes)
            # amount_bboxes = {0: torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
            
            detections.sort(key=lambda x: x[6], reverse=True) # detections의 confidence가 높은 순으로 정렬
            
            TP = torch.zeros((len(detections))) # detections 개수만큼 1차원 TP tensor를 초기화
            FP = torch.zeros((len(detections))) # 마찬가지로 1차원 FP tensor 초기화
            total_true_bboxes = len(ground_truths) # recall의 TP+FN으로 사용됨
            
            # print(detections)
            for detection_idx, detection in enumerate(detections): # 정렬한 detections를 하나씩 뽑음
                # ground_truth_img : detection과 같은 이미지의 ground truth bbox들을 가져옴
                ground_truth_boxes = [bbox for bbox in ground_truths if bbox[0] == detection[0]]         
                best_iou = 0 # 초기화

                for idx, gt in enumerate(ground_truth_boxes): # 현재 detection box를 이미지의 ground truth들과 비교
                    iou = self.intersection_over_union(detection[1:5], gt[1:5])

                    if iou > best_iou: #ground truth들과의 iou중 가장 높은놈의 iou를 저장
                        best_iou = iou
                        best_gt_idx = idx # 인덱스도 저장
                if best_iou > iou_threshold: # 그 iou가 threshold 이상이면 헤당 인덱스에 TP = 1 저장, 이하면 FP = 1 저장 
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                        if c in ["car", "truck", "pedestrian", "cyclist"]:
                            # self.rd_x_e.append((detection[0], ground_truth_boxes[best_gt_idx][7], detection[7]))
                            # self.rd_y_e.append((ground_truth_boxes[best_gt_idx][8], detection[8]))
                            if ground_truth_boxes[best_gt_idx][7] < 43 and ground_truth_boxes[best_gt_idx][8] < 15:
                                self.rd_x_e.append(abs(ground_truth_boxes[best_gt_idx][7] - detection[7]))
                                self.rd_y_e.append(abs(ground_truth_boxes[best_gt_idx][8] - detection[8]))
                    else:
                        FP[detection_idx] = 1 # 이미 해당 물체를 detect한 물체가 있다면 즉 인덱스 자리에 이미 TP가 1이라면 FP=1적용
                else:
                    FP[detection_idx] = 1
            
            # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]    
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon)) # TP_cumsum + FP_cumsum을 하면 1씩 증가하게됨
            
            recalls = torch.cat((torch.tensor([0]), recalls)) # x축의 시작은 0 이므로 맨앞에 0추가
            precisions = torch.cat((torch.tensor([1]), precisions)) # y축의 시작은 1 이므로 맨앞에 1 추가
            AP.append(torch.trapz(precisions, recalls)) # 현재 클래스에 대해 AP를 계산해줌
        
        return sum(AP) / len(AP) # MAP