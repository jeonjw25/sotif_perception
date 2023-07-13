import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class calRelativeVal():
    def __init__(self, img, bbox):
        self.img = img
        self.bbox = bbox

        self.img_left1 = [369, 709]
        self.img_left2 = [666, 625]
        self.img_left3 = [765, 595]
        self.img_left4 = [813, 584]

        self.img_mid1 = [759, 709]
        self.img_mid2 = [860, 625]
        self.img_mid3 = [893, 597]
        self.img_mid4 = [910, 583]

        self.img_mid21 = [1148, 708]
        self.img_mid22 = [1054, 626]
        self.img_mid23 = [1024, 596]
        self.img_mid24 = [1007, 583]

        self.img_right1 = [1536, 709]
        self.img_right2 = [1248, 625]
        self.img_right3 = [1155, 596]
        self.img_right4 = [1105, 583]

        self.obj_left1 = [15, 4.25, 0]
        self.obj_left2 = [30, 4.25, 0]
        self.obj_left3 = [45, 4.25, 0]
        self.obj_left4 = [60, 4.25, 0]

        self.obj_mid1 = [15, 1.42, 0]
        self.obj_mid2 = [30, 1.42, 0]
        self.obj_mid3 = [45, 1.42, 0]
        self.obj_mid4 = [60, 1.42, 0]

        self.obj_mid21 = [15, -1.42, 0]
        self.obj_mid22 = [30, -1.42, 0]
        self.obj_mid23 = [45, -1.42, 0]
        self.obj_mid24 = [60, -1.42, 0]

        self.obj_right1 = [15, -4.25, 0]
        self.obj_right2 = [30, -4.25, 0]
        self.obj_right3 = [45, -4.25, 0]
        self.obj_right4 = [60, -4.25, 0]

        self.img_points = np.array([self.img_left1, self.img_left2, self.img_left3, self.img_left4, 
                                    self.img_mid1, self.img_mid2, self.img_mid3, self.img_mid4,
                                    self.img_mid21, self.img_mid22, self.img_mid23, self.img_mid24,
                                    self.img_right1, self.img_right2, self.img_right3, self.img_right4], dtype=np.float32)
        self.obj_points = np.array([self.obj_left1, self.obj_left2, self.obj_left3, self.obj_left4, 
                                    self.obj_mid1, self.obj_mid2, self.obj_mid3, self.obj_mid4,
                                    self.obj_mid21, self.obj_mid22, self.obj_mid23, self.obj_mid24,
                                    self.obj_right1, self.obj_right2, self.obj_right3, self.obj_right4], dtype=np.float32)
        
        self.H, self._ = cv2.findHomography(self.img_points, self.obj_points)

        # self.poly_x = [46.06, 47.61, 54.93]
        # self.poly_y = [50.42, 57.79, 72.7]
        # self.poly = np.polyfit(self.poly_x, self.poly_y, 2)

    def calculate_3d_coord(self):
        x_coor = float((self.bbox[0] + self.bbox[2]) / 2)
        y_coor = float(self.bbox[3])

        img_coord = np.array([x_coor, y_coor], dtype=np.float32)
        reshaped_coor = np.append(img_coord, np.ones([1])).T
        estimated = np.dot(self.H, reshaped_coor)
        x, y, z = estimated
        dist_x = round((x/z),2)
        dist_y = round((y/z),2)
        # if dist_x >= 43:
        #     dist_x = self.poly[0] * (dist_x**2) + self.poly[1]*dist_x + self.poly[2]
        # if dist_y >= 15:
        #     dist_y += 5
        # elif dist_y <= -15:
        #     dist_y -= 5
            
        return dist_x, dist_y
 
