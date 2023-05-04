import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class calRelativeVal():
    def __init__(self, img, bbox):
        self.img = img
        self.bbox = bbox

        self.img_left1 = [376, 739]
        self.img_left2 = [563, 674]
        self.img_left3 = [658, 642]

        self.img_mid1 = [1261, 735]
        self.img_mid2 = [1159, 676]
        self.img_mid3 = [1110, 641]

        self.img_right1 = [1664, 739]
        self.img_right2 = [1438, 674]
        self.img_right3 = [1320, 641]

        self.obj_left1 = [12.6, 3.5, 0]
        self.obj_left2 = [18.6, 3.5, 0]
        self.obj_left3 = [24.6, 3.5, 0]

        self.obj_mid1 = [12.6, -1.75, 0]
        self.obj_mid2 = [18.6, -1.75, 0]
        self.obj_mid3 = [24.6, -1.75, 0]

        self.obj_right1 = [12.6, -4.25, 0]
        self.obj_right2 = [18.6, -4.25, 0]
        self.obj_right3 = [24.6, -4.25, 0]

        self.img_points = np.array([self.img_left1, self.img_left2, self.img_left3, 
                                    self.img_mid1, self.img_mid2, self.img_mid3, 
                                    self.img_right1, self.img_right2, self.img_right3], dtype=np.float32)
        self.obj_points = np.array([self.obj_left1, self.obj_left2, self.obj_left3, 
                                    self.obj_mid1, self.obj_mid2, self.obj_mid3, 
                                    self.obj_right1, self.obj_right2, self.obj_right3], dtype=np.float32)
        
        self.H, self._ = cv2.findHomography(self.img_points, self.obj_points)
    
    def calculate_3d_coord(self):
        x_coor = float((self.bbox[0] + self.bbox[2]) / 2)
        y_coor = float(self.bbox[3])

        img_coord = np.array([x_coor, y_coor], dtype=np.float32)
        reshaped_coor = np.append(img_coord, np.ones([1])).T
        estimated = np.dot(self.H, reshaped_coor)
        x, y, z = estimated

        return x, y, z
    # appned_image_points = np.append(img_points.reshape(9, 2), np.ones([1, 9]).T, axis=1)
    # for image_point in appned_image_points:
    # # estimation point(object_point) -> homography * src(image_point)
    #     estimation_distance = np.dot(H, image_point)
    #     x, y, z = estimation_distance
    #     print("x: {}, y: {}".format(round(x/z, 2), round(y/z, 2)))
    # # img = cv2.imread("homography.png")
    # # plt.imshow(img)
    # # plt.show()