import cv2
import numpy as np


def classify_traffic_light(img_bgr, box):
        # hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        xmin = int(box[0])
        xmax = int(box[2])
        ymin = int(box[1])
        ymax = int(box[3])
        cropped_img = img_bgr[ymin:ymax, xmin:xmax]
        # cv2.imshow("im0", cropped_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 빨간색 범위 BGR
        lower_red = (0, 0, 100)
        upper_red = (100, 100, 255)

        # 노란색 범위
        lower_yellow = (0, 150, 150)
        upper_yellow = (20, 255, 255)

        # 초록색 범위
        lower_green = (150, 190, 0)
        upper_green = (200, 255, 20)

        # 색상 범위에 해당하는 마스크 생성
        mask_red = cv2.inRange(cropped_img, lower_red, upper_red)
        mask_yellow = cv2.inRange(cropped_img, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(cropped_img, lower_green, upper_green)
        
        red_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_red)
        # yellow_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_yellow)
        #green_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_green)
        
        max_area = max(cv2.countNonZero(mask_red), cv2.countNonZero(mask_yellow), cv2.countNonZero(mask_green))
        xmax = xmax - xmin
        xmin = 0
        center_x = xmax // 2

        Class = 'traffic_light'
        if (xmax - xmin) >= 20:
            if max_area == cv2.countNonZero(mask_red):
                Class = 'red'
                
            elif max_area == cv2.countNonZero(mask_yellow):
                Class = 'yellow'
                
            elif max_area == cv2.countNonZero(mask_green):
                # print(max_area)
                Class = 'green'
                left_pixels = cv2.countNonZero(mask_green[:, xmin:center_x])
                right_pixels = cv2.countNonZero(mask_green[:, center_x:xmax])
                
                print("left: " + str(left_pixels))
                print("right: " + str(right_pixels))


                if left_pixels / right_pixels >= 1.4:
                    Class = 'left'
                elif right_pixels / left_pixels >= 1.4:
                    Class = 'right'
     
        return Class
    
def get_column_overlap(img):
        # Draw 9 vertical lines on the blue area
    height, width = img.shape[:2]
    step = int(width / 5)
    overlap = []
    for i in range(5):
        x = i * step
        overlap.append(cv2.countNonZero(img[:, x:x+step]))
    
    return overlap