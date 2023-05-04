import cv2
import numpy as np

def classify_traffic_light(img_rgb, box):
        xmin = int(box[0])
        xmax = int(box[2])
        ymin = int(box[1])
        ymax = int(box[3])
        cropped_img = img_rgb[ymin:ymax, xmin:xmax]

        # 빨간색 범위
        lower_red = (0, 0, 155)
        upper_red = (160, 160, 255)

        # 노란색 범위
        lower_yellow = (0, 155, 155)
        upper_yellow = (180, 255, 255)

        # 초록색 범위
        lower_green = (0, 150, 0)
        upper_green = (180, 255, 180)

        # 색상 범위에 해당하는 마스크 생성
        mask_red = cv2.inRange(cropped_img, lower_red, upper_red)
        mask_yellow = cv2.inRange(cropped_img, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(cropped_img, lower_green, upper_green)
        
        rows, cols = np.where(mask_green != 0)
        x_min, y_min = np.min(rows), np.min(cols)
        x_max, y_max = np.max(rows), np.max(cols)
        mask_green_area = mask_green[x_min:x_max, y_min:y_max]

        # red_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_red)
        # yellow_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_yellow)
        #green_area = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_green)
        # cv2.imwrite('/root/catkin_ws/src/Yolov5_ros/yolov5_ros/yolov5_ros/yolov5_ros/media/mask_green'+'.png', green_area)
        
        
        
        max_area = max(cv2.countNonZero(mask_red), cv2.countNonZero(mask_yellow), cv2.countNonZero(mask_green))
        if max_area == cv2.countNonZero(mask_red):
            Class = 'red'
        elif max_area == cv2.countNonZero(mask_yellow):
            Class = 'yellow'
        elif max_area == cv2.countNonZero(mask_green):
            width, height, _ = img_rgb.shape
            center_x = int(width/2)
            

            # Get the column overlap for the left and right sides of the shape
            left_overlap = get_column_overlap(mask_green_area[:, :center_x])
            right_overlap = get_column_overlap(mask_green_area[:, center_x:])

            # Determine if the shape is an arrow or a circle based on the column overlap
            if min(left_overlap[2], right_overlap[-3]) < 1.5 * max(left_overlap[-1], right_overlap[0]):
                if sum(left_overlap) > sum(right_overlap):
                     Class = 'left'
                else:
                    Class = 'right'
            else:
                Class = 'green' 

        
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