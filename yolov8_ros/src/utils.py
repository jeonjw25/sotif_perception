import cv2
import numpy as np

def warp(img, src, dst):
    img_size = (img.shape[1], img.shape[0]) # w h
    x_r = img_size[0]/1920        # 1920: bag파일 이미지 크기 가로
    y_r = img_size[1]/1080
    x = 1000*x_r
    y = 800*y_r
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (int(x),int(y)), flags=cv2.INTER_LINEAR)
    
    return warped

def color_filter(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])

    yellow_lower = np.uint8([10, 0, 100])
    yellow_upper = np.uint8([40, 255, 255])

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, lower, upper)

    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(img, img, mask=mask)

    return mask
    

def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    x_r = x/1000        
    y_r = y/800
    _shape = np.array([[0, y-1], [0, 10*y_r], [int(0.3*x), 10*y_r], [int(0.3*x), int(0.3*y)], [int(0.7*x), int(0.3*y)], [int(0.7*x), 10*y_r], [x, 10*y_r], [x, y-1]]) 

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_cnt = img.shape[2]
        ignore_mask_color = (255,) * channel_cnt
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img

def plothistogram(img):
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase

def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    height,width = binary_warped.shape
    x_r = width/1000
    y_r = height/800
    avg_r = (x_r + y_r)/2
    nwindows = int(15 *max(0.5,y_r))   # 15개로 고정 할 것인가, 이미지 크기에 따라 비율을 맞출 것인가
    window_height = np.int(height / nwindows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    
    min_pix_percentage = 30
    margin = int(80*avg_r)
    minpix = int(50*avg_r)

    left_lane = []
    right_lane = []

    color = [0, 255, 0]
    thickness = 2

    l_window_cnt = 0
    r_window_cnt = 0
    lane_quality = 2
    
    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w+1) * window_height # top of window
        win_y_high = binary_warped.shape[0] - w * window_height # bottom of window
        if left_current - margin > 0:
            win_xleft_low = left_current - margin # left top of left window
        else:
            win_xleft_low = 0
        win_xleft_high = left_current + margin # right bottom of left window
        win_xright_low = right_current - margin # left top of right window
        if right_current + margin <= binary_warped.shape[1]:
            win_xright_high = right_current + margin # right bottom of right window
        else:
            win_xright_high = binary_warped.shape[1]

        good_left = ((nonzero_y >=win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >=win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
        
        
        left_lane.append(good_left)
        right_lane.append(good_right)
        

        if len(good_left) > minpix:
            # print(len(good_left))
            left_current = np.int(1.02 * np.mean(nonzero_x[good_left]))
            l_window_cnt += 1
            white_percentage = int((len(good_left) / (window_height * margin * 2)) * 100)

            if white_percentage > min_pix_percentage:
                lane_quality = 1

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
            text_x = (win_xleft_low + win_xleft_high) // 2
            text_y = (win_y_low + win_y_high) // 2
            text = str(white_percentage) + "%"
            cv2.putText(out_img, text,
                            (int(text_x-10), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if len(good_right) > minpix:
            right_current = np.int(1.02 * np.mean(nonzero_x[good_right]))
            r_window_cnt += 1
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
            white_percentage = int((len(good_right) / (window_height * margin * 2)) * 100)

            if white_percentage > min_pix_percentage:
                lane_quality = 1

            text_x = (win_xright_low + win_xright_high) // 2
            text_y = (win_y_low + win_y_high) // 2
            text = str(white_percentage) + "%"
            cv2.putText(out_img, text,
                            (int(text_x-10), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Estimate lane quality
        if win_xleft_high > win_xright_low:
            lane_quality = 0

        
    if l_window_cnt <= 2 or r_window_cnt <= 2:
        lane_quality = 1

    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]
    m_leftx, m_lefty, m_rightx, m_righty = convert_pixel2meter(binary_warped, leftx, lefty, rightx, righty)

    if len(m_leftx) == 0 and len(m_rightx) > 0: # turning left
        lane_quality = 1
        right_fit = np.polyfit(m_righty, m_rightx, 3)
        left_fit = right_fit
        left_fit[-1] = 0 - left_fit[-1]
        # print("left turning!")
    elif len(m_leftx) > 0 and len(m_rightx) == 0: # turning right
        lane_quality = 1
        left_fit = np.polyfit(m_lefty, m_leftx, 3)
        right_fit = left_fit
        right_fit[-1] = 0 - right_fit[-1]
        # print("right turning!")
    elif len(m_leftx) == 0 and len(m_rightx) == 0:
        lane_quality = 0
        print("No lane detected!")
    else:
        right_fit = np.polyfit(m_righty, m_rightx, 3)
        left_fit = np.polyfit(m_lefty, m_leftx, 3)
    text = "lane quality: " + str(lane_quality)
    cv2.putText(out_img, text, (int(50), int(50)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out_img, left_fit, right_fit, lane_quality

def convert_pixel2meter(img, leftx, lefty, rightx, righty):
    ym_per_pix = 12.3 / img.shape[0]
    xm_per_pix = 4 / img.shape[1]
    left_x_mid = np.full((len(leftx)), int(img.shape[1] / 2))
    right_x_mid = int(img.shape[1] / 2)
    m_leftx = (leftx - left_x_mid) * xm_per_pix
    m_rightx = (rightx - right_x_mid) * xm_per_pix
    m_lefty = lefty * ym_per_pix
    m_righty = righty * ym_per_pix


    return m_leftx, m_lefty, m_rightx, m_righty

    
def calculate_curvature(image):
    # 이미지 크기 가져오기
    height, width = image.shape[:2]

    # 이미지 크기 가져오기
    height, width = image.shape[:2]
    x_r = width/1920        # 1920: bag파일 이미지 크기 가로
    y_r = height/1080       # 1080: bag파일 이미지 크기 세로

    blur_img = cv2.GaussianBlur(image, (3, 3), 0)
    
    
    # 관심영역(ROI) 생성
    region_of_interest_vertices = np.array([[(0, height), (0, height-100), (width, height-100), (width, height)]], dtype='uint32')

    src = np.float32([[150*x_r, height], [750*x_r, 700*y_r], [1200*x_r, 700*y_r], [width-(150*x_r), height]])
    dst = np.float32([[100*x_r, 750*y_r], [170*x_r, 50*y_r], [750*x_r, 50*y_r], [750*x_r, 750*y_r]])

    warped = warp(blur_img, src, dst) # birdeye view transform
    w_f_img = color_filter(warped) # filter yellow & white
    masked_w_f_img = roi(w_f_img) # filter middle area of lane

    # canny_img = cv2.Canny(warped, 70, 210)

    leftbase, rightbase = plothistogram(masked_w_f_img)
    out_img, left_fit, right_fit , lane_quality = slide_window_search(masked_w_f_img, leftbase, rightbase)

    lp = np.poly1d(left_fit)
    rp = np.poly1d(right_fit)

    # print("left_curvation: {}" .format(left_fit))
    # print("right_curvation: {}" .format(right_fit))

    return out_img, lp, rp, lane_quality
    
# def crop(org_img, boxes):
#     img = org_img.copy()
#     # print(img.shape)
#     for box in boxes:
#         xmin = np.int64(box[0])
#         ymin = np.int64(box[1])
#         xmax = np.int64(box[2])
#         ymax = np.int64(box[3])
#         img[ymin:ymax, xmin:xmax] = 0
        

#     return img