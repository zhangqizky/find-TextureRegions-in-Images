__authur__ = 'tangxi.zq'
__time__ = '2019-04-27'

import numpy as np
import cv2
import os
import time

'''
找寻图像中纹理丰富的区域
'''


def calculateGravity(points):
    '''
    找一堆点集的重心
    '''
    x_center = 0
    y_center = 0
    x_sum = 0
    y_sum = 0
    length = len(points)
    for each in points:
        x_sum += each[0]
        y_sum += each[1]
    x_center = x_sum / length
    y_center = y_sum / length
    return [x_center,y_center]

def calculateROI(image,points):
    '''
    输出特征点密集的区域，即纹理丰富的区域,返回值为一堆矩形框左上角和右下角坐标
    '''
    img = np.zeros(image.shape,np.uint8)
    h_im, w_im,c_im = image.shape
    img.fill(0)
    for point in points:
        cv2.circle(img,(point[0],point[1]),5,(255,255,255),-1)
    cv2.imshow("huabu",img)

    element_dilated = cv2.getStructuringElement(2,(25,25))
    element_eroded = cv2.getStructuringElement(2,(10,10))
    dilated = cv2.dilate(img,element_dilated)
    imRes = cv2.erode(dilated,element_eroded)

    imgray = cv2.cvtColor(imRes, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    res_rect = []
    for contour in contours:
        x,y,w,h= cv2.boundingRect(np.array(contour))
        # rect = cv2.minAreaRect(np.array(contour))
        if w*h > h_im*w_im*0.8 or w*h < h_im*w_im*0.1:continue
        cv2.rectangle(imRes,(x,y),(x+w,y+h),(0,0,255),2)
        res_rect.append([x,y,w,h])
    return res_rect
    # cv2.drawContours(imRes, contours, -1, (0,255,0), 3)
    # cv2.imshow("dilated",imRes)
    # cv2.waitKey()

def main():
    img_list = [os.path.join("/Users/tangxi/Desktop/LFworsesamples/",each ) for each in os.listdir("/Users/tangxi/Desktop/LFworsesamples/") if each.endswith(".jpg") or each.endswith(".png")]
    for each in img_list:
        print(each)
        img = cv2.imread(each)
        h,w,c = img.shape
        start = time.clock()
        sift =cv2.xfeatures2d_SIFT.create()
        kp,des = sift.detectAndCompute(img,None)
        print(len(kp))
        img_withkeypoints = cv2.drawKeypoints(img,kp,np.array([]),(0,255,0),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        # print(kp)
        points2f = cv2.KeyPoint_convert(kp)
        center = calculateGravity(points2f)
        cv2.circle(img_withkeypoints,(int(center[0]),int(center[1])),10,(0,0,255))
        res_rect = calculateROI(img,points2f)
        for rect in res_rect:
            cv2.rectangle(img_withkeypoints,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255),3)
        end = time.clock()
        print("cost time {}".format(end-start))
        cv2.imshow("test",img_withkeypoints)
        c = cv2.waitKey(0)
        if c == 27:
            break

if __name__ == '__main__':
    main()