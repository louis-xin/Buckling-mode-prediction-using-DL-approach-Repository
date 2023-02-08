import cv2
import numpy as np
import os

#%% Input merging
# 사이즈 조절 W, H
size = (512,512)

data_dir = './/DL_Data'

if not os.path.exists(data_dir + '\\Input Image'):
    os.makedirs(data_dir + '\\Input Image')
    
if not os.path.exists(data_dir + '\\Output Image'):
    os.makedirs(data_dir + '\\Output Image')

if not os.path.exists(data_dir + '\\Image'):
    os.makedirs(data_dir + '\\Image')

#%% Input Image
for i in range(520):
    leftimg = cv2.imread(data_dir + '\\Imperfection\\Imperfection ({}).png'.format(i+1))
    rightimg = cv2.imread(data_dir + '\\Dimensions\\Dim_{}.png'.format(i+1))

    # 사이즈 조정
    leftimg = cv2.resize(leftimg, size)
    rightimg = cv2.resize(rightimg, size)
    
    # 이미지 좌우 합치기
    add_img = np.hstack((leftimg, rightimg))
    cv2.imwrite(data_dir + '//Input Image//Input_Img_{}.png'.format(i),add_img)
    
#%% Output Merging

# for i in range(520):
#     leftimg = cv2.imread(data_dir + '\\Buckling Mode\\BucklingMode ({}).png'.format(i+1))
#     rightimg = cv2.imread(data_dir + '\\L-D plot\\L-D plot ({}).png'.format(i+1))

#     # 사이즈 조정
#     leftimg = cv2.resize(leftimg, size)
#     rightimg = cv2.resize(rightimg, size)
    
#     # 이미지 좌우 합치기
#     add_img = np.hstack((leftimg, rightimg))
#     cv2.imwrite(data_dir + '//Output Image//Output_Img_{}.png'.format(i),add_img)
    
#%% Whole Image

size2 = (1024,512)
for i in range(520):
    leftimg = cv2.imread(data_dir + '\\Buckling_Mode\\BucklingMode ({}).png'.format(i+1))
    rightimg = cv2.imread(data_dir + '\\Input Image\\Input_Img_{}.png'.format(i))

    # 사이즈 조정
    leftimg = cv2.resize(leftimg, size)
    rightimg = cv2.resize(rightimg, size2)
    
    # 이미지 좌우 합치기
    add_img = np.hstack((leftimg, rightimg))
    cv2.imwrite(data_dir + '//Image//Img_{}.png'.format(i),add_img)
    








