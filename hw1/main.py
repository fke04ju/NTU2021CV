import cv2 as cv
import numpy as np


img = cv.imread("lena.bmp",0)
row, col = img.shape[:2]

center_row = int(row/2)
for i in range(center_row):
    for j in range(col):
        px = img[i,j]
        img[i,j] = img[row-i-1,j]
        img[row-i-1,j] = px

cv.imwrite("upside_down_lena.bmp",img)

img = cv.imread("lena.bmp",0)
row, col = img.shape[:2]
center_col = int(col/2)
for i in range(row):
    for j in range(center_col):
        px = img[i,j]
        img[i,j] = img[i,col-j-1]
        img[i,col-j-1] = px

cv.imwrite("right_side_left_lena.bmp",img)

img = cv.imread("lena.bmp",0)
row, col = img.shape[:2]
for i in range(row):
    for j in range(i, col):
        px = img[i,j]
        img[i,j] = img[j,i]
        img[j,i] = px

cv.imwrite("diagonally_lena.bmp",img)

img = cv.imread("lena.bmp")
row, col = img.shape[:2]
M = cv.getRotationMatrix2D((col/2, row/2), 45, 1)
dst = cv.warpAffine(img, M, (col, row))
cv.imwrite("rotate_lena.bmp", dst)

img = cv.imread("lena.bmp",0)
row, col = img.shape[:2]
scale = 0.5
dst = cv.resize(img, ((int(col*scale), int(row*scale))))
cv.imwrite("shrink_lena.bmp", dst)

img = cv.imread("lena.bmp")
row, col = img.shape[:2]
for i in range(row):
    for j in range(col):
        px = img[i,j]
        if px[0] < 128:
            px[0] = 0
        else:
            px[0] = 255
        if px[1] < 128:
            px[1] = 0
        else:
            px[1] = 255
        if px[2] < 128:
            px[2] = 0
        else:
            px[2] = 255

cv.imwrite("binary_lena.bmp", img) 
