import cv2 as cv
import numpy as np
import math

def generate_op(size):
	a = []
	for i in range(-size//2+1, size//2+1):
		for j in range(-size//2+1, size//2+1):
			a.append((i, j))
	return a

def padding(img, expand):
	width, height = img.shape[:2]
	new_img = np.zeros((width+2, height+2))
	for i in range(width):
		for j in range(height):
			new_img[i+1,j+1] = img[i,j]
	new_img[0,0] = img[0,0]
	new_img[width+1, 0] = img[width-1, 0]
	new_img[width+1, width+1] = img[width-1, width-1]
	new_img[0, width+1] = img[0, width-1]
	for i in range(1, width+2):
		new_img[0, i] = new_img[1, i]
		new_img[width+1, i] = new_img[width, i]
		new_img[i, 0] = new_img[i, 1]
		new_img[i, width+1] = new_img[i, width]
	if expand == 1:
		return new_img
	else:
		return padding(new_img, expand-1)

def laplacian_mask(img, threshold, mask, size):
	width, height = img.shape[:2]
	new_img = np.zeros((width, height))
	op = generate_op(size)
	pad_img = padding(img, size//2)
	for i in range(size//2, width+size//2):
		for j in range(size//2, height+size//2):
			grad = 0
			for x, y in op:
				grad += mask[x+size//2][y+size//2]*pad_img[i+x][j+y]
			if grad >= threshold:
				new_img[i-size//2][j-size//2] = 1
			elif grad <= -threshold:
				new_img[i-size//2][j-size//2] = -1
			else:
				new_img[i-size//2][j-size//2] = 0
	return new_img


def zero_crossing(img):
	width, height = img.shape[:2]
	new_img = np.zeros((width, height))
	op = generate_op(3)
	pad_img = padding(img, 1)
	for i in range(1, width+1):
		for j in range(1, height+1):
			mask_pixel = pad_img[i][j]
			cross = 0
			if mask_pixel >= 1:
				for x, y in op:
					if pad_img[i+x][j+y] <= -1:
						cross = 1
						break
			if cross == 0:
				new_img[i-1][j-1] = 255
			else:
				new_img[i-1][j-1] = 0
	return new_img


if __name__ == "__main__":
	img = cv.imread("lena.bmp", 0)
	lap_1_mask = [(0, 1, 0), (1, -4, 1), (0, 1, 0)]
	lap_2_mask = (1/3)*np.array([(1, 1, 1), (1, -8, 1), (1, 1, 1)])
	min_var_lap_mask = (1/3)*np.array([(2, -1, 2), (-1, -4, -1), (2, -1, 2)])
	LoG_mask = [(0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0),
				(0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0),
				(0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0),
				(-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1),
				(-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1),
				(-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2),
				(-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1),
				(-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1),
				(0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0),
				(0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0),
				(0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0)]
	DoG_mask = [(-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1),
				(-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3),
				(-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4),
				(-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6),
				(-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7),
				(-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8),
				(-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7),
				(-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6),
				(-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4),
				(-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3),
				(-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1)]
	lap_1 = laplacian_mask(img, 15, lap_1_mask, 3)
	lap_2 = laplacian_mask(img, 15, lap_2_mask, 3)
	min_var_lap = laplacian_mask(img, 20, min_var_lap_mask, 3)
	LoG = laplacian_mask(img, 3000, LoG_mask, 11)
	DoG = laplacian_mask(img, 1, DoG_mask, 11)
	zc_lap_1 = zero_crossing(lap_1)
	zc_lap_2 = zero_crossing(lap_2)
	zc_min_var_lap = zero_crossing(min_var_lap)
	zc_LoG = zero_crossing(LoG)
	zc_DoG = zero_crossing(DoG)
	cv.imwrite("zero_crossing_laplace_1.bmp", zc_lap_1)
	cv.imwrite("zero_crossing_laplace_2.bmp", zc_lap_2)
	cv.imwrite("zero_crossing_minimum_variance_laplace.bmp", zc_min_var_lap)
	cv.imwrite("zero_crossing_Laplacian_of_Gaussian.bmp", zc_LoG)
	cv.imwrite("zero_crossing_Difference_of_Gaussian.bmp", zc_DoG)

