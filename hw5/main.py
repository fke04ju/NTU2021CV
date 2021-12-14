import cv2 as cv
import numpy as np

def dilation(img, kernel):
	width, height = img.shape[:2]
	new_img = np.zeros((width,height))
	for i in range(width):
		for j in range(height):
			max_px = 0
			for point in kernel:
				tmp_i = i+point[0]
				tmp_j = j+point[1]
				if tmp_i >= 0 and tmp_i < width and tmp_j >= 0 and tmp_j < height:
					max_px = max(max_px, img[tmp_i][tmp_j])
			new_img[i][j] = max_px
	return new_img

def erosion(img, kernel):
	width, height = img.shape[:2]
	new_img = np.zeros((width,height))
	length = len(kernel)
	for i in range(width):
		for j in range(height):
			new_img[i,j] = img[i,j]
	for i in range(width):
		for j in range(height):
			min_px = 255
			flag = 0
			for point in kernel:
				tmp_i = i+point[0]
				tmp_j = j+point[1]
				if tmp_i >= 0 and tmp_i < width and tmp_j >= 0 and tmp_j < height:
					flag += 1
					min_px = min(min_px, img[tmp_i][tmp_j])
			if flag == length:
				new_img[i,j] = min_px
			else:
				new_img[i,j] = 0
	return new_img

def opening(img, kernel):
	ero_img = erosion(img,kernel)
	open_img = dilation(ero_img, kernel)
	return open_img

def closing(img, kernel):
	dil_img = dilation(img, kernel)
	clo_img = erosion(dil_img, kernel)
	return clo_img

def q1(img, kernel, output):
	dil_img = dilation(img, kernel)
	cv.imwrite(output, dil_img)

def q2(img, kernel, output):
	ero_img = erosion(img, kernel)
	cv.imwrite(output, ero_img)

def q3(img, kernel, output):
	open_img = opening(img, kernel)
	cv.imwrite(output, open_img)

def q4(img, kernel, output):
	clo_img = closing(img, kernel)
	cv.imwrite(output, clo_img)


if __name__ == '__main__':
	kernel = [(-1,2),(0,2),(1,2),(-2,1),(-1,1),(0,1),(1,1),(2,1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(-1,-2),(0,-2),(1,-2)]
	img = cv.imread("lena.bmp",0)
	q1(img, kernel, "dilation_lena.bmp")
	q2(img, kernel, "erosion_lena.bmp")
	q3(img, kernel, "opening_lena.bmp")
	q4(img, kernel, "closing_lena.bmp")
