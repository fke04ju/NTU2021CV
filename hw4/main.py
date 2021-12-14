import cv2 as cv
import numpy as np

def binary(img):
	width, height = img.shape[:2]
	for i in range(width):
		for j in range(height):
			px = img[i,j]
			if px < 128:
				img[i,j] = 0
			else:
				img[i,j] = 255
	return img

def dilation(img, kernel):
	width, height = img.shape[:2]
	new_img = np.zeros((width,height))
	for i in range(width):
		for j in range(height):
			if img[i,j] == 255:
				for point in kernel:
					tmp_i = i+point[0]
					tmp_j = j+point[1]
					if tmp_i >= 0 and tmp_i < width and tmp_j >= 0 and tmp_j < height:
						new_img[tmp_i,tmp_j] = 255
	return new_img

def erosion(img, kernel):
	width, height = img.shape[:2]
	new_img = np.zeros((width,height))
	for i in range(width):
		for j in range(height):
			new_img[i,j] = img[i,j]
	for i in range(width):
		for j in range(height):
			flag = 0
			for point in kernel:
				tmp_i = i+point[0]
				tmp_j = j+point[1]
				if tmp_i >= 0 and tmp_i < width and tmp_j >= 0 and tmp_j < height:
					if img[tmp_i,tmp_j] == 0:
						flag = 1
						break
				else:
					flag = 1
					break
			if flag == 1:
				new_img[i,j] = 0
			else:
				new_img[i,j] = 255
	return new_img

def opening(img, kernel):
	ero_img = erosion(img,kernel)
	open_img = dilation(ero_img, kernel)
	return open_img

def closing(img, kernel):
	dil_img = dilation(img, kernel)
	clo_img = erosion(dil_img, kernel)
	return clo_img

def complement(img):
	width, height = img.shape[:2]
	new_img = np.zeros((width,height))
	for i in range(width):
		for j in range(height):
			new_img[i,j] = 255 - img[i,j]
	return new_img

def intersection(img1, img2):
	width, height = img1.shape[:2]
	new_img = np.zeros((width,height))
	for i in range(width):
		for j in range(height):
			if img1[i,j] == 255 and img2[i,j] == 255:
				new_img[i,j] = 255
			else:
				new_img[i,j] = 0
	return new_img

def hit_and_miss(img, ker_j, ker_k):
	comp_img = complement(img)
	hit = erosion(img, ker_j)
	miss = erosion(comp_img, ker_k)
	return intersection(hit, miss)

def q1(img, kernel, output):
	bin_img = binary(img)
	dil_img = dilation(bin_img, kernel)
	cv.imwrite(output, dil_img)

def q2(img, kernel, output):
	bin_img = binary(img)
	ero_img = erosion(bin_img, kernel)
	cv.imwrite(output, ero_img)

def q3(img, kernel, output):
	bin_img = binary(img)
	open_img = opening(bin_img, kernel)
	cv.imwrite(output, open_img)

def q4(img, kernel, output):
	bin_img = binary(img)
	clo_img = closing(bin_img, kernel)
	cv.imwrite(output, clo_img)

def q5(img, ker_j, ker_k, output):
	bin_img = binary(img)
	hit_and_miss_img = hit_and_miss(bin_img, ker_j, ker_k)
	cv.imwrite(output, hit_and_miss_img)



if __name__ == '__main__':
	kernel = [(-1,2),(0,2),(1,2),(-2,1),(-1,1),(0,1),(1,1),(2,1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(-1,-2),(0,-2),(1,-2)]
	ker_j = [(0,-1),(0,0),(1,0)]
	ker_k = [(-1,0),(-1,1),(0,1)]
	img = cv.imread("lena.bmp",0)
	q1(img, kernel, "dilation_lena.bmp")
	q2(img, kernel, "erosion_lena.bmp")
	q3(img, kernel, "opening_lena.bmp")
	q4(img, kernel, "closing_lena.bmp")
	q5(img, ker_j, ker_k, "hit_and_miss_lena.bmp")
