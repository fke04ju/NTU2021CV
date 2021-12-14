import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def make_histogram(image, width, height):
	plt.figure()
	X = [i for i in range(256)]
	Y = np.zeros(256)
	for i in range(width):
		for j in range(height):
			Y[image[i,j]] += 1
	plt.bar(X, Y)
	plt.savefig('Histogram.jpg')
	return image

def make_histogram_divide3(image, width, height):
	plt.figure()
	image_copy = image.copy()
	X = [i for i in range(256)]
	Y = np.zeros(256)
	for i in range(width):
		for j in range(height):
			image_copy[i,j] = int(image[i,j] / 3)
			Y[image_copy[i,j]] += 1
	plt.bar(X, Y)
	plt.savefig('New_Histogram.jpg')
	return image_copy

"""
s_0 = 255*(n_0)/area
s_1 = 255*(n_0+n_1)/area
s_2 = 255*(n_0+n_1+n_2)/area
"""
def histogram_equalization(image, width, height):
	plt.figure()
	image_copy = image.copy()
	X = [i for i in range(256)]
	Y = np.zeros(256)
	new_Y = np.zeros(256)
	s = np.zeros(256)
	for i in range(width):
		for j in range(height):
			Y[image[i,j]] += 1
	s[0] = Y[0]
	area = width*height
	for i in range(1,256):
		s[i] = s[i-1]+Y[i]
	for i in range(256):
		s[i] = 255*(s[i]/area)
	for i in range(width):
		for j in range(height):
			image_copy[i,j] = s[image_copy[i,j]]
			new_Y[image_copy[i,j]] += 1
	plt.bar(X,new_Y)
	plt.savefig('Histogram_equalization.jpg')
	return image_copy		


if __name__ == '__main__':
	img = cv.imread('lena.bmp', 0)
	width = img.shape[0]
	height = img.shape[1]
	img = make_histogram(img, width, height)
	new_img = make_histogram_divide3(img, width, height)
	cv.imwrite("lena_divide_by_3.bmp", new_img)
	new_img = histogram_equalization(new_img, width, height)
	cv.imwrite("lena_histogram_equalization.bmp", new_img)