import numpy as np
import cv2
import matplotlib.pyplot as plt

# get region
def get_region(image, width, height, label):
	x_list = []
	for i in range(width):
		for j in range(height):
			if(image[i,j] == label):
				x.list.append((i,j))
	return x_list[0], x_list[-1]

# show image
def show_image(image):
	cv2.imshow('demo', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# save image
def save_image(path_name, image):
	cv2.imwrite(path_name+'.jpg', image)

# (A) binary image
def binary(image, width, height, TF):
	img_res = image.copy()
	for i in range(width):
		for j in range(height):
			if(img_res[i,j] < 128):
				img_res[i,j] = 0
			else:
				img_res[i,j] = 255
			
	if (TF == True):
		save_image('Binary_lena', img_res)
	return img_res

# (B) Histogram
def histogram(image, width, height):
	X = [i for i in range(256)]
	Y = np.zeros(256)
	for i in range(width):
		for j in range(height):
			Y[image[i,j]] += 1
	plt.bar(X,Y)
	plt.savefig('Histogram.jpg')

# (C) connected components
def connect_component(image, width, height):
	img = cv2.imread('lena.bmp')
	width = img.shape[0]
	height = img.shape[1]
	img_seg = (image==255)-1
	queue = []
	border = []
	idx = 0
	for i in range(width):
		for j in range(height):
			if img_seg[i,j] == 0:
				idx += 1
				queue.append((i,j))
				x_min = img_seg.shape[0]
				y_min = img_seg.shape[1]
				x_max, y_max, x_sum, y_sum, count = 0, 0, 0, 0, 0
				num = 1
				while len(queue) != 0:
					x, y = queue.pop()
					x_min = x if x < x_min else x_min
					y_min = y if y < y_min else y_min
					x_max = x if x > x_max else x_max
					y_max = y if y > y_max else y_max
					x_sum += x
					y_sum += y 
					count += 1
					num += 1
					img_seg[x,y] = idx
					for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
						if (x+dx >= 0) and (x+dx < width) and (y+dy >= 0) and (y+dy < height) and img_seg[x+dx, y+dy] == 0:
							queue.append((x+dx, y+dy))

				if count >= 500:
					border.append((x_min, x_max, y_min, y_max, x_sum//count, y_sum//count, num//count))

	for i in range(width):
		for j in range(height):
			if(img[i,j,0] < 128):
				img[i,j,0] = 0
			else:
				img[i,j,0] = 255
			if(img[i,j,1] < 128):
				img[i,j,1] = 0
			else:
				img[i,j,1] = 255
			if(img[i,j,2] < 128):
				img[i,j,2] = 0
			else:
				img[i,j,2] = 255

	for a in border:
		# print(a)
		cv2.rectangle(img, (a[2], a[0]), (a[3], a[1]), (255,0,0), 2)
		centroid_x = int(a[5]/a[6])
		centroid_y = int(a[4]/a[6])
		cv2.line(img, (centroid_x-10, centroid_y), (centroid_x+10, centroid_y),(0,0,255),2)
		cv2.line(img, (centroid_x, centroid_y-10), (centroid_x, centroid_y+10),(0,0,255),2)

	save_image('Connect_image', img)


if __name__ == '__main__':
	img = cv2.imread('lena.bmp',0)
	width = img.shape[0]
	height = img.shape[1]
	img_binary = binary(img, width, height, True)
	img_histogram = histogram(img, width, height)
	new_binary = binary(img, width, height, False)
	connect_component(new_binary, width, height)