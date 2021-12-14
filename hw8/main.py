from hashlib import new
import cv2 as cv
import numpy as np
import math
import random

filter_3 = [(-1,-1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
filter_5 = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]

def gaussian_noise(img, amptitude):
    new_img = np.copy(img)
    width, height = img.shape[:2]
    for i in range(width):
        for j in range(height):
            gauss_val = img[i,j] + amptitude*np.random.normal(0, 1)
            if gauss_val < 0:
                new_img[i,j] = 0
            elif gauss_val > 255:
                new_img[i,j] = 255
            else:
                new_img[i,j] = int(gauss_val)
    return new_img

def salt_and_pepper(img, threshold):
    width, height = img.shape[:2]
    new_img = np.copy(img)
    for i in range(width):
        for j in range(height):
            rand_val = random.uniform(0, 1)
            if rand_val < threshold:
                new_img[i,j] = 0
            elif rand_val > 1-threshold:
                new_img[i,j] = 255
            else:
                new_img[i,j] = img[i,j]
    return new_img

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
    if expand == 3:
        return new_img
    elif expand == 5:
        return padding(new_img, 3)


def box_filter(img, filter_size):
    width, height = img.shape[:2]
    new_img = np.zeros((width, height))
    if filter_size == 3:
        filter_kernel = filter_3
        padding_img = padding(img, 3)
        for i in range(1, width+1):
            for j in range(1, height+1):
                # cal mean
                mean = 0.0
                N = 9
                for x, y in filter_kernel:
                    mean += padding_img[i+x, j+y]
                mean /= N
                new_img[i-1,j-1] = int(mean)
    if filter_size == 5:
        filter_kernel = filter_5
        padding_img = padding(img, 5)
        for i in range(2, width+2):
            for j in range(2, height+2):
                # cal mean
                mean = 0.0
                N = 25
                for x, y in filter_kernel:
                    mean += padding_img[i+x, j+y]
                mean /= N
                new_img[i-2,j-2] = mean
    return new_img

def median_filter(img, filter_size):
    width, height = img.shape[:2]
    new_img = np.zeros((width, height))
    if filter_size == 3:
        filter_kernel = filter_3
        padding_img = padding(img, 3)
        for i in range(1, width+1):
            for j in range(1, height+1):
                # cal median
                pixel_list=[]
                for x, y in filter_kernel:
                        pixel_list.append(padding_img[i+x, j+y])
                pixel_list.sort()
                idx = int(len(pixel_list)/2)
                new_img[i-1,j-1] = pixel_list[idx]
    if filter_size == 5:
        filter_kernel = filter_5
        padding_img = padding(img, 5)
        for i in range(2, width+2):
            for j in range(2, height+2):
                # cal median
                pixel_list=[]
                for x, y in filter_kernel:
                        pixel_list.append(padding_img[i+x, j+y])
                pixel_list.sort()
                idx = int(len(pixel_list)/2)
                new_img[i-2,j-2] = pixel_list[idx]
    return new_img

def mu_s(img):
    width, height = img.shape[:2]
    N = width * height
    mean = 0.0
    for i in range(width):
        for j in range(height):
            mean += img[i,j]
    return mean/N

def VS(img):
    width, height = img.shape[:2]
    N = width * height
    vs = 0.0
    mean = mu_s(img)
    for i in range(width):
        for j in range(height):
            vs += (img[i,j] - mean) ** 2
    return vs/N

def mu_noise(img, noise_img):
    width, height = img.shape[:2]
    N = width * height
    mn = 0.0
    for i in range(width):
        for j in range(height):
            mn += (noise_img[i,j] - img[i,j])
    return mn/N

def VN(img, noise_img):
    width, height = img.shape[:2]
    N = width * height
    vn = 0.0
    mn = mu_noise(img, noise_img)
    for i in range(width):
        for j in range(height):
            vn += ((noise_img[i,j] - img[i,j]) - mn) ** 2
    return vn/N

def normalize(img):
    width, height = img.shape[:2]
    new_img = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            new_img[i, j] = img[i, j]/255*1
    return new_img

def SNR(img, compare_img):
    nor_img = normalize(img)
    nor_cmp = normalize(compare_img)
    return 20 * math.log((math.sqrt(VS(nor_img)) / math.sqrt(VN(nor_img, nor_cmp))), 10)

if __name__ == '__main__':
    img = cv.imread("lena.bmp",0)

# (a) gaussian noise with amptitude of 10 and 30
    gau_10 = gaussian_noise(img, 10)
    gau_30 = gaussian_noise(img, 30)
    cv.imwrite("gaussian_noise_amptitude_10.bmp", gau_10)
    cv.imwrite("gaussian_noise_amptitude_30.bmp", gau_30)

# (b) salt and pepper with probability 0.1 and 0.05
    sap_1 = salt_and_pepper(img, 0.1)
    sap_2 = salt_and_pepper(img, 0.05)
    cv.imwrite("salt_and_pepper_01.bmp", sap_1)
    cv.imwrite("salt_and_pepper_005.bmp", sap_2)

# (c) 3x3 5x5 box filter on (a)(b)
    gau_10_box_3 = box_filter(gau_10, 3)
    gau_30_box_3 = box_filter(gau_30, 3)
    gau_10_box_5 = box_filter(gau_10, 5)
    gau_30_box_5 = box_filter(gau_30, 5)
    cv.imwrite("gau10_with_box_filter_3x3.bmp", gau_10_box_3)
    cv.imwrite("gau30_with_box_filter_3x3.bmp", gau_10_box_3)
    cv.imwrite("gau10_with_box_filter_5x5.bmp", gau_10_box_5)
    cv.imwrite("gau30_with_box_filter_5x5.bmp", gau_10_box_5)

    sap_1_box_3 = box_filter(sap_1, 3)
    sap_2_box_3 = box_filter(sap_2, 3)
    sap_1_box_5 = box_filter(sap_1, 5)
    sap_2_box_5 = box_filter(sap_2, 5)
    cv.imwrite("sap1_with_box_filter_3x3.bmp", sap_1_box_3)
    cv.imwrite("sap2_with_box_filter_3x3.bmp", sap_2_box_3)
    cv.imwrite("sap1_with_box_filter_5x5.bmp", sap_1_box_5)
    cv.imwrite("sap2_with_box_filter_5x5.bmp", sap_2_box_5)
    
# (d) 3x3 5x5 median filter on (a)(b)
    gau_10_median_3 = median_filter(gau_10, 3)
    gau_30_median_3 = median_filter(gau_30, 3)
    gau_10_median_5 = median_filter(gau_10, 5)
    gau_30_median_5 = median_filter(gau_30, 5)
    cv.imwrite("gau10_with_median_filter_3x3.bmp", gau_10_median_3)
    cv.imwrite("gau30_with_median_filter_3x3.bmp", gau_30_median_3)
    cv.imwrite("gau10_with_median_filter_5x5.bmp", gau_10_median_5)
    cv.imwrite("gau30_with_median_filter_5x5.bmp", gau_30_median_5)

    sap_1_median_3 = median_filter(sap_1, 3)
    sap_2_median_3 = median_filter(sap_2, 3)
    sap_1_median_5 = median_filter(sap_1, 5)
    sap_2_median_5 = median_filter(sap_2, 5)
    cv.imwrite("sap1_with_median_filter_3x3.bmp", sap_1_median_3)
    cv.imwrite("sap2_with_median_filter_3x3.bmp", sap_2_median_3)
    cv.imwrite("sap1_with_median_filter_5x5.bmp", sap_1_median_5)
    cv.imwrite("sap2_with_median_filter_5x5.bmp", sap_2_median_5)

# (e) open_then_close and close_then_open filter on (a)(b) with octogonal35553
    kernel = [(-1,2),(0,2),(1,2),(-2,1),(-1,1),(0,1),(1,1),(2,1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(-1,-2),(0,-2),(1,-2)]
    gau_10_open_then_close = closing(opening(gau_10, kernel), kernel)
    gau_30_open_then_close = closing(opening(gau_30, kernel), kernel)
    gau_10_close_then_open = opening(closing(gau_10, kernel), kernel)
    gau_30_close_then_open = opening(closing(gau_30, kernel), kernel)
    cv.imwrite("gau10_open_close.bmp", gau_10_open_then_close)
    cv.imwrite("gau30_open_close.bmp", gau_30_open_then_close)
    cv.imwrite("gau10_close_open.bmp", gau_10_close_then_open)
    cv.imwrite("gau30_close_open.bmp", gau_30_close_then_open)

    sap_1_open_then_close = closing(opening(sap_1, kernel), kernel)
    sap_2_open_then_close = closing(opening(sap_2, kernel), kernel)
    sap_1_close_then_open = opening(closing(sap_1, kernel), kernel)
    sap_2_close_then_open = opening(closing(sap_2, kernel), kernel)
    cv.imwrite("sap1_open_close.bmp", sap_1_open_then_close)
    cv.imwrite("sap2_open_close.bmp", sap_2_open_then_close)
    cv.imwrite("sap1_close_open.bmp", sap_1_close_then_open)
    cv.imwrite("sap2_close_open.bmp", sap_2_close_then_open)

# (f) signal-to-ratio for each instance
    print("Gaussian 10=",SNR(img, gau_10))
    print("Gaussian 30=",SNR(img, gau_30))
    print("Salt and pepper 0.1=",SNR(img, sap_1))
    print("Salt and pepper 0.05=",SNR(img, sap_2))
    print("Gau 10 3x3 Box=",SNR(img, gau_10_box_3))
    print("Gau 30 3x3 Box=",SNR(img, gau_30_box_3))
    print("Gau 10 5x5 Box=",SNR(img, gau_10_box_5))
    print("Gau 30 5x5 Box=",SNR(img, gau_30_box_5))
    print("SaltPepper 0.1 3x3 Box=",SNR(img, sap_1_box_3))
    print("SaltPepper 0.05 3x3 Box=",SNR(img, sap_2_box_3))
    print("SaltPepper 0.1 5x5 Box=",SNR(img, sap_1_box_5))
    print("SaltPepper 0.05 5x5 Box=",SNR(img, sap_2_box_5))
    print("Gau 10 3x3 Median=",SNR(img, gau_10_median_3))
    print("Gau 30 3x3 Median=",SNR(img, gau_30_median_3))
    print("Gau 10 5x5 Median=",SNR(img, gau_10_median_5))
    print("Gau 30 5x5 Median=",SNR(img, gau_30_median_5))
    print("SaltPepper 0.1 3x3 Median=",SNR(img, sap_1_median_3))
    print("SaltPepper 0.05 3x3 Median=",SNR(img, sap_2_median_3))
    print("SaltPepper 0.1 5x5 Median=",SNR(img, sap_1_median_5))
    print("SaltPepper 0.05 5x5 Median=",SNR(img, sap_2_median_5))
    print("Gau 10 OtC=",SNR(img, gau_10_open_then_close))
    print("Gau 30 OtC=",SNR(img, gau_30_open_then_close))
    print("Gau 10 CtO=",SNR(img, gau_10_close_then_open))
    print("Gau 30 CtO=",SNR(img, gau_30_close_then_open))
    print("SaltPepper 0.1 OtC=",SNR(img, sap_1_open_then_close))
    print("SaltPepper 0.05 OtC=",SNR(img, sap_2_open_then_close))
    print("SaltPepper 0.1 CtO=",SNR(img, sap_1_close_then_open))
    print("SaltPepper 0.05 CtO=",SNR(img, sap_2_close_then_open))