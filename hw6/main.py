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

def down_sampling(img):
    width, height = 512, 512
    new_img = np.zeros((64,64))
    for i in range(64):
        for j in range(64):
            new_img[i][j] = img[i*8][j*8] # left top point = new point
    return new_img # new img is a 64*64 array

def valid(x, y, w, h):
    return x >= 0 and x < w and y >= 0 and y < h 

def yokoi_conn(img, x, y, w, h, cor):
    new_x = x+cor[0]
    new_y = y+cor[1]
    if valid(new_x, new_y, w, h):
        return img[new_x][new_y]
    return 0

def h_equation(b, c, d, e):
    if b == c and (d != e or e != b): return 'q'
    if b == c and (d == b and e == b): return 'r'
    if b != c: return 's'

def yokoi(img, corner):
    width, height = img.shape[:2]
    result = []
    for i in range(width):
        row = []
        for j in range(height):
            if img[i][j] == 0:
                # print space
                row.append(' ')
                continue
            else:
                # check every corner's h_equation then check qrs's number
                cnt = []
                for cor in corner:
                    # four corner test
                    b = img[i][j]
                    c = yokoi_conn(img, i, j, width, height, cor[0])
                    d = yokoi_conn(img, i, j, width, height, cor[1])
                    e = yokoi_conn(img, i, j, width, height, cor[2])
                    cnt.append(h_equation(b, c, d, e))
                # r = 4 : 5 , else print q's number
                if cnt.count('r') == 4:
                    row.append('5')
                else:
                    if cnt.count('q') == 0:
                        row.append(' ')
                    else:
                        row.append(str(cnt.count('q')))
        result.append(row)
    return np.array(result)

def save(result):
    file = open('output.txt', 'w')
    with open('output.txt', 'w') as file:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                file.write(result[i][j])
            file.write('\n')

corner = [[[1,0],[1,-1],[0,-1]],[[0,-1],[-1,-1],[-1,0]],[[-1,0],[-1,1],[0,1]],[[0,1],[1,1],[1,0]]]
img = cv.imread("lena.bmp",0)
binary_img = binary(img)
ds_img = down_sampling(binary_img)
yokoi_arr = yokoi(ds_img, corner)
save(yokoi_arr)