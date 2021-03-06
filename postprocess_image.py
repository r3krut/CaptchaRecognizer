import cv2 as cv
import numpy as np
import argparse

import prepare_data as pd

colors_map = {0:-1, 20:0, 40:1, 60:2, 80:3, 100:4, 120:5, 140:6, 160:7, 180:8, 200:9}

def distinct_colors(img):
	height = img.shape[0]
	width = img.shape[1]

	colors = {}
	for h in range(0, width-1):
		for w in range(0, height-1):
			pix_val = img[w,h]
			colors[pix_val] = colors_map[pix_val]

	return colors

def select_pixels(img, pix_val: int):
	lower = np.array(pix_val, dtype='uint8')
	upper = np.array(pix_val, dtype='uint8')

	mask = cv.inRange(img, lower, upper)
	return mask


def generate_bounds(img, pix_values: dict):
	bound_rects = []

	for key, value in pix_values.items():
		if key == 0:
			continue
		selected_img = select_pixels(img, key)
		_, contours, hierarchy = cv.findContours( selected_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		for contour in contours:
			bb = cv.boundingRect(contour)
			if bb[2] > 10 and bb[3] > 10:
				bound_rects.append(bb)
	bound_rects.sort(key=lambda br: int(br[0])) #sorting bound rectangles by x-coordinate
	return bound_rects


def what_digit(img):
	r"""
 		Calculates what digit contains in image
		Performs search the most common pixels in image, after it do matching with colors_map
		
		Args: 
			img: input image
	"""

	colors = {k*20 : 0 for k in range(0, 11)}

	height = img.shape[0]
	width = img.shape[1]

	for h in range(0, width-1):
		for w in range(0, height-1):
			pix_val = img[w,h]
			if pix_val == 0:
				continue
			colors[pix_val] += 1

	max_val = max(colors, key=colors.get)
	return colors_map[max_val] #what digit

def predict_number_of_digits(img):
	r"""
		Naive predictor
		Tries to predict how much digits contains in given image by the average pix_values
		
		Args: 
			img: input image
	"""

	width = img.shape[1]

	#When in bounding box contains one digit (avg width equals 27)
	if width >= 27 - 20 and width < 27 + 10:
		return 1
	elif width >= 49 - 12 and width < 49 + 16: #for 2 digits (avg width equals 49)
		return 2
	elif width >= 73 - 8 and width < 73 + 17: #for 3 digits (avg width equals 73)
		return 3
	elif width >= 83 and width < 95 + 5: #for 4 digits (avg width equals 95)
		return 4
	return 5

def combine_two_rects(b1, b2):
	r"""
		Combines two rectangles to one

		Args:
			b1 (NumPy array): first bound rect
			b2 (NumPy array): second bound rect
	"""
	x1=b1[0]
	y1=b1[1] 
	w1=b1[2] 
	h1=b1[3]
	
	x2=b2[0]
	y2=b2[1]
	w2=b2[2]
	h2=b2[3]
	
	new_x = x1
	new_w = (x2+w2)-x1
	new_y = 0
	if y1 <= y2:
		new_y = y1
		new_h = (y2+h2)-y1
	else:
		new_y = y2
		new_h = (y1+h1)-y2
	return (new_x, new_y, new_w, new_h)

def combine_nearest(bound_rectangles: list, threshold: int):
	r"""
		Tries combine nearest rectangles by x-coordinate.

		Args:
			bound_rectangles (list): list of rectangles sorted by x-coordinate
			threshold (int): threshold which used to combine rectangles by x-coordinate. 
						     If 'abs(rect1.x - rect2.x) <= threshold' is true then combine to one, else do not combine
	"""
	nearest_rectangles = []
	nearest_dict = {}
	for i in range(0, len(bound_rectangles)-1):
		b1 = bound_rectangles[i]
		for j in range(i+1, len(bound_rectangles)):
			b2 = bound_rectangles[j]
			if abs(b1[0] - b2[0]) <= threshold:
				nearest_rectangles.append((b1,b2))
				nearest_dict[i] = '0'
				nearest_dict[j] = '0'
	combined_rects = []
	for nr in nearest_rectangles:
		combined_rects.append(combine_two_rects(nr[0], nr[1]))

	for i in range(0, len(bound_rectangles)):
		if i not in nearest_dict:
			combined_rects.append(bound_rectangles[i])
	combined_rects.sort(key=lambda br: int(br[0]))
	return combined_rects

def recognize(img):
	gray_img = pd.rgb_mask_to_gray(img)
	unique_colors = distinct_colors(gray_img)
	bound_rects = generate_bounds(gray_img, unique_colors)

	answer = ""
	if len(bound_rects) > 6:
		copy_img = img.copy()
		#try to combine nearest rectangles
		combined_rects = combine_nearest(bound_rects, threshold=12)
		if len(combined_rects) == 6:
			for br in combined_rects:
				x = br[0]
				y = br[1]
				xw = x + br[2]
				yh = y + br[3]
				sub_img = gray_img[y:yh,x:xw]
				digit = what_digit(sub_img)
				answer += str(digit)
		else:
			for br in combined_rects:
				x = br[0]
				y = br[1]
				xw = x + br[2]
				yh = y + br[3]
				sub_img = gray_img[y:yh,x:xw]
				digit = what_digit(sub_img)
				count_digits = int(predict_number_of_digits(sub_img))
				answer += str(digit)*count_digits
	elif len(bound_rects) == 6:
		for br in bound_rects:
			x = br[0]
			y = br[1]
			xw = x + br[2]
			yh = y + br[3]
			sub_img = gray_img[y:yh,x:xw]
			digit = what_digit(sub_img)
			answer += str(digit)
	else:
		for br in bound_rects:
			x = br[0]
			y = br[1]
			xw = x + br[2]
			yh = y + br[3]
			sub_img = gray_img[y:yh,x:xw]
			digit = what_digit(sub_img)
			count_digits = int(predict_number_of_digits(sub_img))
			answer += str(digit)*count_digits
	return answer if len(answer) == 6 else "Bad answer [ " + answer + " ]\nNumber of digits are equals " + str(len(answer))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--img_path', type=str, 
		default='path_to_img/img.png')
	args = parser.parse_args()

	img = cv.imread(args.img_path, 1)
	answer = recognize(img)
	print(answer)