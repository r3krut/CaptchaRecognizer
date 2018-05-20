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
	bound_rects.sort(key=lambda br: int(br[0])) #sorting boud rectangles by x-coordinate
	return bound_rects


"""
 	Calculates what digit contains in image
	Performs search the most common pixels in image, after it do matching with colors_map
"""
def what_digit(img):
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

"""
	Naive predictor
	Tries to predict how much digits contains in given image by the average values
"""
def predict_number_of_digits(img):
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

def recognize(img):
	gray_img = pd.rgb_mask_to_gray(img)
	unique_colors = distinct_colors(gray_img)
	bound_rects = generate_bounds(gray_img, unique_colors)

	answer = ""
	if len(bound_rects) > 6:
		print("Bad captcha. Count of bounding rectangles are equals {0}".format(len(bound_rects)))
		copy_img = img.copy()
		for br in bound_rects:
			x = br[0]
			y = br[1]
			w = x + br[2]
			h = y + br[3]
			cv.rectangle(copy_img, (x,y), (w,h), (255,0,0), 1)
		cv.imshow('Wrong image', copy_img)
		cv.waitKey(0)
		cv.destroyAllWindows()
		return "Wrong image"
	elif len(bound_rects) == 6:
		for br in bound_rects:
			x = br[0]
			y = br[1]
			w = x + br[2]
			h = y + br[3]
			sub_img = gray_img[y:h,x:w]
			digit = what_digit(sub_img)
			answer += str(digit)
	else:
		for br in bound_rects:
			x = br[0]
			y = br[1]
			w = x + br[2]
			h = y + br[3]
			sub_img = gray_img[y:h,x:w]
			digit = what_digit(sub_img)
			count_digits = int(predict_number_of_digits(sub_img))
			answer += str(digit)*count_digits
	return answer if len(answer) == 6 else "Bad answer [ " + answer + " ]\nNumber of digits are equals " + str(len(answer))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--img_path', type=str, 
		default='/home/r3krut/DataSets/NalogCaptchaDataTraining/CaptchaRecognition/predict/predicted_unet16/preds_masks_colored/img_3.png')
	args = parser.parse_args()

	img = cv.imread(args.img_path, 1)
	answer = recognize(img)
	print(answer)
	