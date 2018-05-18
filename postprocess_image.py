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


def generate_bounds(img, img_to_draw, pix_values: dict):
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

	img_copy = img_to_draw.copy()
	cnt_num = 0
	for br in bound_rects:
		x = br[0]
		y = br[1]
		w = x + br[2]
		h = y + br[3]
		cv.rectangle(img_copy, (x,y), (w,h),(255,0,0),1)
		print("Contour [{0}]: {1}".format(cnt_num, br))
		cnt_num+=1
	print("Number of contours: {}".format(len(bound_rects)))
	cv.imshow('img', img_copy)	
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--img_path', type=str, 
		default='/home/r3krut/DataSets/NalogCaptchaDataTraining/CaptchaRecognition/predict/predicted_unet16/preds_masks_colored/img_3.png')

	args = parser.parse_args()

	img = cv.imread(args.img_path, 1)
	gray_img = pd.rgb_mask_to_gray(img)

	unique_colors = distinct_colors(gray_img)
	print(unique_colors)

	generate_bounds(gray_img, img, unique_colors)

	#cv.imshow('Image', gray_img)
	#cv.waitKey(0)
	#cv.destroyAllWindows()