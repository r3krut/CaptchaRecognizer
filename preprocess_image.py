import os
import cv2 as cv
import numpy as np
import argparse

def preprocess_img(img):
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
	denoised_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
	negate_img = cv.bitwise_not(denoised_img)
	return negate_img

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--path_to_img', default='path', type=str)
	arg('--path_to_save', default='path', type=str)

	args = parser.parse_args()

	img = cv.imread(str(args.path_to_img), 1)
	prep_img = preprocess_img(img)
	cv.imwrite(str(args.path_to_save), prep_img, [cv.IMWRITE_JPEG_QUALITY, 100])

