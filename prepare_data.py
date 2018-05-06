"""
Tranform colored masks to a grayscale masks and split images and masks on two dirs
digit 0 - 1 class
digit 1 - 2 class
digit 2 - 3 class
digit 3 - 4 class
digit 4 - 5 class
digit 5 - 6 class
digit 6 - 7 class
digit 7 - 8 class
digit 8 - 9 class
digit 9 - 10 class
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

digit_factor = 20

data_path = Path('/home/r3krut/DataSets/NalogCaptchaDataTraining/CaptchaRecognition/')
final_prepared = data_path / 'FinallPreparedImages'
data_train = data_path / 'train'
data_images = data_train / 'images'
data_masks = data_train / 'masks'

def rgb_mask_to_gray(rgb_img):
	height = rgb_img.shape[0]
	width = rgb_img.shape[1]
	gray_mask = np.zeros((height, width))

	for h in range(0, width-1):
		for w in range(0, height-1):
			pixel = rgb_img[w, h]
			if all(pixel == (255,0,0)):		#Color(BGR) for digit 0
				gray_mask[w,h] = 1
			if all(pixel == (0,255,0)):		#Color(BGR) for digit 1
				gray_mask[w,h] = 2
			if all(pixel == (255,255,0)):	#2
				gray_mask[w,h] = 3
			if all(pixel == (0,0,255)):		#3
				gray_mask[w,h] = 4
			if all(pixel == (255,0,255)):	#4
				gray_mask[w,h] = 5
			if all(pixel == (0,255,255)):	#5
				gray_mask[w,h] = 6
			if all(pixel == (0,150,255)):	#6
				gray_mask[w,h] = 7
			if all(pixel == (115,0,255)):	#7
				gray_mask[w,h] = 8
			if all(pixel == (0,100,0)):		#8
				gray_mask[w,h] = 9
			if all(pixel == (255,140,140)):	#9
				gray_mask[w,h] = 10
	gray_mask = gray_mask.astype(np.uint8) * digit_factor
	return gray_mask


if __name__ == '__main__':
	data_train.mkdir(exist_ok=True, parents=True)
	data_images.mkdir(exist_ok=True, parents=True)
	data_masks.mkdir(exist_ok=True, parents=True)

	#Total 121 samples
	for num_dir in tqdm(range(0,121)): 
		current_dir = final_prepared / str(num_dir)
		for file in list(current_dir.glob('*')):
			if 'mask' in str(file):
				mask_img = cv2.imread(str(file), 1)
				gray_mask = rgb_mask_to_gray(mask_img)
				cv2.imwrite(str(data_masks / ('img_' + str(num_dir) + '.png')), gray_mask)
			else:
				src_img = cv2.imread(str(file), 0)
				cv2.imwrite(str(data_images / ('img_' + str(num_dir) + '.jpg')), src_img, 
					[cv2.IMWRITE_JPEG_QUALITY, 100]) 

