import numpy as np
import os
import cv2 as cv

#Размеры каптчи
imgs_rows = 100
imgs_cols = 200

def load_images(path_to_data=''):
	path_exists = os.access(path_to_data, os.F_OK)
	if not path_exists:
		print("Specified path not exists!\n")
	else:
		count = 0
		files = os.listdir(path_to_data)
		
		imgs = np.ndarray((len(files), imgs_rows, imgs_cols), dtype=np.uint8)
		img_masks = np.ndarray((len(files), imgs_rows, imgs_cols), dtype=np.uint8)

		for f in files:
			images = os.listdir(path_to_data + f)
			for i in images:
				if i.find('maks') != -1 :
					mask_img = cv.imread(path_to_data + f + '/' + i, 1)
					mask_img = np.array([mask_img])
					img_masks[count] = mask_img
				else:
					src_img = cv.imread(path_to_data + f + '/' + i, 0)
					src_img = np.array([src_img])
					imgs[count] = src_img
			count += 1
			print('===Processed: {0}'.format(path_to_data + f + '/'))
		print("Count processed dirs: {0}".format(count))
	return (imgs, img_masks)