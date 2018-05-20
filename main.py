import argparse
import cv2 as cv
import prepare_train_val as ptv
import dataset as d
import transforms
import model as md
import numpy as np
import postprocess_image as post
import preprocess_image as pre
import prepare_data as pd
import generate_masks as gm

import torch

from pathlib import Path

path_to_tmp = Path('tmp')
path_to_tests = Path('/home/r3krut/DataSets/NalogCaptchaDataTraining/CaptchaRecognition/tests')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--model_path', type=str, default='data/models/unet11_binary_20', help='path to model folder')
	arg('--model_type', type=str, default='unet11', help='network architecture',
	    choices=['unet11', 'unet16'])
	arg('--image_path', type=str, help='path to image', default='.')
    
	args = parser.parse_args()
	
	#load model
	model = gm.get_model(str(Path(args.model_path).joinpath('model_{model}.pt'.format(model=args.model_type))),
    	model_type=args.model_type)

	t = 0 #true count
	f = 0 #false count
	for file in list(path_to_tests.glob('*')):
		true_label = file.name.split(".")[0]
		
		#preprocessing
		img = cv.imread(str(file), 1)
		negate_img = pre.preprocess_img(img)
		cv.imwrite(str(path_to_tmp / ('1.jpg')), negate_img, [cv.IMWRITE_JPEG_QUALITY, 100]) 

		#mask prediction
		gm.predict(model, [str(path_to_tmp / '1.jpg')], 1, path_to_tmp)
		mask_img = cv.imread(str(path_to_tmp / 'mask.png'), 1)

		#postprocessing and generate answer
		answer = post.recognize(mask_img)
		if true_label == answer:
			t += 1
			print("True : {0} = {1}".format(true_label, answer))
		else:
			f += 1
			print("False : {0} = {1}".format(true_label, answer))
	accuracy = t / (t + f)

	print("Model : {0}".format(args.model_type))
	print("Count of test samples : {0}".format(len(list(path_to_tests.glob('*')))))
	print("Accuracy : {0}".format(accuracy))