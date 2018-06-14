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

path_to_tmp = Path('tmp') #path to temporary directory. Nessesary to save images after preprocessing and also saving just generated mask.
accuracy_test = Path('cap_data/accuracy_test')

def calculate_accuracy(model):
	t = 0 #true count
	f = 0 #false count
	for file in list(accuracy_test.glob('*')):
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
	print("Count of test samples : {0}".format(len(list(accuracy_test.glob('*')))))
	print("Accuracy : {0}".format(accuracy))

def recognize_cap(model, path_to_cap):
	#preprocessing
	img = cv.imread(str(path_to_cap), 1)
	negate_img = pre.preprocess_img(img)
	cv.imwrite(str(path_to_tmp / ('1.jpg')), negate_img, [cv.IMWRITE_JPEG_QUALITY, 100]) 

	#mask prediction
	gm.predict(model, [str(path_to_tmp / '1.jpg')], 1, path_to_tmp)
	mask_img = cv.imread(str(path_to_tmp / 'mask.png'), 1)

	#postprocessing and generate answer
	answer = post.recognize(mask_img)
	return answer

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--model_path', type=str, default='data/models/unet11', help='path to model folder')
	arg('--model_type', type=str, default='unet11', help='network architecture',
		choices=['unet11', 'unet16'])
	arg('--image_path', type=str, help='path to image', default='.')
	arg('--work_mode', type=str, help='Chose a work mode', default='recognize', choices=['recognize', 'accuracy'])
	
	args = parser.parse_args()
	
	#load model
	model = gm.get_model(str(Path(args.model_path).joinpath('model_{model}.pt'.format(model=args.model_type))), model_type=args.model_type)

	if args.work_mode == 'recognize':
		answer = recognize_cap(model, args.image_path)
		print("Model : {0}".format(args.model_type))
		print("Answer : {0}".format(answer))
	else:
		calculate_accuracy(model)