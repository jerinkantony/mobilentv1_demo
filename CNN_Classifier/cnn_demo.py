import cv2
import sys
import os
import numpy as np
from parent_cnn import getDeepNNClass


if __name__=="__main__":


	if 1:
		###### categorical experiment ###########
		dbname= 'japanese_NPR_purpose_class'
		
		CNN_Classifier = getDeepNNClass('CNNCategorical')
		if 1: # 0/1 training
			for i in range(1):
				CNN_Classifier = getDeepNNClass('CNNCategorical')
				if 1: # 0/1 training
					classifier_obj = CNN_Classifier(dbname=dbname)
					#classifier_obj.pretrain()
					classifier_obj.train_model(epochs = 30)
					#classifier_obj.SplitTrainVal()

		if 1: #0/1 validation	
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate(folder='val',show_flag=False)
	
		if 0: #0/1 testing
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate(folder='test',show_flag=False)
		if 0: #0/1 training folder
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate(folder='train',show_flag=False)
					
		if 0: # test on image
			classifier_obj = CNN_Classifier(dbname=dbname)      
			img=cv2.imread('DB/watermeter_digit_class/fixed_roi_data/val/7/338_4.jpg')
			class_index, predicted_label = classifier_obj.Predict(img)
			#cv2.imshow('img',img)
			#cv2.waitKey(10)
			print('\n### predicted_label:',predicted_label)

		if 0: #Live test
			classifier_obj = CNN_Classifier(dbname=dbname) 
			from ign_utils.general_utils import get_timestamp, mkdir_safe
			cap = cv2.VideoCapture(0)
			count=0
			
			print('Available Classes:')
			for i,clas in enumerate(classifier_obj.classes):
			    print(i,clas)
			    
			ch = input('class number to write images, esc to quit, enter to not write')
			wrt=0
			
			if ch.isdigit():
			    if int(ch)<len(classifier_obj.classes):
			        wrt=1
			        folder_path = 'test_rec'
			        
			        folder_path = os.path.join(folder_path, classifier_obj.classes[int(ch)])
			        mkdir_safe(folder_path)
			k=0
			
			while 1:
				ret,img = cap.read()
				
				class_index, predicted_label = classifier_obj.Predict(img)
				
				cv2.imshow('img',img)
				sys.stdout.write( '### frame: '+str(count)+' predicted_label:'+ predicted_label)
				sys.stdout.flush()
				sys.stdout.write("\r")
				sys.stdout.flush()
				
				k=cv2.waitKey(30)
				#if k==ord('w'):
				#	cv2.imwrite('')
				#import pdb;pdb.set_trace()
				if wrt==1 and (predicted_label != classifier_obj.classes[int(ch)]):
				    filename = os.path.join(folder_path, get_timestamp()+ '.jpg')
				    cv2.imwrite(filename, img)
               
				count+=1
				if k==27:
					break

		##################################################

		###### Audio experiment ###########
	if 0:
		import librosa
        
		###### categorical experiment ###########
		dbname = 'Renesas_audio'
		CNN_Classifier = getDeepNNClass('Audio_categorical')
		if 1: # 0/1 training
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.train_model(epochs=200)
		
		if 1: # validate 
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate()

		if 1: # test on audio sample
			classifier_obj = CNN_Classifier(dbname=dbname)  
			audiosample = 'DB/Renessas_audio/val/fan_off/Akshaykashok_Cloud_20200408_00_original_01.wav'
			data, __ = librosa.load(audiosample, sr=8000)
			data = data.astype(np.float32)    
			class_index, predicted_label = classifier_obj.Predict(data)
			print('\n### '+audiosample+' predicted_label:', predicted_label)
		if 1: # Live record and test
			classifier_obj = CNN_Classifier(dbname=dbname)  
			classifier_obj.LiveTest()
			
	if 0:
		###### regression experiment ###########
		dbname='mnist_regress'
		CNN_Classifier = getDeepNNClass('CNNIRegression')
		if 1: # 0/1 training
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.train_model(epochs=10)

		if 1: # test on folder
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate()
		##################################################

	if 0:
		###### siamese example ###########
		dbname='attire_siamese' #'mnist_regress'
		CNN_Classifier = getDeepNNClass('CNNSiamese')
		if 1: # 0/1 training
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.train_model(epochs=10)

		if 1: # test on folder
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate()
		##################################################
		
	if 0:
		######  classification binary ###########
		dbname='sample_binary_gear'
		CNN_Classifier = getDeepNNClass('CNNImageBinary')
		if 1: # 0/1 training
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.train_model(epochs=10)

		if 1: # test on folder
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate()
		##################################################

	if 0:
		###### Meter regression ###########
		dbname='meter_regression'
		CNN_Classifier = getDeepNNClass('CNNIRegression')
		if 1: # 0/1 training
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.train_model(epochs=500)

		if 1: # test on folder
			classifier_obj = CNN_Classifier(dbname=dbname)
			classifier_obj.validate()
		##################################################
	if 0:
		dbname='flki_attire_dataset'
		CNN_Classifier = getDeepNNClass('CNNCategorical')
		classifier_obj = CNN_Classifier(dbname=dbname)
		classifier_obj.SplitTrainVal()
			
	

				





		

	


		
		 
		
