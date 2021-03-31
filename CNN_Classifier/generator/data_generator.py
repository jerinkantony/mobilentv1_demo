import os
import sys
import os.path as osp
path = osp.dirname(osp.abspath(__file__))
sys.path.append(path)
sys.path.append(os.path.join(path,'..'))
import cv2
import glob

import math
import random
import itertools
import numpy as np
import os.path as osp

from keras.utils import Sequence, to_categorical
from sklearn.utils.class_weight import compute_class_weight

from ign_utils.general_utils import get_minimum_seq, GetAllFiles, insert_val_to_list, do_sometimes, safe_clone, read_json
from ign_utils.img_utils import readxml, processRects, four_point_transform, Get_Min_Max_intensity_level,\
								get_points_refined, Change_Background

def GetGeneratorObject(objectname):
	if objectname=='LSTMGENERATOR':
		return SeqDataGenerator
	
	elif objectname=='IMAGEGENERATOR':
		return DataGenerator
	
	elif objectname=='IMAGEGENERATOR_Binary':
		return IMAGEGENERATOR_Binary
	
	elif objectname=='REGIMAGEGENERATOR':
		return RegDataGenerator
	
	elif objectname=='SIAMESEGENERATOR2D':
		return SiameseGenerator2D
	
	elif objectname=='SIAMESEGENERATOR1D':
		global ReadAudio, generate_random_noise_signal
		from ign_utils.audio_utils import ReadAudio, generate_random_noise_signal
		return SiameseGenerator1D

	elif objectname=='XMLDataGenerator':
		return XMLDataGenerator
	
	elif objectname=='AudioDataGenerator':

		global ReadAudio, generate_random_noise_signal
		from ign_utils.audio_utils import ReadAudio, generate_random_noise_signal
		return AudioDataGenerator
	
	elif objectname=='AugDataGenerator':
		safe_clone('http://10.201.0.12:9999/ml/DB', 'backgrounds', dst=osp.join(path, 'augment', 'backgrounds'))
		global ET
		import xml.etree.ElementTree as ET
		return AugDataGenerator
	
	else:
		print('Invalid Generator Type!')


class DataGenerator(Sequence):
	'''
	Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
	'''
	def __init__(self, pre_process=None,datamode='train', dbname='car_videos', inp_dim=(224, 224, 3), shuffle=True, augmentation=None,  batch_size=4, verify_data=False,classes=None, sample_rate=None):
		self.path              = osp.dirname(osp.abspath(__file__))
		self.datamode          = datamode
		self.dbname            = dbname
		self.inp_dim           = inp_dim
		self.shuffle           = shuffle
		self.augmentation      = augmentation
		self.pre_process       = pre_process
		self.batch_size        = batch_size
		self.data_aug          = None
		self.datadir           = os.path.join(self.path, '..','DB', dbname, datamode)
		self.classes           = classes 
		self.x_set, self.y_set = None, None
		self.allX              = None
		self.max_count         = 0
		self.verify_data       = verify_data
		self.unk_set           = []
		self.unk_count         = 0
		if len(self.inp_dim)>=3:
			self.img_height, self.img_width, self.ch = self.inp_dim[:3]
		self.x_set1 = self.LoadFilenames()
		if len(self.unk_set):
			self.allX = self.x_set1 + [self.unk_set]
			self.unk_count = min(2*self.max_count, len(self.unk_set))
			print('self.unk_count: ', self.unk_count)
		else:
			self.allX = self.x_set1
		self.GetXYPair()
		self.on_epoch_end()
		
		if not self.augmentation is None:
			if len(self.inp_dim) == 2:
				from augment.audio_aug import Audioaug
				self.data_aug   = Audioaug(dbname=dbname, sample_rate=sample_rate, mode=self.augmentation)
			else:
				from augment.img_aug import Imgaug
				self.data_aug   = Imgaug(mode=self.augmentation)
			
	def GetXYPair(self):
		#shortest_length, _ = get_minimum_seq(self.x_set1)
		#chosen_X_list = [random.sample(item, shortest_length) for item in self.x_set1]
		#import pdb;pdb.set_trace()
		self.x_set = [item for sublist in self.x_set1 for item in sublist]
		self.y_set = [l.split('/')[-2] for l in self.x_set]
		print('Total: ', len(self.x_set))
		self.y_set = self.EncodeLabels(self.y_set)
		if self.datamode == 'val':
			self.indexes = np.arange(len(self.x_set) + len(self.unk_set))
		else:
			self.indexes = np.arange(len(self.x_set) + self.unk_count)
		assert len(self.x_set)==len(self.y_set)
	
	def GetClassWeights(self):
		temporary_y_set = [l.split('/')[-2] for l in list(itertools.chain.from_iterable(self.allX))]
		class_weights = compute_class_weight('balanced', np.unique(temporary_y_set), temporary_y_set)
		d_class_weights = dict(enumerate(class_weights))
		return d_class_weights
		
	def on_epoch_end(self):  
		np.random.shuffle(self.indexes)
		
	def __len__(self):
		return math.ceil(len(self.x_set) / self.batch_size)

	def __getitem__(self, idx):
		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		batch_x = [self.x_set[ind] for ind in indexes]
		batch_y = [self.y_set[ind] for ind in indexes]
		X, Y = self.getXYFromBatch(batch_x, batch_y)
		return (X, Y)

	def getXYFromBatch(self, batchx, batchy):
		X = np.array(([self.ReadFile(file_name) for file_name in batchx]))
		Y = np.array(batchy)
		return X, Y

	def LoadFilenames(self):
		print('Loading Data from {}'.format(self.datadir))
		x_dirs = glob.glob(osp.join(self.datadir, '*'))
		
		x_set = []
		for dir_ in x_dirs:
			files = glob.glob(osp.join(dir_, '*'))
			if len(files)>=1: x_set.append(files)
		return x_set

	def EncodeLabels(self, labels):
		y_set = [self.GetClassOneHot(class_name) for  class_name in labels]
		return y_set

	def ReadFile(self, filename):
		img = cv2.imread(filename)
		if self.augmentation is not None:
			img = self.data_aug.apply(img)
		return self.ProcessImage(img, self.img_height, self.img_width, self.ch)
		
	def ProcessImage(self, image, wd, ht, ch):
		
		if not self.pre_process is None:
			return self.pre_process(image).astype(np.float32)
			

	def GetClassOneHot(self, class_str):

		"""Given a class as a string, return its number in the classes
		list. This lets us encode and one-hot it for training."""

		# Encode it first.
		label_encoded = self.classes.index(class_str)
		# Now one-hot it.
		label_hot = to_categorical(label_encoded, len(self.classes))
		assert len(label_hot) == len(self.classes)
		return label_hot

class SeqDataGenerator(DataGenerator):
	def __init__(self, datamode='train', dbname='car_videos', inp_dim=(10, 224, 224, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True):
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, shuffle=shuffle)
		self.inp_dim = inp_dim
		self.seq_length, self.img_height, self.img_width, self.ch = self.inp_dim
		self.extensions = ['.png', '.jpg', '.JPG']
		assert len(self.inp_dim) == 4

	def ReadFile(self, folder):
		frames = GetAllFiles(folder, self.extensions)
		frames = sorted(self.GetSeqFrames(frames, self.seq_length))
		sequence = self.BuildImageSequence(frames)
		return sequence
	
	def GetSeqFrames(self, input_list, size):

		"""Given a list and seq length, return consecutive frames with a starting
		point taken as random
		"""
		assert len(input_list) >= size
		start_pt = random.randint(0, len(input_list)-(size))
		end_pt = start_pt + size
		return input_list[start_pt:end_pt]

	def BuildImageSequence(self, fileslist):
		"""Given a set of frames (filenames), build our sequence.""" 
		imglist = []
		for img in fileslist:
			imgn = cv2.imread(img)
			imgn = self.ProcessImage(imgn, self.img_height, self.img_width, self.ch)
			imglist.append(imgn)
		return imglist

class RegDataGenerator(DataGenerator):
	'''
	Custom Data Generator for Regression
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(28, 28, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True):
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, shuffle=shuffle)

	def EncodeLabels(self, labels):
		print('ENCODING FOR REGRESSION')
		y_set = [float(class_name) for  class_name in labels]
		return y_set

	def LoadFilenames(self):
		print('Loading Data from {}'.format(self.datadir), )
		x_set = glob.glob(osp.join(self.datadir, '*'))
		return x_set

	def GetXYPair(self):
		self.x_set = self.x_set1
		self.y_set = [float(osp.basename(l).split('.')[0].split('_')[-1]) for l in self.x_set]
		self.d_class_weights = None
		assert len(self.x_set)==len(self.y_set)

class IMAGEGENERATOR_Binary(DataGenerator):
	'''
	Custom Data Generator for Regression
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(28, 28, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True, verify_data=False, classes=None):
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, verify_data=verify_data, shuffle=shuffle, classes=classes)

	def EncodeLabels(self, labels):
		print('ENCODING FOR IMAGEGENERATOR_Binary')
		y_set = [self.classes.index(class_name) for  class_name in labels]
		return y_set

class SiameseGenerator2D(DataGenerator):
	'''
	Custom Data Generator for SiameseNetwork
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(28, 28, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True, verify_data=False, classes=None):
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, shuffle=shuffle, verify_data=verify_data, classes=classes)
		print('CALLED SiameseGenerator.......!!!!!!!')
	
	def ReadFile(self, filename):
		img1 = cv2.imread(filename[0])
		img2 = cv2.imread(filename[1])

		if self.augmentation is not None:
			img1 = self.data_aug.apply(img1)
			img2 = self.data_aug.apply(img2)
		processed_img1 = self.ProcessImage(img1, self.img_height, self.img_width, self.ch)
		processed_img2 = self.ProcessImage(img2, self.img_height, self.img_width, self.ch)
		return processed_img1, processed_img2

	def getXYFromBatch(self, batchx, batchy):
		X1=[]
		X2=[]
		for file_name in batchx:
			x1, x2 = self.ReadFile(file_name)
			X1.append(x1)
			X2.append(x2)
		X1=np.array(X1)
		X2=np.array(X2)
		X = [X1,X2]
		Y = np.array(batchy)
		return X, Y

	def GetXYPair(self):
		shortest_length, _ = get_minimum_seq(self.x_set1)
		chosen_X_list = [random.sample(item, shortest_length) for item in self.x_set1]
		
		pairs = []
		labels = []
		n = shortest_length - 1
		num_classes = len(chosen_X_list)

		for d in range(num_classes):
			for i in range(n):
				z1, z2 = chosen_X_list[d][i], chosen_X_list[d][i + 1]
				pairs.append([z1,z2])
				labels.append(1)
				inc = random.randrange(1, num_classes)
				dn = (d + inc) % num_classes
				z1, z2 = chosen_X_list[d][i], chosen_X_list[dn][i]
				pairs.append([z1,z2])
				labels.append(0)
		self.x_set = pairs
		self.y_set = labels 
		self.y_set = self.EncodeLabels(self.y_set)
		self.indexes = np.arange(len(self.x_set))

	def EncodeLabels(self, labels):
		return labels

class XMLDataGenerator1(DataGenerator):
	'''
	Custom Data Generator from XML files
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(28, 28, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True, verify_data=None):
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, shuffle=shuffle, verify_data=verify_data)

	def EncodeLabels(self, labels):
		pass
	
	def ReadFile(self, filename):
		img = cv2.imread(filename)
		extn = os.path.basename(filename).split('.')[-1]
		xmlp = filename.replace('images', 'labels')
		xmlp = xmlp.replace('.{}'.format(extn), '.xml')
		datadict = readxml(xmlp)
		imgList, labelList = processRects(datadict, img)
		if self.augmentation is not None:
			imgList = [self.data_aug.apply(img) for img in imgList]
		imgList = [self.ProcessImage(img, self.img_height, self.img_width, self.ch) for img in imgList]
		return imgList, labelList
		
	def LoadFilenames(self):
		print('Loading Data from {}'.format(self.datadir), )
		x_set = glob.glob(osp.join(self.datadir, 'images', '*'))
		print('x_set: ', len(x_set))
		return x_set
	
	def GetXmlPath(self, impath):
		xmlp = impath.replace('images', 'labels')
		extn = os.path.basename(impath).split('.')[-1]
		xmlp = xmlp.replace('.{}'.format(extn), '.xml')
		assert osp.isfile(xmlp)
		return xmlp

	def GetXYPair(self):
		# shortest_length, _ = get_minimum_seq(self.x_set1)
		# chosen_X_list = [random.sample(item, shortest_length) for item in self.x_set1]
		# import pdb;pdb.set_trace()
		self.x_set = self.x_set1
		self.y_set = [self.GetXmlPath(l) for l in self.x_set]
		self.indexes = np.arange(len(self.x_set))
		assert len(self.x_set)==len(self.y_set)

	def getXYFromBatch(self, batchx, batchy):
		img_ = []
		label_ = []
		for filename in batchx:
			imgarray, labelarray = self.ReadFile(filename)
			img_.append(imgarray)
			label_.append(labelarray)
		X = np.array(img_)
		X = np.squeeze(X)
		Y = np.array(label_)
		return X, Y

class XMLDataGenerator(DataGenerator):
	'''
	Custom Data Generator from XML files
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(28, 28, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True, classes=None, verify_data=None):
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, shuffle=shuffle, classes=classes, verify_data=verify_data)

	
	def ProcessImage(self, image, wd, ht, ch):
		if not self.pre_process is None:
			return self.pre_process(image).astype(np.float32)
		else:
			print('preprocess function missing in datagenerator line 133')
			exit()
			img = cv2.resize(image, (wd, ht), interpolation=cv2.INTER_LINEAR)
			if self.ch==1:
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				
			img=img.astype(np.float32)
			img = (img / 255.0)
			if self.ch == 1:
				img =np.expand_dims(img,axis=2)
			return img

	def ReadFile(self, filename):
		image = cv2.imread(filename[0])
		x, y, w, h = filename[1]
		img = image[y:y+h, x:x+w]
		box = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
		# unit_vec = get_direction_vector(box)
		# box = direction_boxsort(box, unit_vec, True)
		img, __ = four_point_transform(image, box, random_interpolation=True)
		if self.augmentation is not None:
			img = self.data_aug.apply(img, filename)
		return self.ProcessImage(img, self.img_height, self.img_width, self.ch)

	def GetImageLabelFromXml(self, filepath):
		extn = os.path.basename(filepath).split('.')[-1]
		xmlp = filepath.replace('images', 'labels')
		xmlp = xmlp.replace('.{}'.format(extn), '.xml')
		datadict = readxml(xmlp)
		X, Y = processRects(datadict, filepath)
		return X, Y	

	def LoadFilenames(self):
		print('Loading Data from {}'.format(self.datadir), )
		self.x_set = []
		self.y_set = []
		x_set = glob.glob(osp.join(self.datadir, 'images', '*'))
		for filename in x_set:
			X, Y = self.GetImageLabelFromXml(filename)
			self.x_set.extend(X)
			self.y_set.extend(Y)
		self.labelset = self.y_set.copy()
		return self.x_set
	
	def GetXmlPath(self, impath):
		xmlp = impath.replace('images', 'labels')
		extn = os.path.basename(impath).split('.')[-1]
		xmlp = xmlp.replace('.{}'.format(extn), '.xml')
		assert osp.isfile(xmlp)
		return xmlp

	def GetClassWeights(self):
		class_weights = compute_class_weight('balanced', np.unique(self.labelset), self.labelset)
		d_class_weights = dict(enumerate(class_weights))
		return d_class_weights

	def GetXYPair(self):
		self.x_set = self.x_set1
		print('\nFirst element in xset: {}\n'.format(self.x_set[0]))
		self.y_set = self.EncodeLabels(self.y_set)
		self.indexes = np.arange(len(self.x_set))
		assert len(self.x_set)==len(self.y_set)


class AugDataGenerator(DataGenerator):
	'''
	Custom Data Generator With OntheFly Augmentation
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(28, 28, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True, classes=None, verify_data=False):
		self.req_labelList = ['person']
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, shuffle=shuffle, classes=classes, verify_data=verify_data)
		self.bg_imglist = []
		
		self.bg_imglist = glob.glob(osp.join(self.path, 'augment', 'backgrounds', '*.jpg')) + glob.glob(osp.join(self.path, '..', 'backgrounds', '*.png'))
	
	def Get_coords_from_XML(self, xmlfilename):
		_coords_ = []
		tree = ET.parse(xmlfilename)
		root = tree.getroot()
		for obj in root.iter('object'):
			xmlbox = obj.find('bndbox')
			label_name = obj.find('name').text
			if label_name in self.req_labelList:
				x_min, y_min = self.process_int(xmlbox.find('xmin').text), self.process_int(xmlbox.find('ymin').text)
				x_max, y_max = self.process_int(xmlbox.find('xmax').text), self.process_int(xmlbox.find('ymax').text)
				_coords_.append([x_min, y_min, x_max, y_max])
		return _coords_

	def Get_contours_from_json(self, json_file):
		data = read_json(json_file)
		labeldata_list = data['shapes']
		return labeldata_list			
				
	def process_int(self, text):
		return int(text.strip("'").strip().split(".")[0])

	def Get_object_points(self, filename):
		xmlfile = osp.join(osp.dirname(filename), 'xml', osp.basename(filename).split('.')[0] + '.xml')
		jsonfile = osp.join(osp.dirname(filename), 'json', osp.basename(filename).split('.')[0] + '.json')
		
		if osp.isfile(jsonfile):
			pts = self.Get_contours_from_json(jsonfile)
			return [filename, (pts)]

		elif osp.isfile(xmlfile):
			pts = self.Get_coords_from_XML(xmlfile)
			return [filename, (pts)]
		else:
			return [filename]

	def LoadFilenames(self):
		
		print('Loading Data from {}'.format(self.datadir), )
		x_dirs = glob.glob(osp.join(self.datadir, '*'))
		x_set = []
		
		for dir_ in x_dirs:
			files = glob.glob(osp.join(dir_, '*'))
			if len(files)>=1:
				files = [self.Get_object_points(f) for f in files if osp.isfile(f)]
				x_set.append(files)
		return x_set

	def GetXYPair(self):
		self.x_set = [item for sublist in self.x_set1 for item in sublist]
		self.y_set = [l[0].split('/')[-2] for l in self.x_set]
		self.label_set = self.y_set.copy()
		self.y_set = self.EncodeLabels(self.y_set)
		self.indexes = np.arange(len(self.x_set))
		assert len(self.x_set)==len(self.y_set)

	def ReadFile(self, filename):
		if len(filename) > 1:
			contours = filename[1]
			if len(contours) and do_sometimes(1) and self.datamode == 'train':
				if len(self.bg_imglist):
					img = Change_Background(filename, bgImage=random.choice(self.bg_imglist))
				else:
					img = Change_Background(filename)
			else:
				img = cv2.imread(filename[0])
		else:
			img = cv2.imread(filename[0])
		if self.augmentation is not None:
			img = self.data_aug.apply(img)
		return self.ProcessImage(img, self.img_height, self.img_width, self.ch)

	def GetClassWeights(self):
		#print('\ntemporary_y_set: ', temporary_y_set, '\n')
		class_weights = compute_class_weight('balanced', np.unique(self.label_set), self.label_set)
		d_class_weights = dict(enumerate(class_weights))
		return d_class_weights

class AudioDataGenerator(DataGenerator):
	'''
	Custom Data Generator for AudioSamples
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(8000, 1), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True, verify_data=False, classes=None):
		sample_rate = 8000
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, verify_data=verify_data, shuffle=shuffle, classes=classes, sample_rate=sample_rate)
	
	def Filter_classes(self):
		unk_set = []
		x_dirs = [osp.join(self.datadir, i) for i in os.listdir(self.datadir) if i!='unknown']
		if 'unknown' in os.listdir(self.datadir):
			unk_set = glob.glob(osp.join(self.datadir, 'unknown', '*'))
		return x_dirs, unk_set

	def LoadFilenames(self):
		print('Loading Data from {}'.format(self.datadir))
		x_dirs, self.unk_set = self.Filter_classes()
		print('Unknown Data: ', len(self.unk_set))
		x_set = []
		max_count = 0
		for dir_ in x_dirs:
			files = glob.glob(osp.join(dir_, '*'))
			if len(files)>=1:
				if len(files) > max_count:
					max_count = len(files)
				x_set.append(files)
		self.max_count = max_count
		return x_set

	def EncodeLabels(self, labels):
		y_set = [self.GetClassOneHot(class_name) for  class_name in labels]
		return y_set

	def ReadFile(self, filename):
		if filename=='noise':
			data = generate_random_noise_signal()
		else:	
			data, __ = ReadAudio(filename)
		if self.augmentation is not None:
			data = self.data_aug.apply(data)
		
		return self.ProcessAudio(data)

	def __getitem__(self, idx):
		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		if self.unk_set:
			if self.datamode == 'val':
				sub_unklist_X = random.sample(self.unk_set, len(self.unk_set))
			else:
				sub_unklist_X = insert_val_to_list(random.sample(self.unk_set, self.unk_count), percentage=0.3, string_val='noise')

			sub_unklist_y = ['unknown' for l in sub_unklist_X]
			
			sub_unklist_y = self.EncodeLabels(sub_unklist_y)
			pool_list_to_draw_X = self.x_set + sub_unklist_X
			pool_list_to_draw_Y = self.y_set + sub_unklist_y
		else:
			pool_list_to_draw_X = self.x_set
			pool_list_to_draw_Y = self.y_set
		batch_x = [pool_list_to_draw_X[ind] for ind in indexes]
		batch_y = [pool_list_to_draw_Y[ind] for ind in indexes]
		X, Y = self.getXYFromBatch(batch_x, batch_y)
		X = np.expand_dims(X, axis=2)
		return (X, Y)

	def ProcessAudio(self, data):
		if self.pre_process is not None:
		    data = self.pre_process(data)
		return data

class SiameseGenerator1D(DataGenerator):
	'''
	Custom Data Generator for SiameseNetwork1D
	'''
	def __init__(self, datamode='train', dbname='mnist', inp_dim=(28, 28, 3), 
		batch_size=4, augmentation=None, pre_process=None, shuffle=True, verify_data=False, classes=None):
		sample_rate = 8000
		super().__init__(datamode=datamode, dbname=dbname, inp_dim=inp_dim, 
		pre_process=pre_process, batch_size=batch_size, augmentation=augmentation, shuffle=shuffle, verify_data=verify_data, classes=classes, sample_rate=sample_rate)
		print('CALLED SiameseGenerator1D.......!!!!!!!')
	
	def LoadFilenames(self):
		print('Loading Data from {}'.format(self.datadir))
		x_dirs, self.unk_set = self.Filter_classes()
		print('Unknown Data: ', len(self.unk_set))
		x_set = []
		max_count = 0
		for dir_ in x_dirs:
			files = glob.glob(osp.join(dir_, '*'))
			if len(files)>=1:
				if len(files) > max_count:
					max_count = len(files)
				x_set.append(files)
		self.max_count = max_count
		return x_set
	
	def Filter_classes(self):
		unk_set = []
		x_dirs = [osp.join(self.datadir, i) for i in os.listdir(self.datadir) if i!='unknown']
		if 'unknown' in os.listdir(self.datadir):
			unk_set = glob.glob(osp.join(self.datadir, 'unknown', '*'))
		return x_dirs, unk_set

	def ReadFile(self, filename):
		#get 1st sample
		if filename[0]=='noise':
			data1 = generate_random_noise_signal()
		else:	
			data1, __ = ReadAudio(filename[0])
		#get 2nd sample
		if filename[1]=='noise':
			data2 = generate_random_noise_signal()
		else:	
			data2, __ = ReadAudio(filename[1])

		if self.augmentation is not None:
			
			data1 = self.data_aug.apply(data1)
			data2 = self.data_aug.apply(data2)
			
		processed_audio1 = self.ProcessAudio(data1)
		processed_audio2 = self.ProcessAudio(data2)
		return processed_audio1, processed_audio2

	def getXYFromBatch(self, batchx, batchy):
		X1=[]
		X2=[]
		for file_name in batchx:
			x1, x2 = self.ReadFile(file_name)
			X1.append(x1)
			X2.append(x2)
		X1 = np.array(X1)
		X1 = np.expand_dims(X1, axis=2)
		X2 = np.array(X2)
		X2 = np.expand_dims(X2, axis=2)
		X = [X1,X2]
		Y = np.array(batchy)
		return X, Y

	def GetXYPair(self):
		shortest_length, _ = get_minimum_seq(self.x_set1)
		chosen_X_list = [random.sample(item, shortest_length) for item in self.x_set1]
		pairs = []
		labels = []
		n = shortest_length - 1
		num_classes = len(chosen_X_list)
		for d in range(num_classes):
			for i in range(n):
				z1, z2 = chosen_X_list[d][i], chosen_X_list[d][i + 1]
				pairs.append([z1,z2])
				labels.append(1)
				inc = random.randrange(1, num_classes)
				dn = (d + inc) % num_classes
				z1, z2 = chosen_X_list[d][i], chosen_X_list[dn][i]
				pairs.append([z1,z2])
				labels.append(0)
		self.x_set = pairs
		self.y_set = labels 
		self.y_set = self.EncodeLabels(self.y_set)
		if self.datamode == 'val':
			self.indexes = np.arange(len(self.x_set) + len(self.unk_set))
		else:
			self.indexes = np.arange(len(self.x_set) + self.unk_count)
	
	def __getitem__(self, idx):
		indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
		if self.unk_set:
			if self.datamode == 'val':
				sub_unklist_X = random.sample(self.unk_set, len(self.unk_set))
			else:
				sub_unklist_X = insert_val_to_list(random.sample(self.unk_set, self.unk_count), percentage=0.3, string_val='noise')

			sub_unklist_y = ['unknown' for l in sub_unklist_X]
			
			sub_unklist_y = self.EncodeLabels(sub_unklist_y)
			pool_list_to_draw_X = self.x_set + sub_unklist_X
			pool_list_to_draw_Y = self.y_set + sub_unklist_y
		else:
			pool_list_to_draw_X = self.x_set
			pool_list_to_draw_Y = self.y_set
		batch_x = [pool_list_to_draw_X[ind] for ind in indexes]
		batch_y = [pool_list_to_draw_Y[ind] for ind in indexes]
		X, Y = self.getXYFromBatch(batch_x, batch_y)
		
		return (X, Y)
		
	def EncodeLabels(self, labels):
		return labels

	def ProcessAudio(self, data):
		if self.pre_process is not None:
		    data = self.pre_process(data)
		return data

if __name__ == '__main__':
	from scipy import signal
	classes = os.listdir(osp.join(path, '..', 'DB', 'person_class', 'train'))
	sample_rate = 8000

	def pre_process(data):
		if len(data) < sample_rate:
			requid_nsamples = sample_rate - len(data)
			left = int(random.randrange(0, requid_nsamples+1))
			right = int(requid_nsamples-left)
			data = np.pad(data, (left, right), 'constant', constant_values=(0, 0))
		else:
			data = signal.resample(data, sample_rate)
		return data


	DataGenerator = GetGeneratorObject('AugDataGenerator')
	d_obj = DataGenerator(datamode = 'train',
		dbname='person_class',
		inp_dim=(224, 224, 3),
		batch_size=8,
		augmentation=None, 
		pre_process=None, 
		shuffle=True,
		classes=classes)
	# for i in range(10):
	x, y = d_obj.__getitem__(0)
	# print(y)
	print('x.shape: ', len(x), 'y: ', y)
