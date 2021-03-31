import os
import cv2
import sys
import math
import time
import glob
import json
import keras 
import random
import shutil
import itertools
import statistics
import numpy as np
import collections
import os.path as osp
import tensorflow as tf
import keras.backend as K

path = osp.dirname(osp.abspath(__file__))

import os
import sys

'''
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path,'ign_utils'))
from sort_pts import sort_aniclkwise
'''


def safe_clone_ign_utils():
    '''
    Copy this function to your main code nd call it. Then import any script in ign_utils. Enjoy
    '''
    path = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(path,'ign_utils')
    try:
        if not os.path.exists(dst):
            os.system("git clone http://10.201.0.12:9999/ml/ign_utils "+dst+" -b dev1 --single-branch --depth=1")
        else:
            print('found ign_utils')
            #os.system("git -C ign_utils/ pull")
    except:
         print('cloning failed: git clone http://10.201.0.12:9999/ml/ign_utils "+ " -b " + "dev1" + " --single-branch --depth=1')
safe_clone_ign_utils()
     
sys.path.append(path)
sys.path.append(os.path.join(path,'model'))
sys.path.append(os.path.join(path,'ign_utils'))

from sklearn.utils import class_weight
from keras.optimizers import SGD, Adadelta,Adam, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from ign_utils.ArgsManager import ArgsManager
from ign_utils.acc_utils import acc_matrics

from model.models import MODEL

from ign_utils.vis_perform_graph import PlotLosses
from ign_utils.general_utils import create_directory_safe, mkdir_safe
from ign_utils.model_utils import limit_gpu, copy_weights

def getDeepNNClass(objectname):
    if objectname   == 'CNNCategorical':
        return CNNCategorical
    
    elif objectname == 'CNNImageBinary':
        return CNNImageBinary
    
    elif objectname == 'CNNIRegression':
        return CNNIRegression
    
    elif objectname == 'CNNSiamese':
        return CNNSiamese
    
    elif objectname == 'Grex_categorical':
        return Grex_categorical
    
    elif objectname == 'Audio_categorical':
        global signal, wavfile, playsound, ReadAudio, record_audio
        from scipy import signal
        from scipy.io import wavfile
        from playsound import playsound
        from ign_utils.audio_utils import ReadAudio, record_audio
        return Audio_categorical
    
    elif objectname == 'AudioSiamese':
        global signal, wavfile, playsound, ReadAudio, record_audio
        from scipy import signal
        from scipy.io import wavfile
        from playsound import playsound
        from ign_utils.audio_utils import ReadAudio, record_audio
        return AudioSiamese
    
    else:
        print('Invalid option..!')
        
class DEEP_NN():
    def __init__(self, dbname, args):
        #limit_gpu(.3)
        
        print('dbname: ',dbname)
        self.path              = osp.dirname(osp.abspath(__file__))
        self.root_path         = self.path
        self.DB_path           = osp.join(self.root_path,'DB', dbname)
        self.wtpath            = osp.join(self.root_path,'weights', dbname)
        if osp.isdir(osp.join(self.DB_path, 'train')) and len(os.listdir(osp.join(self.DB_path, 'train'))) > 0:
            self.classlist = os.listdir(osp.join(self.DB_path, 'train'))
        else:
            self.classlist = None
        self.args              = args
        
        self.dbname            = self.args( name='dbname', value=dbname,
        choices='mnist, cardoor_classifier, plate_class', help='Name of db',overwrite=True)
         
        self.generator_type    = self.args( name='datagenerator', value='AugDataGenerator',
        choices='IMAGEGENERATOR_Binary, REGIMAGEGENERATOR, LSTMGENERATOR, IMAGEGENERATOR',
        help='Generator Type')
         
        self.load_prev_weights = self.args( name='load_prev_weights', value=False, 
        choices='true, false', help='Train from scratch/Continue training Mode')
        
        self.drop_outs         = self.args( name='drop_outs', value=[0.1], 
        choices='null, [0.1],[0.1,0.1]', help='list of dropout layers, except last layer')
        
        self.FC_LAYERS         = self.args( name='FC_LAYERS', value=[16], 
        choices='null, [16],[16,32]', help='list of fully connected layers, except last layer')
        
        self.inp_dim           = self.args( name='inp_dim', value=(120, 160, 3), 
        choices=None, help='Input Dimension')
        
        self.mode              = self.args( name='class_mode', value='categorical', 
        choices='regression, categorical, binary', help='Statistical Data Type')
        
        self.augm              = self.args( name='augment_type', value='sample_opencv_aug',
        choices='simple_albumentaion, strong_albumentaion, sample_opencv_aug', 
        help='Type of Augmentation')

        self.verify_data      = self.args( name='verify_data', value=True,
        choices='true, false', help='Enable/Disable imshow')
        
        self.show_plot      = self.args( name='show_plot', value=False,
        choices='true, false', help='Enable/Disable training plot')

        self.modelname         = self.args( name='model', value='People_model_lite',
        choices='MobileNet, MobileNetv2,ResNet18, customcnn,RX_model_digit, regressioncnn, customLSTM, attire_model...',
        help='Available Network Models')
        
        '''
        import string
        import random
        def string_generator(size=0):
            chars = string.ascii_uppercase + string.ascii_lowercase
            return ''.join(random.choice(chars) for _ in range(size))
        +string_generator()
        '''
        self.modelfilename     = self.args( name='modelfilename', value=self.modelname+'.h5',
        choices=None, help='Name of h5 file',overwrite=True)
         
        self.transfer_weights  = self.args( name='transfer_weights', value=None,
        choices='null, model.h5, mobilenet.h5', help='Name of h5 file to copy from')
         
         
        self.batch_size        = self.args( name='batch_size', value=16,
        choices=None, help='Batch Size')
        
        self.pooling_layer= self.args( name='pooling_layer', value='GlobalAvgPool',
        choices='GlobalAvgPool, GlobalMaxPool, Flatten', help='Specify Pooling Layer')
         
        self.freeze            = self.args( name='freeze', value=True,
        choices='true,false', help='true for freezing , false for unfreezing')
        self.freezedTrain = self.freeze
         
        self.num_unfreeze      = self.args( name='num_unfreeze', value=5,
        choices='5,6,7', help='number of layers to freeze')
        
        self.classes           = self.args( name='classes', value=self.classlist,
        choices=None, help='Class List')
        print('classes: ', self.classes)
        
        self.num_outnodes      = self.args( name='num_outnodes', value=len(self.classes),
        choices=None, help='Total number of nodes in the final Dense Layer')

        self.modelfile         = osp.join(self.wtpath, self.modelfilename)
        
        
        self.model_obj   = MODEL(inp_dim=self.inp_dim,
            modelname=self.modelname,
            freeze=self.freeze, 
            num_unfreeze=self.num_unfreeze, 
            pooling_layer=self.pooling_layer,
            dropout=self.drop_outs, 
            fc_layers=self.FC_LAYERS, 
            num_outnodes=self.num_outnodes, 
            mode=self.mode,
            load_prev_weights=self.load_prev_weights
         )

        self.model, self.preprocessing_function = self.model_obj.get_model()
        
        self.SetLossFunction()
        self.SetOptimizer()
        self.compileModel()
        
        #import pdb;pdb.set_trace()
        
        #self.model.summary()
        
        if len(self.inp_dim)>=3:
            self.img_height, self.img_width, self.ch = self.inp_dim[-3:]
        
        self.session  = K.get_session()
        self.graph    = tf.get_default_graph()
        #import pdb;pdb.set_trace()

        create_directory_safe(self.wtpath)

        if self.transfer_weights is not None:
             
            self.transfer_weights_path = osp.join(self.root_path, 'weights', self.dbname)
            self.transfer_weights = osp.join(self.transfer_weights_path, self.transfer_weights)
            #if osp.isfile(self.transfer_weights):
            copy_weights(curr_model = self.model, transfer_weights_path = self.transfer_weights, till_dense=True)
            self.freezedTrain=True
            print(self.transfer_weights,' copy_weights done')
        else:
        
            if self.load_prev_weights:
                try:
                    if not osp.exists(self.modelfile):
                        print('Check model file path!', self.modelfile)
                        sys.exit()

                    self.model.load_weights(self.modelfile)
                    #self.model = keras.models.load_model(self.modelfile)
                    print('Loaded previous weights..!')
                except:
                    try:
                        print('loading prev weights failed, trying to copy')
                        copy_weights(curr_model = self.model, transfer_weights_path = self.modelfile, till_dense=True)
                        print(self.modelfile,' copy_weights done')
                    except:
                        print('failed to copy any weights, pretraining with cifar')
                        self.pretrain()


    def clone_pretrained_weights(self, wtpath):
        if not os.path.exists(wtpath):
            print('Selected attire_model!, cloning weights..now..')
            try:
                os.system("git clone http://10.201.0.12:9999/ml/weights "+wtpath + " -b " + 'attire_siamese' + " --single-branch --depth=1")
            except:
                print('Unable to clone ', wtpath)	

    def SetLossFunction(self):
        self.loss = 'categorical_crossentropy'
    
    def SetOptimizer(self):
        #self.opt = Adam(lr=0.002)
        self.opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
        
    def compileModel(self):
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['accuracy'])

    def SetCallbacks(self, epochs):
        from model_checkpoint.Custom_ModelCheckpoint import CustomModelCheckpoint
        if self.freezedTrain is True: #initial stage-> train accuracy, later -> val los.
            monitor='acc'
        else:
            monitor= 'acc_val_loss'#'val_loss' #  'val_acc', 'val_loss'
            
        checkpoint = CustomModelCheckpoint(dbname=self.dbname, modelname=self.modelfilename, verbose=1, mode=monitor, num_epochs=epochs)
        if self.show_plot:
            plot_losses = PlotLosses()
            return [checkpoint, plot_losses]
        else:
            return [checkpoint]
    
    def pre_process(self, img):
        if self.ch==1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_width, self.img_height)) #, interpolation=cv2.INTER_NEAREST
        
            
        if self.verify_data:
            cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
            cv2.imshow('Image', img)
            k=cv2.waitKey(0)
            if k==27:
                sys.exit()
        
        if not self.preprocessing_function is None and self.ch==3:
            img =  self.preprocessing_function(img)
            img = img.astype(np.float32)
        else:
            img = img.astype(np.float32)
            img = img / 255.
            
        if self.ch == 1:
            img =np.expand_dims(img,axis=2)
        return img
        
    def pretrain(self):
        from cifar_train import pretrain_cifar
        #pretrain_cifar(self.model,self.inp_dim[0],self.inp_dim[1])
        copy_weights(curr_model = self.model, transfer_weights_path = 'cifar_model.h5', till_dense=True)

    def train_model(self, epochs=20):
        from generator.data_generator import GetGeneratorObject
        from model_checkpoint.Custom_ModelCheckpoint import CustomModelCheckpoint
        
        self.callbacks = self.SetCallbacks(epochs)
            
        self.tr_dir = osp.join(self.dbname+"_DB", self.dbname , 'train') # Train directory
        self.val_dir = osp.join(self.dbname+"_DB", self.dbname + 'val') # Val directory
        
        # Creating train generator with custom data generator fn
        DataGenerator = GetGeneratorObject(self.generator_type)

        train_generator = DataGenerator(datamode = 'train',
            dbname=self.dbname,
            inp_dim=self.inp_dim,
            batch_size=self.batch_size,
            augmentation=self.augm, 
            pre_process=self.pre_process, 
            shuffle=True, 
            verify_data=self.verify_data,
            classes = self.classes)

        # Creating val generator with custom data generator fn
        validation_generator = DataGenerator(datamode = 'val',
            dbname=self.dbname,
            inp_dim=self.inp_dim,
            batch_size=self.batch_size,
            augmentation=None, 
            pre_process=self.pre_process, 
            shuffle=True, 
            verify_data=self.verify_data,
            classes = self.classes)
            
        if self.verify_data:
            num_workers = 0 
            multiprocessing_flag = False 
        else:
            num_workers = 3
            multiprocessing_flag = False
        #self.freezedTrain = self.freeze and self.preprocessing_function != None and self.ch==3 and (not(self.load_prev_weights))
        #self.freezedTrain = self.freeze
        

        if self.freezedTrain:
            
            #for layer in self.model.layers[:len(self.model.layers)-self.num_unfreeze]:
            #	layer.trainable = False
            #print('freezed ',len(self.model.layers)-self.num_unfreeze, 'layers, making freeze flag false in json now')
            
            with self.session.as_default():
                with self.graph.as_default():
                    self.model.fit_generator(
                    train_generator,
                    epochs = 30,
                    validation_data = validation_generator,
                    callbacks=self.callbacks,
                    class_weight= train_generator.GetClassWeights(),
                    workers=num_workers, use_multiprocessing = multiprocessing_flag)
        
            K.clear_session()
            self.freezedTrain=False
            

        for layer in self.model.layers:
             layer.trainable = True
            
        with self.session.as_default():
                with self.graph.as_default():
                    self.model.fit_generator(
                    train_generator,
                    epochs = epochs,
                    validation_data = validation_generator,
                    callbacks=self.callbacks,
                    class_weight= train_generator.GetClassWeights(),
                    workers=num_workers, use_multiprocessing = multiprocessing_flag) 
        
        K.clear_session()
        
            
        print('#### changing load previous model value to true in config if false ###')
        self.args( name='load_prev_weights', value=True, choices='True, False', help='Train from scratch/Continue training Mode',overwrite=True)
        print('#### changing freeze value to false in config if true###')
        self.args( name='freeze', value=False, choices='true,false', help='true for freezing , false for unfreezing',overwrite=True)
        self.transfer_weights  = self.args( name='transfer_weights', value=None, choices='model.h5, mobilenet.h5', help='Name of h5 file to copy from',overwrite=True)

    def Predict(self, mat):
        assert self.load_prev_weights is True
        img1= self.pre_process(mat)
        img1 = np.expand_dims(img1, axis=0)
        #import pdb;pdb.set_trace()
        K.clear_session()
        with self.session.as_default():
            with self.graph.as_default():
                probs = self.model.predict(img1).tolist()[0]
        K.clear_session()
        #print('probs: ', probs)
        class_index = np.argmax(probs)
        label = self.classes[class_index]
        return class_index, label

    def batch_predict(self, cv_arr):
        cv_arr2 = []
        for i in range(len(cv_arr)):
            img1= cv_arr[i]
            img1= self.pre_process(img1)
            cv_arr2.append(img1)
        img_arr= np.array(cv_arr2)
        img_arr = np.expand_dims(img_arr, axis=0)
       
        #K.clear_session()
        with self.session.as_default():
            with self.graph.as_default():
                class_probabilities = self.model.predict_on_batch(img_arr).tolist()
        #K.clear_session()
        predictions = [np.argmax(i) for i in class_probabilities]
        labels = [self.classes[p] for p in predictions]
        return predictions, labels  
    
    def predict_from_buffer(self, bufferlist):
        predictions, labels = self.batch_predict(bufferlist)
        return predictions, labels
        
    def validate(self, folder = 'val',show_flag=True):
        dir_ = osp.join(self.DB_path, folder)
        allfiles = glob.glob(osp.join(dir_, '*', '*'), recursive=True) #other exts TOTO
        good_count=0
        y_true=[]
        y_pred=[]
        for f in allfiles:
            try:
                img= cv2.imread(f)
                class_index, predicted_label = self.Predict(img)
                classname = os.path.split(f)[0].split('/')[-1]
                
                if show_flag:
                    cv2.imshow('img',img)
                    k=cv2.waitKey(1)
                    if k==27:
                        break
                    print('Actual: ', classname, 'Predicted: ', predicted_label)
                    
                y_pred.append(predicted_label)
                y_true.append(classname)
                
                if str(predicted_label)==str(classname):
                    good_count+=1
                else:
                    if 1:
                        cv2.imshow('img',img)
                        print('Actual: ', classname, 'Predicted: ', predicted_label)
                        print('Press enter, c to move image to confused folder, esc to quit')
                        k=cv2.waitKey(0)
                        if k==ord('c'):
                            #import pdb;pdb.set_trace()
                            import shutil 
                            dir_confused = osp.join(self.DB_path, 'confused')
                            mkdir_safe(dir_confused)
                            shutil.move(f,dir_confused)
                        elif k==27:
                            break
            except:
                print(f, 'not an image')
        acc_matrics(y_true, y_pred,self.classes)

    def __del__(self):
        print('Destructor called')


class CNNCategorical(DEEP_NN):
    def __init__(self, dbname='flki_attire_dataset'):
        self.args = ArgsManager(dbname=dbname, currentPath=path)
        super().__init__(dbname, self.args)


class Grex_categorical(DEEP_NN):
    def __init__(self, dbname='flki_attire_dataset'):
        self.args = ArgsManager(dbname=dbname, currentPath=path)
        print("Entering Grex_categorical..........")
        super().__init__(dbname, self.args)

    def contrast_stretch(self,image):
        r1 = 120
        s1 = 0
        r2 = 140
        s2 = 255
        #contrast_stretched = np.zeros(image.shape).astype(np.uint8)
        # Vectorize the function to apply  it to each value in the Numpy array.
        contrast_stretched = (image - r1)/(r2 - r1)*(s2-s1) + s1
        return np.array(contrast_stretched)

    def equalizeHist(self, img):
        img = img.astype(np.uint8)
        img = cv2.equalizeHist(img).reshape(img.shape)
        return img

    def enhance_and_stack(self, org_feature, as_list = False):
        feature1 = np.array(self.contrast_stretch(org_feature))
        feature2 = np.array(self.equalizeHist(org_feature))
        if as_list:
            feature = [org_feature, feature1, feature2]
        else:
            feature = np.hstack([org_feature, feature1, feature2])
        return feature    

    def pre_process(self, img):
        img = cv2.resize(img, (int(self.img_width/3), self.img_height), interpolation=cv2.INTER_NEAREST)
        if self.ch==1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if self.verify_data:
            cv2.imshow('Image', img)
            cv2.waitKey(30)
    
        img = self.enhance_and_stack(img)
        if not self.preprocessing_function is None and self.ch==3:
            img =  self.preprocessing_function(img)
            img = img.astype(np.float32)
        else:
            img = img.astype(np.float32)
            img = img / 255.			
        if self.ch == 1:
            img =np.expand_dims(img,axis=2)
        return img		  



class CNNImageBinary(DEEP_NN):
    
    def __init__(self, dbname='sample_binary_gear'):
        print('###############################inside child init#######################')
        self.args = ArgsManager(dbname=dbname, currentPath=path)
        self.modelname         = self.args( name='model', value='RX_model_digit', choices='mobilenet, RX_model_gear, RX_model_digit,mobilenetv2, customcnn, customLSTM, ...', help='Available Network Models')
        self.mode              = self.args( name='class_mode', value='binary', 
        choices='regression, categorical, binary', help='Statistical Data Type')
        self.generator_type    = self.args( name='datagenerator', value='IMAGEGENERATOR_Binary', choices='IMAGEGENERATOR_Binary, REGIMAGEGENERATOR, LSTMGENERATOR, IMAGEGENERATOR', help='Generator Type')
        self.num_outnodes      = self.args( name='num_outnodes', value=1,
        choices=None, help='Total number of nodes in the final Dense Layer')
        super().__init__(dbname=dbname, args=self.args)
    
    def SetLossFunction(self):
        self.loss = 'binary_crossentropy'

    def Predict(self, mat):
        assert self.load_prev_weights is True
        img1= self.pre_process(mat)
        img1 = np.expand_dims(img1, axis=0)
        #K.clear_session()
        with self.session.as_default():
            with self.graph.as_default():
                probs = self.model.predict(img1).tolist()[0]
        #K.clear_session()
        prediction = probs[0]
        class_index = int(np.round(probs[0]))
        label = self.classes[class_index]
        return prediction, label

class CNNIRegression(DEEP_NN):
    def __init__(self, dbname='mnist2'):
        self.args = ArgsManager(dbname=dbname, currentPath=path)
        self.generator_type    = self.args( name='datagenerator', value='REGIMAGEGENERATOR', choices='REGIMAGEGENERATOR, LSTMGENERATOR, IMAGEGENERATOR', help='Generator Type')
        self.num_outnodes      = self.args( name='num_outnodes', value=1, choices=None, help='Total number of nodes in the final Dense Layer')
        self.mode              = self.args( name='class_mode', value='regression', choices='regression, categorical, binary', help='Statistical Data Type')
        super().__init__(dbname, self.args)

    def SetLossFunction(self):
        self.loss = 'mean_squared_error' #'mean_squared_error', 'mean_absolute_error'
    def SetOptimizer(self):
        self.opt=Adam(lr=0.0002)
        
    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.opt, metrics=['mae','accuracy'])
    
    def predict(self, mat):
        assert self.load_prev_weights is True
        img1= self.pre_process(mat)
        img1 = np.expand_dims(img1, axis=0)
        # K.clear_session()
        # with self.session.as_default():
        #     with self.graph.as_default():
        probs = self.model.predict(img1).tolist()[0]
        # K.clear_session()
        prediction = int(np.round(probs)[0])
        class_index = int(np.round(probs[0]))
        return prediction, class_index

    def validate(self):
        self.load_prev_weights = True#self.args( name='load_prev_weights', value=True, choices='True, False', help='Train from scratch/Continue training Mode')
        dir_ = osp.join(self.DB_path, 'val')
        allfiles = glob.glob(osp.join(dir_, '*', '*'), recursive=True) #other exts TOTO

        good_count=0
        for f in allfiles:
            img= cv2.imread(f)
            class_index, predicted_label = self.Predict(img)
            classname = f.split('/')[-2]
            print('Predicted: ', predicted_label, 'Actual: ', classname)
            if str(predicted_label)==str(classname):
                good_count+=1
        print('Val Acc: ', good_count/len(allfiles) * 100)


class Audio_categorical(DEEP_NN):
    def __init__(self, dbname='sample_audio'):
        self.args              = ArgsManager(dbname=dbname, currentPath=path)
        self.modelname         = self.args( name='model', value='Siamese1D', choices='mobilenet, RX_model_gear, RX_model_digit,mobilenetv2, customcnn, customLSTM, ...', help='Available Network Models')
        self.generator_type    = self.args( name='datagenerator', value='AudioDataGenerator', choices='IMAGEGENERATOR_Binary, REGIMAGEGENERATOR, LSTMGENERATOR, IMAGEGENERATOR', help='Generator Type')
        self.inp_dim           = self.args( name='inp_dim', value=(8000, 1), 
        choices=None, help='Input Dimension')
        self.pooling_layer= self.args( name='pooling_layer', value='Flatten',
        choices='GlobalAvgPool, GlobalMaxPool, Flatten, None', help='Specify Pooling Layer')
        self.augm              = self.args( name='augment_type', value='simple_audio_aug',
        choices='simple_albumentaion, strong_albumentaion, sample_opencv_aug', 
        help='Type of Augmentation')
        self.verify_data      = self.args( name='verify_data', value=False,
        choices='true, false', help='Enable/Disable imshow')
        super().__init__(dbname=dbname, args=self.args)
        self.sample_rate      = 8000
    
    def pre_process(self, data):
        if len(data) < self.sample_rate:
            requid_nsamples = self.sample_rate - len(data)
            left = int(random.randrange(0, requid_nsamples+1))
            right = int(requid_nsamples-left)
            data = np.pad(data, (left, right), 'constant', constant_values=(0, 0))
        else:
            data = signal.resample(data, self.sample_rate)
        if self.verify_data:
            wavfile.write(osp.join(self.DB_path, 'temp.wav'), self.sample_rate, data)
            playsound(osp.join(self.DB_path, 'temp.wav'))
        return data

    def validate(self):
        dir_ = osp.join(self.DB_path, 'val')
        allfiles = glob.glob(osp.join(dir_, '*', '*'), recursive=True) #other exts TOTO
        good_count=0
        for f in allfiles:
            print('File: ', f)
            aud, __= ReadAudio(f)
            class_index, predicted_label = self.Predict(aud)
            classname = os.path.split(f)[0].split('/')[-1]
            print('Predicted: ', predicted_label, 'Actual: ', classname)

            if str(predicted_label)==str(classname):
                good_count+=1
        print('Val Acc: ', good_count/len(allfiles) * 100, '%')

    def Predict(self, audio):
        feature_vec = self.pre_process(audio)
        print('feature_vec Shape; ', feature_vec.shape)
        # feature_vec = feature_vec.T
        feature_vec = feature_vec.reshape(1, self.sample_rate, 1)
        # K.clear_session()
        # with self.session.as_default():
        # 	with self.graph.as_default():
        probs = self.model.predict(feature_vec)
        # K.clear_session()
        class_index = np.argmax(probs)
        label = self.classes[class_index]
        return class_index, label
    
    def LiveTest(self, num_cycle=10):
        for i in range(num_cycle):
            print('Available Classes: ', self.classes)
            __, audio = record_audio(duration=2, sr=12000)
            __, label = self.Predict(audio)
            print('Prediction: ', label)
            time.sleep(1)
            #playsound(filename)

class CNNSiamese(DEEP_NN):
    def __init__(self, dbname='flki_attire_dataset'):
        self.args = ArgsManager(dbname=dbname, currentPath=path)
        self.generator_type    = self.args( name='datagenerator', value='SIAMESEGENERATOR2D', choices='REGIMAGEGENERATOR, LSTMGENERATOR, IMAGEGENERATOR', help='Generator Type')
        self.modelname         = self.args( name='model', value='Siamese2D', choices='mobilenet, mobilenetv2, customcnn, customLSTM, ...', help='Available Network Models')
        self.pooling_layer= self.args( name='pooling_layer', value=None,
        choices='GlobalAvgPool, GlobalMaxPool, Flatten', help='Specify Pooling Layer')
        
        super().__init__(dbname, self.args)
        self.opt = RMSprop()
        
        
        
    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def SetLossFunction(self):
        self.loss = self.contrastive_loss
        
    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    def compile(self):
        self.model.compile(loss=self.contrastive_loss, optimizer=self.opt, metrics=[self.accuracy])

    def Predict(self, mat1, mat2):
        assert self.load_prev_weights is True
        img1= self.pre_process(mat1)
        img2= self.pre_process(mat2)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        # K.clear_session()
        # with self.session.as_default():
        #     with self.graph.as_default():
        X = [img1, img2]
        #import pdb;pdb.set_trace()
        probs = self.model.predict(X).tolist()[0][0]
        print('probs: ', probs)
        # K.clear_session()
        prediction = 1 if probs<0.5 else 0
        class_index = prediction
        return prediction, class_index

    def validate(self):
        self.load_prev_weights = True
        dir_ = osp.join(self.dbname+'_DB', self.dbname, 'val')
        allfiles = glob.glob(osp.join(dir_, '*', '*'), recursive=True) #other exts TOTO
        allfiles = allfiles[:100]
        combinations = list(itertools.combinations(allfiles, 2))
        good_count=0
        
        for imp1, imp2 in combinations:
            img1= cv2.imread(imp1)
            img2= cv2.imread(imp2)
            class_index, predicted_label = self.Predict(img1, img2)
            classname1 = osp.split(imp1)[0].split('/')[-1]
            classname2 = osp.split(imp2)[0].split('/')[-1]
            val = 1 if classname1==classname2 else 0
            print('Predicted: ', predicted_label, 'Actual: ', val)
            if predicted_label==val:
                good_count+=1
            
        print('Val Acc: ', good_count/len(combinations) * 100)


class AudioSiamese(DEEP_NN):
    def __init__(self, dbname='sample_audio'):
        self.args              = ArgsManager(dbname=dbname, currentPath=path)
        self.modelname         = self.args( name='model', value='Siamese1D', choices='mobilenet, RX_model_gear, RX_model_digit,mobilenetv2, customcnn, customLSTM, ...', help='Available Network Models')
        self.generator_type    = self.args( name='datagenerator', value='SIAMESEGENERATOR1D', choices='IMAGEGENERATOR_Binary, REGIMAGEGENERATOR, LSTMGENERATOR, IMAGEGENERATOR', help='Generator Type')
        self.inp_dim           = self.args( name='inp_dim', value=(8000, 1), 
        choices=None, help='Input Dimension')
        self.pooling_layer= self.args( name='pooling_layer', value='Flatten',
        choices='GlobalAvgPool, GlobalMaxPool, Flatten, None', help='Specify Pooling Layer')
        self.augm              = self.args( name='augment_type', value='simple_audio_aug',
        choices='simple_albumentaion, strong_albumentaion, sample_opencv_aug', 
        help='Type of Augmentation')
        self.verify_data      = self.args( name='verify_data', value=False,
        choices='true, false', help='Enable/Disable imshow')
        super().__init__(dbname=dbname, args=self.args)
        self.sample_rate      = 8000
        self.opt = RMSprop()
        
    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def SetLossFunction(self):
        self.loss = self.contrastive_loss
        
    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    def compile(self):
        self.model.compile(loss=self.contrastive_loss, optimizer=self.opt, metrics=[self.accuracy])

    def Predict(self, audio1, audio2):
        feature_vec1 = self.pre_process(audio1).reshape(1, self.sample_rate, 1)
        feature_vec2 = self.pre_process(audio2).reshape(1, self.sample_rate, 1)
        # K.clear_session()
        # with self.session.as_default():
        # 	with self.graph.as_default():
        X = [feature_vec1, feature_vec2]
        probs = self.model.predict(X)
        # K.clear_session()
        prediction = 1 if probs<0.5 else 0
        class_index = prediction
        return class_index, prediction

    def pre_process(self, data):
        if len(data) < self.sample_rate:
            requid_nsamples = self.sample_rate - len(data)
            left = int(random.randrange(0, requid_nsamples+1))
            right = int(requid_nsamples-left)
            data = np.pad(data, (left, right), 'constant', constant_values=(0, 0))
        else:
            data = signal.resample(data, self.sample_rate)
        if self.verify_data:
            wavfile.write(osp.join(self.DB_path, 'temp.wav'), self.sample_rate, data)
            playsound(osp.join(self.DB_path, 'temp.wav'))
        return data

    def validate(self):
        dir_ = osp.join(self.DB_path, 'val')
        allfiles = glob.glob(osp.join(dir_, '*', '*'), recursive=True) #other exts TOTO
        combinations = list(itertools.combinations(allfiles, 2))
        good_count=0
        for aud1, aud2 in combinations:
            audiosample1, __= ReadAudio(aud1)
            audiosample2, __= ReadAudio(aud2)
            class_index, predicted_label = self.Predict(audiosample1, audiosample2)
            classname1 = os.path.split(aud1)[0].split('/')[-1]
            classname2 = os.path.split(aud2)[0].split('/')[-1]
            val = 1 if classname1==classname2 else 0
            print('Predicted: ', predicted_label, 'Actual: ', val)
            if str(predicted_label)==str(classname):
                good_count+=1
        print('Val Acc: ', good_count/len(allfiles) * 100, '%')
