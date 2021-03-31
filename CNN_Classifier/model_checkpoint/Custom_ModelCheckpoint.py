import keras
import os.path as osp
from ign_utils.general_utils import read_json, write_json
import numpy as np
import warnings

class CustomModelCheckpoint(keras.callbacks.Callback):
    '''
    Custom model accuracy;
    Callback check both training and validation accuracy together, 
    and save model when both accuracies increase.\n
    '''
    def __init__(self, dbname=None, modelname=None, verbose=0, mode='acc', num_epochs=None):
        self.path = osp.dirname(osp.abspath(__file__))
        self.root_path = osp.join(self.path, '..')
        self.DB_path = osp.join(self.root_path,'DB', dbname)
        self.weights_path = osp.join(self.root_path,'weights', dbname)
        self.threshold_acc = 0.005
        self.num_epochs = num_epochs
        self.acc = 0
        self.val_acc = 0
        self.loss = np.Inf
        self.val_loss = np.Inf
        self.LogInfo = None
        self.verbose = verbose
        self.mode = mode
        self.jsonpath = osp.join(self.weights_path, 'logInfo.json')
        self.model_path = osp.join(self.weights_path, modelname)
        if osp.isfile(self.jsonpath) and self.mode=='val_loss':
            LogInfo = self.Read_Metric()
            self.acc = LogInfo['acc']
            self.val_acc = LogInfo['val_acc']
            self.loss = LogInfo['loss']
            self.val_loss = LogInfo['val_loss']
        
    def Read_Metric(self):
        config = read_json(self.jsonpath)
        return config
    
    def Write_Metric(self, jsonInfo):
        write_json(self.jsonpath, jsonInfo)
    
    def on_last_epoch(self, logs):
        '''
        Save model after last epoch if no models are saved
        '''
        #import pdb;pdb.set_trace()
        if not osp.isfile(self.model_path):
            self.model.save(self.model_path)
        
        
    def on_epoch_end(self, epoch, logs={}):
        current_acc = logs.get('acc')
        current_val_acc = logs.get('val_acc')
        current_loss = logs.get('loss')
        current_val_loss = logs.get('val_loss')
        
        if current_acc is None or current_val_acc is None:
            warnings.warn('current_acc is None or current_val_acc is None. Can save best model only ...', RuntimeWarning)
        else:
            #if self.mode=='acc_val_loss':
            if ((current_acc + current_val_acc) >= (self.acc + self.val_acc)) or \
               ((current_loss + current_val_loss) <= (self.loss + self.val_loss)) and \
               ((current_acc + current_val_acc) >= (self.acc + self.val_acc) - self.threshold_acc):
                
                if self.verbose > 0:
                    if ((current_acc + current_val_acc) >= (self.acc + self.val_acc)):
                        print('\nSaving model\n acc improved: train acc %0.5f, val acc %0.5f, \
                                                            \ntrain loss %0.5f, val loss %0.5f , \n%s\n'\
                                                             %(current_acc  , current_val_acc,\
                                                               current_loss , current_val_loss,  self.model_path))
                        
                        self.acc = current_acc
                        self.val_acc = current_val_acc
                    else:
                        print('\nSaving model\n loss improved:  train acc %0.5f, val acc %0.5f, \
                                                            \ntrain loss %0.5f, val loss %0.5f , \n%s\n'\
                                                             %(current_acc  , current_val_acc,\
                                                               current_loss , current_val_loss,  self.model_path))
                self.loss = current_loss
                self.val_loss = current_val_loss
                        
                self.model.save(self.model_path)
                
                try:
                    self.Write_Metric(logs) # Dump json config with last log info(acc, loss, val_acc, val_loss)
                except TypeError:
                    logs = {k:float(np.array(v).astype(np.float64)) for k, v in logs.items()}
                    self.Write_Metric(logs)
            
            else:
                if self.verbose > 0:
                    print() 

                    print('acc did not improved from %0.5f+ %0.5f'  \
                        %(self.acc , self.val_acc), \
                        'loss did not improved from %0.5f+ %0.5f '\
                        %(self.loss , self.val_loss))
                    
        if epoch+1>=self.num_epochs :
            #import pdb;pdb.set_trace()
            self.on_last_epoch(logs)

            
