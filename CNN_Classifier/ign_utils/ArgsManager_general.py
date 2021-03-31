import os
import sys
import os.path as osp
path = osp.dirname(osp.abspath(__file__))
sys.path.append(path)

import json

class ArgsManager():
    def __init__(self, jsonpath=None):
        self.path = osp.dirname(osp.abspath(__file__))
        if jsonpath is not None:
            self.jsonpath = jsonpath
        else:
            self.jsonpath = osp.join(self.path)
            self.jsonpath = osp.join(self.jsonpath, '..', 'configuration.json')
        
        self.config={}
        if osp.exists(self.jsonpath):
            print(self.jsonpath, \
            '\nConfig json exists- values are read from json,\
            except for keys with overwrite mode "True"')
            with open(self.jsonpath, 'r') as config_json:
                self.config = json.load(config_json)
            
    def write_config(self):
        with open(self.jsonpath,'w') as config_json:
            json.dump(self.config, config_json, indent=2) 
                
    def checkkey(self,key):
        if key in self.config:
            return True
        else:
            return False
                    
    def __call__(self, name, value=None, choices=None, help=None, overwrite=False):       
        if self.checkkey(name) == False:
            self.config[name]={'value' : value, 'choices':choices, 'help':help}
            self.write_config()
        elif overwrite is True:
            self.overwrite(name,value)
        return self.config[name]['value']
        
    def overwrite(self,name,value):
        if self.checkkey(name) == True:
            if self.config[name]['value'] != value:
                print( 'Config json overwrite,',name, 'from:',self.config[name]['value'], 'to:',value)
                self.config[name]['value']=value
                self.write_config()
                
        else:
            print(name, 'not present in config')
        
if __name__ == '__main__':    
    #usage example
    # availableModes = 'detect_mode', 'test_mode', 'label_mode'
    args=ArgsManager() 
    mode_selected = 'test_mode'
    submode_selected = 'jump_through_labels'
    UImode_selected  = True
    debugmode_selected = False
    csv_file_selected = 'JBResults_0.csv'
    k1 = args( name='mode', value=mode_selected, choices='detect_mode, test_mode, label_mode', help='App Mode')
    k2 = args( name='submode', value=submode_selected, choices='jump_through_labels, go_through_all_frames', help='Test through all frames/Test through labelled frames only')
    k3 = args( name='UImode', value=UImode_selected, choices='true, false', help='Enable UI Mode or Terminal only Mode')
    k4 = args( name='debugmode', value=debugmode_selected, choices='true, false', help='Enable Debug')
    k5 = args( name='csvfile', value=csv_file_selected, choices='JBResults_0.csv, JBResults_1.csv, ...', help='Provide filename to write results')
    import pdb;pdb.set_trace()