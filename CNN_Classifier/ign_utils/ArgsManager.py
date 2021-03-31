import os
import sys
import os.path as osp
path = osp.dirname(osp.abspath(__file__))
sys.path.append(path)

import json
from general_utils import appendLineSafe, create_directory_safe

class ArgsManager():
    def __init__(self, dbname, currentPath=None):
        
        self.path = osp.dirname(osp.abspath(__file__))
        if currentPath:
            self.root_path = currentPath
        else:
            self.root_path = osp.join(self.path, '..')
        self.DB_path = osp.join(self.root_path,'DB', dbname)
        self.weights_path = osp.join(self.root_path,'weights', dbname)
        
        print('weights_path: ', self.weights_path)
        print('db: ', osp.join(self.DB_path, 'train'))
        
        
        self.make_DB_weights(dbname)

        if not os.path.exists(self.weights_path):
            raise Exception(self.weights_path, "doesn't exist, please create folders\
            / clone weights-repo(http://10.201.0.12:9999/ml/weights) with branch: ",dbname)
            
        self.config={}
        self.config_path = osp.join(self.weights_path, 'config.json')
        
        if os.path.exists(self.config_path):
            print(self.config_path, \
            '\nConfig json exists- values are read from json,\
            except for keys with overwrite mode "True"')
            with open(self.config_path, 'r') as config_json:
                self.config = json.load(config_json)
        else:
            #import pdb;pdb.set_trace()
            #self.classes = sorted(os.listdir(osp.join(self.DB_path, 'train')))
            #print('self.classes: **********', self.classes)
            self.config={}
            #self.config['classes']={'value' : self.classes, 'choices':None, 'help':'Class List'}
            self.__call__(name='dbname', value=dbname, choices='plate, gear', help='name of db')
            
    def make_DB_weights(self,dbname):
        try:#DB
            if not os.path.exists(self.DB_path):
                os.system("git clone http://10.201.0.12:9999/ml/DB "+self.DB_path
                 + " -b " + dbname + " --single-branch --depth=1")
        except:
            # create_directory_safe(osp.join(self.DB_path, 'train'))
            # create_directory_safe(osp.join(self.DB_path, 'val' ))
            print('Add to repo: http://10.201.0.12:9999/ml/DB, \
            check https://stackoverflow.com/questions/51123925/how-can-i-git-init-in-existing-folder')
           
        try:#weights
            if not os.path.exists(self.weights_path):    
                os.system("git clone http://10.201.0.12:9999/ml/weights "+self.weights_path
                 + " -b " + dbname + " --single-branch --depth=1")
        except:
            # create_directory_safe(self.weights_path)
            print('Add to repo: http://10.201.0.12:9999/ml/weights, \
            check https://stackoverflow.com/questions/51123925/how-can-i-git-init-in-existing-folder')
            
        #appendLineSafe(osp.join(self.root_path, '.gitignore'), dbname+"_DB")
        #appendLineSafe(osp.join(self.root_path, '.gitignore'), dbname+"_weights")

    def write_config(self):
        #saving config to json file for loading later

        with open(self.config_path,'w') as config_json:
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
        #import pdb;pdb.set_trace()
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
    class Myclass: 
        def __init__(self,dbname_):
            args=ArgsManager(dbname=dbname_) 
            
            self.k1=args( name='dbname', value=dbname_, choices='plate, rtt', help='name of db')

            self.k2=args( name='datamode', value='audio', choices='audio, image, video', help='input data type')

            self.k3=args( name='fc_layers', value=[32,16], choices=None, help='list of fully connected layers, except last layer')
            args.overwrite(name='datamode', value='video')
            
            self.func(args)
            
            
        def func(self,args): #won't prefer this, but if its inevitable pass args into functions
            self.k4 = args(name='something', value='someval', choices='someval,someval2', help='some options')
            
    obj = Myclass(dbname_ = 'mnist_regress')
            
    #################ToDO ###############
    '''
    import re
    import traceback
    def func(var):
        a='abc'
        stack = traceback.extract_stack()
        filename, lineno, function_name, code = stack[-2]
        vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
        print (vars_name)
        return vars_name, var
    k = "foo"
    vars_name,val=func(k)
    print ('vars_name',vars_name, 'value', val)
'''



   
