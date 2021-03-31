import os
import re
import csv
import ast
import math
import json
import glob
import shutil
import random
import itertools
import numpy as np
import os.path as osp
from pathlib import Path
from subprocess import call

def reorder_list(lukup_list, keyword='unknown'):
    key_index = None
    lukup_list = sorted(lukup_list)
    if keyword in lukup_list:
        lukup_list.remove(keyword)
        lukup_list.append(keyword)
        key_index = len(lukup_list) - 1
    return lukup_list, key_index

def get_filenames(folder_path, ext, sort_ = False):
    folder = Path(folder_path)
    ext = '*.{}'.format(ext)
    file_list = list(folder.rglob(ext))
    file_list = [str(fn) for fn in file_list]
    if sort_:
        file_list = sorted(file_list)
    return file_list

def remove_file(filename, destination='temp'):
    '''
    mkdir_safe(destination)
    try:
        dest = shutil.move(filename,destination) 
        print('Moved ',filename, 'to', destination)
    except:
    '''
    #import pdb;pdb.set_trace()
    if osp.isfile(filename):
        os.remove(filename)
        print('Deleted ',filename)
        return True
    else:
        print('File {} not exists'.format(filename))
        return False
    
def remove_ifempty(dirName):
    if os.path.exists(dirName) and os.path.isdir(dirName):
        if not os.listdir(dirName):
            print("Directory is empty, deleting")
            os.rmdir(dirName)
        else:    
            print("Directory is not empty")
    else:
        print("Given Directory don't exists")     
            
    
def remove_folder(folderpath):
    try:
        shutil.rmtree(folderpath)
    except:
        print('{} does not exists'.format(folderpath))
        pass
        
def mkdir_safe(direc):
    return create_directory_safe(direc)

def create_directory_safe(direc):
    if not os.path.exists(direc):
        os.makedirs(direc)
        return True
    else:
        return False

def CheckDirectoryEmpty(folderpath):
	if osp.isdir(folderpath):
		if len(os.listdir(folderpath)) > 0:
			return False
		else:
			return True
	else:
		create_directory_safe(folderpath)
		return True

def get_timestamp():
    '''
    import calendar;
    import time;
    ts = calendar.timegm(time.gmtime())
    #print(ts)
    return str(ts)
    '''
    from datetime import datetime
    (dt, micro) = datetime.utcnow().strftime('%Y%m%d%H%M%S.%f').split('.')
    dt = "%s%03d" % (dt, int(micro) / 1000)
    return str(dt)

def do_sometimes(n):
    if n < 1:
        return False
    if n==1:
        return True
    rn = random.randint(0, n)
    if rn==0:
        return True
    else:
        return False

def read_from_txt(filepath):
    with open(filepath) as f:
        _list = f.read()
    return _list

def kill_previous_process(processname):
    try:
        pid=np.load(processname + '.npy')
        import signal
        os.kill(pid, signal.SIGKILL)
        print('previous process killed, pid: ',pid)
    except:
        pass
    pid = os.getpid()
    np.save(processname, pid)


def get_frames(src, dst):
    call(["ffmpeg", "-i", src, dst])

def gen_rnd(minv, maxv):
    
    return(random.randint(minv, maxv))

def get_minimum_seq(list_):
    shortest_length = min(map(len, list_))
    #shortest_list = min(list_, key=len)
    return shortest_length

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def remove_directory_safe(folderpath):
    if osp.isdir(folderpath):
        shutil.rmtree(folderpath)
        
def move_file(file_list, dst, max_num=None):
    if not max_num is None:
        file_list = random.sample(file_list, max_num) 
    for fpath in file_list:
        person_name, classname, name =  str(fpath).split('/')[-3:]
        if os.path.isfile(fpath):
            shutil.move(fpath, osp.join(dst, classname, person_name + '_' + name))

def RemoveFile(fpath):
    os.remove(fpath)

def LoadFilenames(dir_, recursive=False):
    x_set = []
    if recursive is True:
        x_dirs = glob.glob(osp.join(dir_, '*', '*'))
        temp_list = []
        for imgp in x_dirs:
            x_set1 = []
            imglist = glob.glob(osp.join(imgp, '*'))
            temp_list.append(imglist)
            shortest_len, _ = get_minimum_seq(temp_list)
            balanced_imglist = random.sample(imglist, shortest_len)
            [RemoveFile(fpath) for fpath in list(set(imglist)-set(balanced_imglist))]
            if len(imglist)>=1: x_set1.extend(balanced_imglist)
            x_set.append(x_set1) 
            if len(temp_list)>2:
                temp_list = []
        return x_set
    else:
        x_dirs = glob.glob(osp.join(dir_, '*'))
        for dir_ in x_dirs:
            files = glob.glob(osp.join(dir_, '*'))
            if len(files)>=1: x_set.append(files)
        return x_set
'''
def LoadFilenamesold(dir_, recursive=False):
    x_set = []
    if recursive is True:
        x_dirs = glob.glob(osp.join(dir_, '*', '*'))
        for dir_ in x_dirs:
            x_set1 = []
            files = glob.glob(osp.join(dir_, '*'))
            if len(files)>=1: x_set1.extend(files)
            x_set.append(x_set1)
        return x_set
    else:
        x_dirs = glob.glob(osp.join(dir_, '*'))
        for dir_ in x_dirs:
            files = glob.glob(osp.join(dir_, '*'))
            if len(files)>=1: x_set.append(files)
        return x_set
'''

def SplitTrainVal(dbname):
        path = osp.dirname(osp.abspath(__file__))
        DBpath = osp.join(path, '..', 'DB', dbname)
        tmpDBPath = osp.join(DBpath, 'temp_DB')
        classes = None
        if osp.isdir(tmpDBPath):
            if len(os.listdir(osp.join(tmpDBPath, os.listdir(tmpDBPath)[0])))>=1:
                classes = sorted(os.listdir(osp.join(tmpDBPath, os.listdir(tmpDBPath)[0])))
        elif osp.isfile(osp.join(DBpath, 'train')):
            if len(os.listdir(osp.join(DBpath, 'train'))) >=1:
                classes = sorted(os.listdir(osp.join(DBpath, 'train')))
        if classes is None:
            classes = ['bad', 'good']
        CustomMove(dbname, true_val=True, classes=classes)
        CustomMove(dbname, classes=classes)
        
def CustomMove(dbname, fraction=0.2, true_val=False, classes=None):
    path = osp.dirname(osp.abspath(__file__))
    DBpath = osp.join(path, '..', 'DB', dbname)
    tmpDBPath = osp.join(DBpath, 'temp_DB')
    
    if osp.isdir(tmpDBPath):
        dstpath_val = osp.join(DBpath, 'val')
        [create_directory_safe(osp.join(dstpath_val, str(item))) for item in classes]

        dstpath_train = osp.join(DBpath, 'train')
        [create_directory_safe(osp.join(dstpath_train, str(item))) for item in classes]
        total_individuals = os.listdir(tmpDBPath)
        print('total_individuals: ', total_individuals)
        if true_val and len(total_individuals) >3:
            val_person_db = random.sample(total_individuals, math.ceil(len(total_individuals)*fraction))
            val_files = LoadFilenames(osp.join(tmpDBPath, str(val_person_db[0])))
            if val_files:
                shortest_len, _ = get_minimum_seq(val_files)
                [move_file(list_, dstpath_val, max_num=shortest_len) for list_ in val_files]
                shutil.rmtree(osp.join(tmpDBPath, str(val_person_db[0])))
            
        else:
            val_files = LoadFilenames(tmpDBPath, recursive=True)
            if val_files:
                for list_ in val_files:
                    val_list = random.sample(list_, math.ceil(len(list_)*fraction))
                    move_file(val_list, dstpath_val)
                train_files = glob.glob(osp.join(tmpDBPath, '*', '**'), recursive=True)
                move_file(train_files, dstpath_train)
                shutil.rmtree(tmpDBPath)
                create_directory_safe(tmpDBPath)
        #print('Move Files Completed..dstpath_train: ', len(os.listdir(dstpath_train)))
    else:
        print('Not a Directory {}'.format(tmpDBPath))

def insert_val_to_list(list_, percentage=0.3, string_val='noise'):
    rc = int(len(list_)*percentage)
    random_indices = [list_.index(i) for i in set(np.random.choice(list_, rc))]
    for i in random_indices:
        list_[i]=string_val
    return list_
    
def write_json(jsonpath, jsondata):
    #saving config to json file for loading later
    with open(jsonpath, 'w') as config_json:
        json.dump(jsondata, config_json, indent=2)

def read_json(jsonfile):
    with open(jsonfile, 'r') as config_json:
        config = json.loads(config_json.read())
    return config
            
def get_roi(frame, rectpoints, x=0, y=0):
    import cv2
    if isinstance(frame, str):
        frame = cv2.imread(frame)
        
    rectpoints = ast.literal_eval(rectpoints)

    x_start = int(rectpoints[0][0] + int(x))
    y_start = int(rectpoints[0][1] + int(y))
    x_end = int(rectpoints[2][0] + int(x))
    y_end = int(rectpoints[2][1] + int(y))
    refPoint = [(x_start, y_start), (x_end, y_end)]
    
    roi_img = frame[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
    return roi_img


def write_text_lines(txtpath, _list):
    with open(txtpath, 'w') as filehandle:
        json.dump(_list, filehandle)
    return True

def read_from_csv(csvfile):
    
    l = []
    if osp.isfile(csvfile):
        with open(csvfile, 'r') as fin:
            data = csv.reader(fin)
            for i in data:
                l.extend(i)
    return l

def appendLineSafe(filename, textline):
    with open(filename, "r+") as file:
        flag=0
        for line in file:
            if textline in line:
                flag=1
                break
        if flag==0: # not found, we are at the eof
            file.write("\n"+textline) # append missing data

def divide_chunks(l, n):
    """
    Break a list into chunks of size n
    """ 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]
    
def GetAllFiles(folder, extensions_list):
    all_files = []
    """Given a folder, get all the filenames based on extension list."""
    for extension in extensions_list:
        all_files.extend(glob.glob(folder + "/*" + extension))
    return all_files
    
def safe_clone(repositoryURL, branchName='dev1', dst=None):
	repositoryName = osp.basename(repositoryURL)
	try:
		if not os.path.exists(dst):
			os.system("git clone -b {} {} {} --depth=1".format(branchName, repositoryURL, dst))
		else:
			print('found ', repositoryName)
	except:
		 print("cloning failed: git clone -b {} {} {} --depth=1".format(branchName, repositoryURL, dst))   

def get_onlyfilesfromfolder(foldername):
    '''
    List all files(skip if isdir and return fileslist)
    '''
    ret_files = []
    files = glob.glob(os.path.join(foldername,"*")) 
    [ret_files.append(f) for f in files if osp.isfile(f) is True]
    return ret_files

def equalDrawFromFolder(folderpath, max_draw, shuffle=False):
    '''
    equally draw files from a folder
    '''
    root_files = []
    for root, dirs, files in os.walk(folderpath, topdown=True):
        if root==folderpath:
            if len(files):
                fileList = [osp.join(folderpath, f) for f in files]
                root_files.extend(fileList)
            break

    # import pdb;pdb.set_trace()
    x_dirs = os.listdir(folderpath)
    total_subdir_count = len(x_dirs)
    if len(root_files):
        total_subdir_count+=1
    if shuffle:
        random.shuffle(x_dirs)
        random.shuffle(root_files)
    

    if max_draw > total_subdir_count and total_subdir_count > 0:
        max_draw_count = max_draw//total_subdir_count
    else:
        x_dirs = x_dirs[:max_draw]
        max_draw_count = 1
    
    subfiles = []
    for dir_ in x_dirs:
        files = get_onlyfilesfromfolder(os.path.join(folderpath, dir_))
        
        if len(files):
            if shuffle:
                random.shuffle(files)
            subfiles.append(files)
    if len(root_files):
        subfiles.append(root_files)
    
    upd_files = []
    all_files = list(itertools.chain.from_iterable(subfiles))
    for ind, sublist in enumerate(subfiles):
        upd_files.extend(sublist[:max_draw_count])
    
    if len(upd_files) < max_draw:
        lukup_list = list(np.setdiff1d(all_files, upd_files))
        if len(lukup_list) >0:
            if shuffle:
                random.shuffle(lukup_list)
            r_count = max_draw - len(upd_files)
            upd_files.extend(lukup_list[:r_count])
    return upd_files

def tryint(s):
	try:
		return int(s)
	except:
		return s

def alphanum_key(s):
	""" Turn a string into a list of string and number chunks.
		"z23a" -> ["z", 23, "a"]
	"""
	return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
	""" Sort the given list in the way that humans expect.
	"""
	l.sort(key=alphanum_key)
	return l

    
if __name__ == '__main__':
    get_timestamp()
    
