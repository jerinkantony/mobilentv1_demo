import os
import shutil
import sys
sys.path.append("../")

def rsync(check_filename_only=False, check_size_only=True, delete_extras=True):
    size_only, delete, ignore_existing= "", "", ""
    if check_size_only==True:
        size_only=" --size-only "
    if check_filename_only==True:
        ignore_existing = " --ignore-existing "
        size_only=""
    if delete_extras==True:
        delete = " --delete "
        
        
    print('Executing dry-run..')
    cmd = "rsync  --dry-run --update -raz --progress {} {} -e  'ssh -o StrictHostKeyChecking=no' {} {} ".format(src_loc,destloc,size_only,delete)
    ret = os.system(cmd)
    
    cmd = "rsync -av {} {} -e  'ssh -o StrictHostKeyChecking=no' {} {} ".format(src_loc,destloc,size_only,delete)
    print('Executing: ',cmd)
    ret = os.system(cmd)
    return ret
    

def getfilelist(folderpath):
    all_files = []
    root_counter=1
    for root, dirs, files in os.walk(folderpath):
        print("Get file list in progress {},folder count {}".format(folderpath,root_counter),"\r",end="")
        root_counter+=1
        for f in files:
            fp = os.path.join(root, f)
            all_files.append(fp)
    print()
    return all_files
def getmissingfiles_dest(src, dest, src_files, dest_files):
    common_fp_src = []
    src_splitted = src.split("/")
    for x in src_files:
        
        split_x = x.split("/")[len(src_splitted):]
        new_fp = "".join("/"+y for y in split_x)
        common_fp_src.append(new_fp)
    
    common_fp_dest = []
    dest_splitted = dest.split("/")

    for x in dest_files:
        split_x = x.split("/")[len(dest_splitted):]
        new_fp = "".join("/"+y for y in split_x)
        common_fp_dest.append(new_fp)

    # import pdb;pdb.set_trace()
    dest_missing = [src+x for x in common_fp_src if x not in common_fp_dest]
    return dest_missing



def delete_non_matchingfiles_dest(src,dest):
    # import pdb; pdb.set_trace()
    print("Going to delete non matching files in {},{}".format(src, dest))
    files_src = getfilelist(src)
    files_dest = getfilelist(dest)
    extra_files_dest = getextrafiles_dest( src, dest,files_src, files_dest)
    for f in extra_files_dest:
        if os.path.exists(f):
            os.remove(f)
    print("Deleted {} files ".format(len(extra_files_dest)))
            

        
def remove_duplicates(foldername):
    print("Checking for duplicate files in {}".format(foldername))
    file_list = getfilelist(foldername)
    filenames = [os.path.basename(x) for x in file_list]
    unique_filenames = list(set(filenames))
    for f in unique_filenames:
        f_indices = [ind for ind in range(0, len(filenames)) if filenames[ind] == f]
        to_delete_files = []
        
        if len(f_indices) > 1:
            print("{} duplicates found for {}".format(len(f_indices), f))
            print()
            to_delete_files = f_indices[1:]
        for del_ind in to_delete_files:
            print("Deleting file {}".format(file_list[del_ind]))
            print()
            os.remove(file_list[del_ind])
        



def getextrafiles_dest(src, dest, src_files, dest_files):
    
    return getmissingfiles_dest(dest, src, dest_files, src_files)


def preprocess_folder(src,dest,func ,args, checkfolder_existing, check_file_count, overwrite_mode):
    if overwrite_mode:
        if os.path.exists(dest):
            shutil.rmtree(dest)

    if checkfolder_existing:
        if  os.path.exists(dest):
           pass
           print("Destination folder exists, skipping {}".format(func))
           return True
    src_files  =getfilelist(src)
    dest_files = getfilelist(dest)

    if check_file_count:
        
        if len(src_files)== len(dest_files):
            print("File count in src and dest are same")
            return True
    
    extra_files_dest = getextrafiles_dest(src, dest, src_files, dest_files)
    if len(extra_files_dest) >0:
        for f in extra_files_dest:
            os.remove(f)
        print("Found {} extra files  and deleted".format(len(extra_files_dest)))
    
    

if __name__ == '__main__':
    if 0:   # rsync test
        srcloc = "root@45.79.125.168:/root/FlaskServer/raw_clean/"
        destloc = os.path.abspath("DB/raw_clean/")
        rsync(srcloc, destloc)  

    if 1:  #check missing files in dest
        src_test = "/home/ignitarium/projects/AudioClassifier/dev8_branch/DB/test_folder_src"
        dest_test = "/home/ignitarium/projects/AudioClassifier/dev8_branch/DB/test_folder_dest"
        print(">>>>>",getmissingfiles_dest(src_test, dest_test, getfilelist(src_test), getfilelist(dest_test)))
        print(">>>>>",getextrafiles_dest(src_test, dest_test, getfilelist(src_test), getfilelist(dest_test)))
        # remove_duplicates(src_test)
        # missing_src = getmissingfiles_dest(src_test, dest_test, getfilelist(src_test), getfilelist(dest_test))
        # move_missing_files_to_proper_folder(missing_src, src_test,dest_test)
        delete_non_matchingfiles_dest(src_test,dest_test)



