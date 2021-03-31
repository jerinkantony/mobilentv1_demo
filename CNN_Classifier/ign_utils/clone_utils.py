
def safe_clone_ign_utils():
    import os
    import sys
    '''
    Copy this function to your main code nd call it. Then import any script in ign_utils. Enjoy
    '''
    path = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(path,'ign_utils')
    found_ign = False
    for p in sys.path:
        if os.path.basename(p)=='ign_utils':
            found_ign = True
    if not found_ign:
        sys.path.append(dst)
        try:
            os.rmdir(dst)
        except:
            pass
        try:
            if not os.path.exists(dst):
                os.system("git clone http://10.201.0.12:9999/ml/ign_utils "+dst+" -b dev1 --single-branch --depth=1")
            else:
                print('found ign_utils')
        except:
             print('cloning failed: git clone http://10.201.0.12:9999/ml/ign_utils "+ " -b " + "dev1" + " --single-branch --depth=1')

def clone_repo_safe(root_dir, repo_name, repo_url,branchname):
    import os.path as osp
    import sys
    import os
    path = sys.path
    # import pdb;pdb.set_trace()
    if repo_name in path:
        pass

    repo_path = osp.join(root_dir,repo_name)
    found_repo = False
    for p in sys.path:
        if os.path.basename(p)==repo_name:
            found_repo = True
    if not found_repo:
        sys.path.append(repo_path)
        try:
            os.rmdir(repo_path)
        except:
            pass
        try:
            if not os.path.exists(repo_path):
                os.system("git clone "+ repo_url +" -b " + branchname + " " + repo_path+" --single-branch --depth=1")
                return True
            else:
                print('found repo path',repo_path)
                return False
        except:
            print('cloning failed',repo_url,branchname)    
            return False
    
    
def Clone_DB(DB_path, dbname, dst_folder_name=None,server_mode=False, makedirs=True):
    import os.path as osp
    import sys
    import os
    try:
        from general_utils import mkdir_safe
    except:
        from .general_utils import mkdir_safe
    
    if dst_folder_name is not None:
        DBPath = osp.join(DB_path, dst_folder_name)
    else:
        DBPath = osp.join(DB_path,dbname)
    if not server_mode:
        try:#DB
            if not os.path.exists(DBPath):
                os.system("git clone http://10.201.0.12:9999/ml/DB "+DBPath
                + " -b " + dbname + " --single-branch --depth=1")
        except:
            if makedirs:
                mkdir_safe(osp.join(DBPath, 'train'))
                mkdir_safe(osp.join(DBPath, 'val' ))
            print('Add to repo: http://10.201.0.12:9999/ml/DB, \
            check https://stackoverflow.com/questions/51123925/how-can-i-git-init-in-existing-folder')
    elif makedirs:
        mkdir_safe(osp.join(DBPath, 'train'))
        mkdir_safe(osp.join(DBPath, 'val' ))     
    
def Clone_DB_weights(path, dbname, server_mode=False,makedirs=False):
    import os.path as osp
    import sys
    import os
    try:
        from general_utils import mkdir_safe
    except:
        from .general_utils import mkdir_safe
    DBPath = osp.join(path, 'DB',dbname)
    wtPath = osp.join(path, 'weights',dbname)
    if not server_mode:
        try:#DB
            if not os.path.exists(DBPath):
                os.system("git clone http://10.201.0.12:9999/ml/DB "+DBPath
                + " -b " + dbname + " --single-branch --depth=1")
        except:
            if makedirs:
                mkdir_safe(osp.join(DBPath, 'train'))
                mkdir_safe(osp.join(DBPath, 'val' ))
            print('Add to repo: http://10.201.0.12:9999/ml/DB, \
            check https://stackoverflow.com/questions/51123925/how-can-i-git-init-in-existing-folder')
        #import pdb;pdb.set_trace()  
        # import pdb;pdb.set_trace()
        
        try:#weights
            if not os.path.exists(wtPath):
                os.system("git clone http://10.201.0.12:9999/ml/weights "+wtPath
                + " -b " + dbname + " --single-branch --depth=1")

        except:
            print('Add to repo: http://10.201.0.12:9999/ml/weights, \
            check https://stackoverflow.com/questions/51123925/how-can-i-git-init-in-existing-folder')

        mkdir_safe(wtPath)
    else:
        # mkdir_safe(osp.join(DBPath, 'train'))
        # mkdir_safe(osp.join(DBPath, 'val' ))
        mkdir_safe(wtPath)
        

if __name__=='__main__':
    #Auto clone function call
    safe_clone_ign_utils()
    #example usage
    from ign_utils.sort_pts import sort_aniclkwise
