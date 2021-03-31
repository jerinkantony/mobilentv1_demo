#All imports are kept inside the function call for unnecessary imports.

def read_text_lines(txt_path):
    ''''read a text and returns after splitting by lines'''
    with open(txt_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    
def write_text_lines(txt_path, content):
    '''writes txt lines to txt path'''
    with open(txt_path, 'w') as f:
        f.writelines(content)
    return True
    
    
if __name__=='__main__':
    # read_text_lines(txt_path) ***unit test***
    txt_path = 'README.md'
    txt = read_text_lines(txt_path)
    print('txt: ',txt)
    
    # write_text_lines(txt_path, content) ***unit test***
    txt_path = 'sample.txt'
    content = 'abcd'
    write_text_lines(txt_path, content)
    
    
    
