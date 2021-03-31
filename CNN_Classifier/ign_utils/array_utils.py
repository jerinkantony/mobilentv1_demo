import numpy as np

def rolling_window(a, window,stride):
    '''Returns 2D array of window with stride'''
    assert (len(a) - (window - stride))%stride == 0, 'len(a) - (window - stride))%stride!=0'
    rowcount = (len(a) - (window - stride))//stride
    shape = (rowcount, window)
    strides = (a.itemsize*stride, a.itemsize)
    #import pdb;pdb.set_trace()
    b = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return b
    
def moving_window(a, window,stride, padding=True):
    '''Returns 2D array of window with stride'''
    if padding:
        assert window %2 !=0, 'Windo size should be odd'
        pad_val = window//2
        a = np.pad(a, pad_val, 'constant')    
#     assert (len(a) - (window - stride))%stride == 0, 'len(a) - (window - stride))%stride!=0'
    rowcount = (len(a) - (window - stride))//stride
    shape = (rowcount, window)
    strides = (a.itemsize*stride, a.itemsize)
    #import pdb;pdb.set_trace()
    b = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return b
    
if __name__=='__main__':
    # Example:
    a = np.arange(11)
    print (moving_window(a=a, window=3, stride=1))
    
    
    
# def moving_normalize(data, window,maxval=1,minval=-1):
    
