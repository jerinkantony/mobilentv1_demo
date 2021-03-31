import numpy as np


def downsample(data, sr_in,sr_out):
    #import pdb;pdb.set_trace()
    assert sr_in>sr_out,'sr_in<sr_out'+str(sr_in)+', '+str(sr_out)
    ratio = sr_in/sr_out
    k=ratio-int(ratio)
    assert k==0, 'ratio not an integer'+str(ratio)
    dataout = data[::int(ratio)]
    return dataout
    

def resize1D(xin,outLength):
    ratio = len(xin)/(outLength)
    
    #import pdb;pdb.set_trace()
    #print('ratio:',ratio)
    k=ratio-int(ratio)
    if k!=0:
        xout = xin
        '''
        #source x points
        xp = np.arange(0, len(xin))
        
        #destination x points
        x = np.arange(0, len(xin), ratio)
        
        #numpy.interp(x, xp, fp, left=None, right=None, period=None)
        xout = np.interp(x, xp, xin)
        #print('xout len:',len(xout))
        
        #xout = np.resize(xin,outLength)
        '''
    else:
        xout = xin[::int(ratio)]
        
    if len(xout) != 12000:
        print('len(data)',len(xout))
        import pdb;pdb.set_trace() 
                
                
    return xout
    
    
    
if __name__=='__main__':  
    #Source audio samples
    xin = np.array([0,1,2,3]).astype('float32')
    outLength = 3
    xout = resize1D(xin,outLength)


    print('xout',xout)
