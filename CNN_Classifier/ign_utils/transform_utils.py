from MousePts import MousePts
import numpy as np
import cv2



if __name__=='__main__':
    img = np.zeros((512,512,3), np.uint8)
    
    pts,img = MousePts(img).getpt(4)
    print(pts)
        
    
    
    
    #cv2.imshow(windowname,img)
    cv2.waitKey(0)
    
