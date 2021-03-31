import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

class Stitcher():
    def __init__(self):
        self.orb_detector = cv2.ORB_create(1000)
        
    def stitch(self, img, img_):
        img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.orb_detector.detectAndCompute(img1,None)
        kp2, des2 = self.orb_detector.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2) 

        #print matches
        # Apply ratio test
        good = []
        for m in matches:
             if m[0].distance < 0.5*m[1].distance:         
             	good.append(m)
        matches = np.asarray(good)


        if len(matches[:,0]) >= 4:
            src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            print (H)
        else:
            raise AssertionError("Can't find enough keypoints.")  	
           
        dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0])) 
           
        rslt = np.zeros([ img.shape[0], img.shape[1] + img_.shape[1],3],dtype=np.uint8)
        rslt[0:img.shape[0], 0:img.shape[1]] = img
        
        blackpixs=np.all(rslt == [0,0,0], axis = -1)
        
        rslt[blackpixs]=dst[blackpixs]
        
        #cv2.imwrite('resultant_stitched_panorama.jpg',dst)
        return rslt
        

if __name__=='__main__':
    img_ = cv2.imread('data/images/1Hill.JPG')
    img = cv2.imread('data/images/2Hill.JPG')
    
    
    dst1 = Stitcher().stitch(img_, img)
    cv2.imshow('dst1',dst1)
    
    img_ = dst1# cv2.imread('data/images/2Hill.JPG')
    img = cv2.imread('data/images/3Hill.JPG')
    
    dst2 = Stitcher().stitch(img_, img)
    
    cv2.imshow('dst2',dst2)
    
    #dst3 = Stitcher().stitch(dst1, dst2)
    #cv2.imshow('dst3',dst3)
    
    cv2.waitKey(0)
    
    
    

