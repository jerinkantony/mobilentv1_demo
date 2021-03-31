from MousePts import MousePts
from img_utils import apply_transform
import numpy as np
import cv2
from math import sqrt

import cv2
import numpy as np

def convert_pts(boxpoints, M):
	boxpoints = np.float32(boxpoints)
	warp_boxes = []
	for b in boxpoints:
		#import pdb;pdb.set_trace()
		b = np.array(b).reshape(1, 1, 2)
		w_b = apply_transform(b, M)
		w_box_pt = list(w_b[0][0])
		warp_boxes.append(w_box_pt)
	return warp_boxes
		
if __name__=='__main__':
	
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	

	img = cv2.imread('data/newsample.png')
	img1=img.copy()
	
	rows,cols,ch = img.shape
	print('SHAPE: ', rows, cols)
	#Color = [0,0,0]
	#cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
	#top, bottom, left, right = rows//2,rows//2,cols//2,cols//2 #rows,rows,cols,cols #
	#img= cv2.copyMakeBorder(img,top, bottom, left, right,cv2.BORDER_CONSTANT,value=Color)
	#rows,cols,ch = img.shape
	
	cv2.imshow('image',img)
	cv2.waitKey(30)
	
	
	if 0:
		pts1,img = MousePts(img).getpt(4)
		print(pts1)
		np.savetxt('data/pts1.txt',pts1)
	else:
		pts1 = np.float32(np.loadtxt('data/pts1.txt'))
	
	p1 = pts1[1]
	p2 = pts1[0]
	w1 = sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
	p1 = pts1[1]
	p2 = pts1[2]
	h1 = sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
	
	x_offset,y_offset = 200,600
	x1,y1 = pts1[0][0]+x_offset,pts1[0][1]+y_offset
	pts2 = np.float32([[x1,y1], [x1+w1, y1], [x1+w1,y1+h1], [x1, y1+h1]])
	pts1 = np.float32(pts1)	
	print('Points: ', pts1, pts2)
	
	#import pdb;pdb.set_trace()
	M = cv2.getPerspectiveTransform(pts1,pts2)
	print('Matr8ix: ', M)
	dst = cv2.warpPerspective(img1, M, (2*cols,3*rows)) 
	#import pdb;pdb.set_trace()
	
	#Transforming points 
	pointsOut = convert_pts(pts1, M)
	print('pointsOut: ', pointsOut)
	cv2.polylines(img=dst, pts=np.array([pointsOut]), isClosed=True, color=(0,0,255), thickness=2)
	
	if 0:
	    #Transforming a point
	    pts1,img = MousePts(img).getpt(2)
	    pointsOut = convert_pts(pts1, M)
	    print('pointsOut: ', pointsOut)
	    cv2.polylines(img=dst, pts=np.array([pointsOut]), isClosed=True, color=(0,0,255), thickness=2)
	
	if 0:
		p1,p2 = MousePts(dst).selectRect(dst,'image')
		print(p1,p2)
		np.savetxt('data/p1.txt',p1)
		np.savetxt('data/p2.txt',p2)
	else:
		p1=np.loadtxt('data/p1.txt')
		p2=np.loadtxt('data/p2.txt')
	crop_img = dst[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
    
	cv2.imwrite('dst.png',dst)
	cv2.imwrite('crop_img.png',crop_img)
    


	cv2.namedWindow('birdscrop',cv2.WINDOW_NORMAL)
	cv2.imshow('birdscrop',crop_img)
	
	
	cv2.waitKey(0)
	#import pdb;pdb.set_trace()
	
