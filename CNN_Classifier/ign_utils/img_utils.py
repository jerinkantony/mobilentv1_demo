import os
import cv2
import math
import time
import random
import numpy as np
import os.path as osp
from PIL import Image
from operator import itemgetter
import xml.etree.ElementTree as ET 
from itertools import combinations, groupby
from scipy.spatial import distance as dist

def get_direction_vector( box):
	'''
		Description: For finding the direction vector of a contour
		input box: numpy float32 box list e.g np.float32(rectlist)
		return direction vector 
		
	'''
	dist2 = np.sqrt((box[1][1] - box[0][1]) ** 2 + (box[1][0] - box[0][0]) ** 2)
	dist1 = np.sqrt((box[2][1] - box[1][1]) ** 2 + (box[2][0] - box[1][0]) ** 2)
	
	if dist2 > dist1:
		pt1 = box[0]
		pt2 = box[1]
		# print("PTS", pt1, pt2)
		dir_vec_Contour = get_unit_vec(pt1, pt2)
	else:
		pt1 = box[1]
		pt2 = box[2]

		dir_vec_Contour = get_unit_vec(pt1, pt2)

	new_dir_vec = dir_vec_Contour 
	return new_dir_vec

# def get_unit_vec(pt1, pt2):
# 	direction_vec = (pt2[0]-pt1[0], pt2[1]-pt1[1])
# 	unit_vec = direction_vec/magnitude(direction_vec)
# 	return tuple(unit_vec)

def four_point_transform_whole_image(img, box=None, pts_from_file=False):
	rows,cols,ch = img.shape
	path = os.path.dirname(os.path.abspath(__file__))
	if pts_from_file:
		ratio = 1
		Color = [0,0,0]
		#cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
		top, bottom, left, right = rows//2,rows//2,cols//2,cols//2 #rows,rows,cols,cols #
		img= cv2.copyMakeBorder(img,top, bottom, left, right,cv2.BORDER_CONSTANT,value=Color)
		rows,cols,ch = img.shape
		
		src_pts = np.float32(np.loadtxt(os.path.join(path, 'data', 'pts1.txt')))
		dst_pts = np.float32(np.loadtxt(os.path.join(path, 'data', 'pts2.txt')))
	
	else:
		ratio = 2.5
		tl, tr, br, bl = box
		box_width = int(getboxwidth(box))
		box_height = int(getboxheight(box))
		box_height = box_width*ratio
		src_pts=np.array(box, dtype="float32")
		tl = [3000, 1300]
		dst_pts = np.array([[tl[0], tl[1]],[tl[0] + box_width-1, tl[1]],[tl[0] + box_width-1, tl[1] + box_height-1],[tl[0], tl[1]+ box_height-1]], dtype="float32")
	#import pdb;pdb.set_trace()
	M = cv2.getPerspectiveTransform(src_pts,dst_pts)
	print('Matr8ix: ', M)
	dst = cv2.warpPerspective(img, M, (int(cols*ratio),int(rows*ratio)))
	return dst, M


def four_point_transform_whole_imageOld(img, box=None, random_interpolation=False, pts_from_file=False):
	'''
	Description: Performs perspective correction of a region 
	Inputs:
	img : The original Image
	box : the coordinates of the contour (usually the length of the list should be 4 ; the extremal points)
	'''
	img_ht, img_wd = img.shape[:2]
	top, bottom, left, right = img_ht//2,img_ht//2,img_wd//2,img_wd//2 #rows,rows,cols,cols #
	print('SHAPE: ', img_ht, img_wd)
	if pts_from_file:
		img= cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=[0, 0, 0])
		cv2.namedWindow('image',cv2.WINDOW_NORMAL)
		cv2.imshow('image',img)
		cv2.waitKey(30)
		ratio = 1
		path = os.path.dirname(os.path.abspath(__file__))
		# src_pts = np.float32(np.loadtxt(os.path.join(path, 'data', 'pts1.txt')))
		# dst_pts = np.float32(np.loadtxt(os.path.join(path, 'data', 'pts2.txt')))
		src_pts = np.load(os.path.join(path, 'data', 'pts1.npy'))
		dst_pts = np.load(os.path.join(path, 'data', 'pts2.npy'))
		print('Points: ', src_pts, dst_pts)
	else:
		ratio = 2.5
		tl, tr, br, bl = box
		box_width = int(getboxwidth(box))
		# width = img.shape[1]
		box_height = int(getboxheight(box))
		box_height = box_width*ratio
		# height = img.shape[0]
		src_pts=np.array(box, dtype="float32")
		# dst_pts = np.array([[tl[0], tl[1]],[width-1, tl[1]],[width-1, height-1],[tl[0], height-1]], dtype="float32")

		tl = [3000, 1300]
		# dst_pts = np.array([[tl[0], tl[1]],[tl[0] + box_width-1, tl[1]],[tl[0] + box_width-1, tl[1] + box_height-1],[tl[0], tl[1]+ box_height-1]], dtype="float32")
		dst_pts = np.array([[tl[0], tl[1]],[tl[0] + box_width-1, tl[1]],[tl[0] + box_width-1, tl[1] + box_height-1],[tl[0], tl[1]+ box_height-1]], dtype="float32")
	
	M = cv2.getPerspectiveTransform(src_pts, dst_pts) 
	print('Matr8ix: ', M)
	interpolation = cv2.INTER_LINEAR
	if random_interpolation:
		interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST])
	dst = cv2.warpPerspective(img, M, (img_wd*ratio, img_ht*ratio))
	return dst, M

def four_point_transform1(image, pts, maxHeight, maxWidth):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def make_fit_rect(box):
	rect = cv2.minAreaRect(np.array(box))
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	return box

def order_points(pts):
	"""
	sort points clockwise
	"""
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")
	
def get_min_rect(cnt):
	"""
	returns four corner points fitting given given contour
	"""
	return make_fit_rect(cnt)
	approx_rect = None
	epsilon = 0.0001
	# print('iteration starts')
	cnt = cv2.convexHull(cnt, returnPoints=True)  # also to reduce points
	while 1:
		epsilon += .1
		# print('epsilon',epsilon)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		# print(len(approx))
		if len(approx) < 4:
			break
		elif len(approx) == 4:
			approx_rect = approx
			break
		else:
			pass
	# print(len(approx))
	# print('rect points:',approx_rect)
	return approx_rect

def four_point_transform(img, box, random_interpolation=False):
	'''
	Description: Performs perspective correction of a region 
	Inputs:
	img : The original Image
	box : the coordinates of the contour (usually the length of the list should be 4 ; the extremal points)
	'''
	width = int(getboxwidth(box))
	height = int(getboxheight(box))
	src_pts=np.array(box, dtype="float32")
	dst_pts = np.array([[0, 0],[width-1, 0],[width-1, height-1],[0, height-1]], dtype="float32")
	M = cv2.getPerspectiveTransform(src_pts, dst_pts) 
	interpolation = cv2.INTER_LINEAR
	if random_interpolation:
		interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST])
	warped = cv2.warpPerspective(img, M, (width, height), interpolation, borderMode=cv2.BORDER_CONSTANT, 
	borderValue=(255, 255, 255))
	return warped, M
def four_point_transform2(img, box, random_interpolation=False):
    '''
    Description: Performs perspective correction of a region 
    Inputs:
    img : The original Image
    box : the coordinates of the contour (usually the length of the list should be 4 ; the extremal points)
    '''
    width = int(getboxwidth(box))
    height = int(getboxheight(box))
    src_pts=np.array(box, dtype="float32")
    dst_pts = np.array([[0, 0],[width-1, 0],[width-1, height-1],[0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts) 
    interpolation = cv2.INTER_LINEAR
    _, IM = cv2.invert(M)
    if random_interpolation:
        interpolation = random.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST])
    warped = cv2.warpPerspective(img, M, (width, height), interpolation, borderMode=cv2.BORDER_CONSTANT, 
    borderValue=(255, 255, 255))
    return warped, IM
def get_bounding_boxes(contours, pad_l=0, pad_r=0, pad_u=0, pad_b=0, area_threshold=10):
	filtered_contours = []
	for contour in contours:
		(x,y,w,h) = cv2.boundingRect(contour)
		if area_threshold < cv2.contourArea(contour) and w > 10 and h > 10:
			bbox = cv2.boundingRect(contour)
			xmin, ymin = bbox[0]-pad_l, bbox[1]-pad_u
			xmax, ymax = xmin + bbox[2] + pad_r, ymin + bbox[3] + pad_u
			filtered_contours.extend([xmin, ymin, xmax, ymax])
	return filtered_contours

def get_bounding_box(contours, pad_l=0, pad_r=0, pad_u=0, pad_b=0, area_threshold=10, width_threshold=10, ht_threshold=10):
	filtered_contours = []
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	cnt=contours[max_index]
	(x,y,w,h) = cv2.boundingRect(cnt)
	if area_threshold < cv2.contourArea(cnt) and w > width_threshold and h > ht_threshold:
		bbox = cv2.boundingRect(cnt)
		xmin, ymin = bbox[0]-pad_l, bbox[1]-pad_u
		xmax, ymax = xmin + bbox[2] + pad_r, ymin + bbox[3] + pad_u
		filtered_contours.extend([xmin, ymin, xmax, ymax])
	else:
		print('Ignored..!')
	return filtered_contours
	
def rotate(point, angle_deg=math.radians(90), origin=(0, 0)):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle_deg) * (px - ox) - math.sin(angle_deg) * (py - oy)
	qy = oy + math.sin(angle_deg) * (px - ox) + math.cos(angle_deg) * (py - oy)
	return qx, qy

def magnitude(pt1):
	return np.sqrt((pt1[0])**2 + (pt1[1])**2)

def get_unit_vec(pt1, pt2):
	#print("PTS:", pt1,pt2)
	direction_vec = (pt2[0]-pt1[0], pt2[1]-pt1[1])
	if magnitude(direction_vec) >0:
		unit_vec = direction_vec/magnitude(direction_vec)
	else:
		unit_vec = direction_vec
	return tuple(unit_vec)

def autocrop(org,box):
	img = org.copy()
	im_gray = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
	(thresh, im_bw) = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY)
	col_sum = np.sum(im_bw, axis=0)
	col_sum_max = max(col_sum)
	rows, cols = im_bw.shape
	hz = []
	for x in range(0, cols):
		if col_sum[x] == col_sum_max:
			hz.append(x)
			break
	for x in reversed(range(0, cols)):
		if col_sum[x] == col_sum_max:
			hz.append(x)
			break
	left_dist = hz[0]
	right_dist =org.shape[1] - hz[1]
	cropped = img[0:rows, hz[0]:hz[1]]
	box=expandbox(box,padleft=-left_dist,padright=-right_dist,padtop=0,padbottom=0)
	return cropped,box

def display_image(img, winName='ImageWindow', waitkey=1):
	cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
	cv2.imshow(winName, img)
	cv2.waitKey(waitkey)

def apply_transform(point_list,M):
	out_point_list= (cv2.perspectiveTransform(np.array(point_list),M)).astype(int)
	return out_point_list
def apply_transform1(point_list,M):
    point = (np.array([point_list],dtype=np.float32))
    out_point_list= (cv2.perspectiveTransform(point,M)).astype(int)
    print("out_point_list",out_point_list)
    return out_point_list
def findContours(color_img):
	'''findContours with some morphology first'''
	imgray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
	imgray = cv2.medianBlur(imgray, ksize=7)
	ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((5, 5), np.uint8)
	open_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	contours, _ = cv2.findContours(open_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	return contours

def area_filter(contours,threshold_area):
	contours_out = []
	for c in contours:
		if cv2.contourArea(c)>threshold_area:
			contours_out.append(c)
	return contours_out

def direction_boxsort(boxpoints, unit_vec,vertical_sort_first=True):
	'''
	   description : for sorting all the points in a specific direction
	   inputs
	   boxpoints : contour points of type float32
	   unit_vec: the unit vector in the direction of sort is desired in tuple format
	   output:
	   returns the sorted points 
	   #provide vertical unitvec when vertical first is true ,else provide Horizontal unit vec
	'''

	sorted_list = sorted(boxpoints, key=lambda coord: np.dot(coord, unit_vec))
	first_pts = sorted_list[:2]
	second_pts = sorted_list[2:]
	normal_vec = rotate(unit_vec, angle_deg=math.radians(90))
	if vertical_sort_first:
		bl, br = sorted(first_pts, key=lambda coord: np.dot(coord, normal_vec))
		tl, tr = sorted(second_pts, key=lambda coord: np.dot(coord, normal_vec))    
	else:
		tl, bl = sorted(first_pts, key=lambda coord: np.dot(coord, normal_vec))
		tr, br = sorted(second_pts, key=lambda coord: np.dot(coord, normal_vec))
	return [tl, tr, br, bl]

def expandbox(box,padleft=10,padright=10,padtop=10,padbottom=10):
	'''
		description :  expand the contour, provide pad values explicitly else pad value of 10 will be taken
		input : the contour box points
		result : the expanded box points
	'''
	#print("Box", box)
	tl, tr, br, bl=box
	u1=get_unit_vec(tl,tr)
	print("U1:", u1)
	u2=get_unit_vec(tr,br)
	print("U2:", u2)
	tl[0]+=-u1[0]*padleft-u2[0]*padtop
	tl[1]+=-u1[1]*padleft-u2[1]*padtop
	tr[0]+=u1[0]*padright-u2[0]*padtop
	tr[1]+=u1[1]*padright-u2[1]*padtop
	br[0]+=u1[0]*padright+u2[0]*padbottom
	br[1]+=u1[1]*padright+u2[1]*padbottom
	bl[0]+=-u1[0]*padleft+u2[0]*padbottom
	bl[1]+=-u1[1]*padleft+u2[1]*padbottom
	box1=[tl,tr,br,bl]
	return box1

def find_centroid(boxpoints):
		M = cv2.moments(np.array(boxpoints))
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		return cX, cY

def getbox(cnt,unitvec,verticalfirst):
#provide vertical unitvec when vertical first is true ,else provide Horizontal unit vec
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box1=direction_boxsort(box,unitvec,verticalfirst)
		return box1

def checkboxsize(box,minwidth,minheight):
		boxwidth=getboxwidth(box)
		boxheight=getboxheight(box)
		if boxwidth>minwidth and boxheight>minheight:
			return 1
		else :
			return 0
def getboxwidth(box):
	tl,tr,br,bl=box
	return np.sqrt((tl[0]-tr[0])**2 + (tl[1]-tr[1])**2)

def getboxheight(box):
	tl,tr,br,bl=box
	return np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
def getboxangle(box):
	tl,tr,br,bl=box
	ydist=bl[1]-tl[1]
	xdist=bl[0]-tl[0]
	slope=ydist/xdist
	boxangle=math.degrees(math.atan(slope))
	return boxangle

def readxml(xmlfile):
	tree = ET.parse(xmlfile) 
	root = tree.getroot() 
	all_dict = {}
	for ind, item in enumerate(root.iter('object')):
		for child in item:
			if child.tag == "name":
				tag_name = child.text + '*'+str(ind)
				all_dict[tag_name] = []
			if child.tag == "bndbox":
				for child1 in child.iter('bndbox'):
					for child2 in child1:
						all_dict[tag_name].append(int(child2.text))
	return all_dict

def generatenewrandnum(num):
	if num !=0:
		return num
	else:
		while (num==0):
			num = random.randint(-4,4)
		return num

def rotateImg(point, centerPoint, angle):
	"""Rotates a point around another centerPoint. Angle is in degrees.
	Rotation is counter-clockwise"""
	angle = math.radians(angle)
	temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
	temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
	temp_point = [int(temp_point[0]+centerPoint[0]) , int(temp_point[1]+centerPoint[1])]
	return temp_point

def getContourfromRect(rect, theta):
	x1 = rect[0]
	y1 = rect[1]
	w = rect[2] 
	h = rect[3] 
	x2 = x1 +w
	y2 = y1
	x3 = x1+w
	y3 = y1 + h
	x4 = x1
	y4 = y1 + h
	center_x = x1+ w//2
	center_y = y1 + h//2
	pt1 =  rotateImg([x1,y1], [center_x, center_y], theta)
	pt2 =  rotateImg([x2,y2], [center_x, center_y],theta)
	pt3 =  rotateImg([x3,y3], [center_x, center_y], theta)
	pt4 =  rotateImg([x4,y4], [center_x, center_y], theta)
	return [pt1[0], pt1[1], pt2[0]-pt4[0],pt3[1]-pt2[1]]

def processRects(datadict, filepath):
	x_list= []
	y_list = []
	image = cv2.imread(filepath)
	for key, rect in datadict.items():
		label = str(key.split('*')[0])
		x= rect[0]
		y = rect[1]
		w = rect[2] - rect[0]
		h = rect[3] - rect[1]
		# frac = int(w*self.frac)
		frac_w  = random.randint(-int(w*0.15),int(w*0.15))
		frac_h  = random.randint(-int(w*0.2),int(w*0.2))
		
		frac_h = generatenewrandnum(frac_h)
		frac_w = generatenewrandnum(frac_w)
		
		newX = x - frac_w
		newY = y - frac_h
		newW = w + frac_w
		newH = h + frac_h
		newrect = [newX, newY, newW, newH]
		img = image[newY:newY + newH, newX:newX + newW]
		min_level, max_level = Get_Min_Max_intensity_level(img, 2)
		newrect = getContourfromRect(newrect, random.randrange(-3, 3, 1))
		x_list.append((filepath, newrect, (min_level, max_level)))
		y_list.append(label)
	return x_list, y_list

def get_contours(mask):
	contours = None
	im_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
	if cv2.countNonZero(im_gray) > 0:
		ret, open_mask = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY)
		kernel = np.ones((50,100),np.uint8)
		open_mask = cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, kernel)
		# display_image(open_mask, waitkey=0)
		mean_val = int(np.average(open_mask))
		if mean_val>=150:
			open_mask = cv2.bitwise_not(open_mask) 
		contours, hierarchy = cv2.findContours(
			open_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def get_points_refined(img, max_=False, hull=False):
	contours = get_contours(img)
	ref_pts = []
	max_area = 0
	index = None
	for i in range(len(contours)):
		if not max_:
			if hull:
				ref_pts.append(cv2.convexHull(contours[i], True))
			else:
				ref_pts.append(contours[i])
		else:
			area = cv2.contourArea(contours[i])
			if area > max_area:
				max_area = area
				index = i
	if max_:
		if hull:
			ref_pts.append(cv2.convexHull(contours[index], True))
		else:
			ref_pts.append(contours[index]) 
	return ref_pts

def create_background_image(width, height, rgb_color=None):
	"""Create new image(numpy array) filled with certain color in RGB"""
	if not rgb_color:
		rgb_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	image = np.zeros((height, width, 3), np.uint8)
	color = tuple(reversed(rgb_color))
	# Fill image with color
	image[:] = color
	return image

def Get_Min_Max_intensity_level(img, fraction=2):
	if type(img)==str:
		img = cv2.imread(img)
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	pix_count_list = hist.astype(int).tolist()
	ref_cumulative_count = img.shape[0]*img.shape[1]//fraction
	max_intensity_val = cumulative_count = 0
	for i in range(len(pix_count_list)-1, -1, -1):
		cumulative_count+=pix_count_list[i][0]
		if cumulative_count>=ref_cumulative_count:
			max_intensity_val = i
			# print('cumulative pixel count "{}" exceeded reference count "{}" max_intensity_level is "{}"'.format(cum_count, ref_cum_count, max_intensity_val))
			break
	min_intensity_val = max_intensity_val//fraction
	return min_intensity_val, max_intensity_val

def Change_Background2(contourpts, ht, wd, bgImage=None, color=[255, 255, 255]):
	drawing = np.zeros((ht, wd, 3), np.uint8)
	f = cv2.drawContours(drawing, contourpts, -1, color, -1, 8)
	if bgImage and osp.isfile(bgImage):
		bgImage = cv2.imread(bgImage)
		bgImage = cv2.resize(bgImage, (wd, ht), interpolation=cv2.INTER_AREA)
	else:
		bgImage = create_background_image(wd, ht)
	new_bg_image = (bgImage * cv2.bitwise_not(drawing) ) + drawing
	return new_bg_image

def Change_Background1(filename, bgImage=None):
	crop_object = False
	coordList = filename[1]
	image = cv2.imread(filename[0])
	ht, wd = image.shape[:2]
	if bgImage and osp.isfile(bgImage):
		bgImage = cv2.resize(cv2.imread(bgImage), (wd, ht))
	else:
		bgImage = create_background_image(wd, ht)
	for coord in coordList:
		x1, y1, x2, y2 = coord
		ht_ratio = (y2-y1)/ht
		
		if ht_ratio >=0.6:
			crop_img = image[int(y1):int(y2), int(x1):int(x2)]
			bgImage[y1:y2, x1:x2] = crop_img
			crop_object = True
	if crop_object is False:
		bgImage = image
	return bgImage

def get_mask_from_json(labeldata_list, ht, wd):
	all_points = []
	canvas = np.zeros((ht, wd, 3), np.uint8)
	for items in labeldata_list:
		points = items['points']
		points=[[int(p[0]),int(p[1])] for p in points]
		cv2.fillPoly(canvas, [np.array(points)], [255, 0, 0])
		all_points.append(points)
	if is_empty(all_points):
		return None
	else:
		return canvas

def is_empty(seq):
	try:
		return all(map(is_empty, seq))
	except TypeError:
		return False

def Stitch_to_bg(img,img2,img2_mask):
	img_copy = img.copy()
	img2_mask[img2_mask[:,:,0]==255] = 255
	mask = img2_mask[:,:,:]==255
	img_copy[mask] = img2[mask]
	return img_copy

def Change_Background(filename, bgImage=None):
	coordList = filename[1]
	image = cv2.imread(filename[0])
	ht, wd = image.shape[:2]
	if bgImage and osp.isfile(bgImage):
			bgImage = cv2.resize(cv2.imread(bgImage), (wd, ht))
	else:
		bgImage = create_background_image(wd, ht)

	if type(coordList[0])==list:
		
		crop_object = False
		for coord in coordList:
			x1, y1, x2, y2 = coord
			ht_ratio = (y2-y1)/ht
			if ht_ratio >=0.6:
				crop_img = image[int(y1):int(y2), int(x1):int(x2)]
				bgImage[y1:y2, x1:x2] = crop_img
				crop_object = True
		if crop_object is False:
			bgImage = image
		
	elif type(coordList[0])==dict:
		mask = get_mask_from_json(coordList, ht, wd)
		if mask is not None:
			bgImage = Stitch_to_bg(bgImage, image, mask)
		else:
			bgImage = image
	return bgImage

def get_transparent_image(img, mask, alpha=0.3):
	output = img.copy()
	s = cv2.addWeighted(mask, alpha, output, 1-alpha, 0, output)
	return output

def get_padding_on_boxList(boxlist, pad_val):
	new_box = []
	if len(boxlist):
		for box in boxlist:
			x, y, w, h = box
			x1, y1, w1, h1 = x+pad_val, y, w, h
			new_box.append([x1, y1, w1, h1])
	return new_box

def draw_rect_from_xywh(image, boxList, color=(0, 255, 0), thickness=6, xywh=True):
	if len(boxList) > 0:
		for box in boxList:
			if len(box):
				x1, y1 = box[:2]
				if xywh:
					width = box[2]
					height = box[3]
					x2 = x1 + width
					y2 = y1 + height
				else:
					x2, y2 = box[2:]
				cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
	return image

if __name__ == "__main__":
	#four_point_transform example
	if 0: 
		import os
		import json
		imgfolder = 'fourpointtransformsamples'
		for files in os.listdir(imgfolder):
			if files.endswith('json'):
				samplejson = os.path.join(imgfolder, files)
			else:
				sampleimg = cv2.imread(os.path.join(imgfolder,files))
		#read the  json
		SampleContour = []
		with open(samplejson) as json_file:
				data = json.load(json_file)
				all_data = data['shapes']
				for datum in all_data:
					print(datum['points'])
					SampleContour.append(datum['points'])
		#map the contour to original image
		sampleImgcopy = sampleimg.copy()
		cv2.namedWindow("Mapped Contours", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Expand Box Contours", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Warped  Img", cv2.WINDOW_NORMAL)
		for cnt in SampleContour:
			cnt = np.int0(cnt)
			cv2.drawContours(sampleimg, [np.asarray(cnt)], -1, (0, 255, 0), 3) 
		cv2.imshow("Mapped Contours", sampleimg)
		#finding direction vector of the contour points
		for cnt in SampleContour:
			cnt = np.int0(cnt)
			rect = cv2.minAreaRect(np.asarray(cnt))
			box = np.int0(cv2.boxPoints(rect))
			unit_vec = get_direction_vector(box) # calculating unit vector
			print("Unit Vector", unit_vec)
			box = direction_boxsort(box,unit_vec,True) # sorting the points wrt the unit vector
			print("Box Coordinates before expansion {}".format(box))
			box = expandbox(box)
			print("Box Coordinates after expansion {}".format(box))
			
			cv2.drawContours(sampleimg, [np.asarray(box)], -1, (0, 0, 255), 3) 
			cv2.imshow("Expand Box Contours", sampleimg)
			warped, box_new,m = four_point_transform(sampleImgcopy, box)
			cv2.imshow("Warped  Img", warped)
		cv2.waitKey(0)

	if 1:
		if os.path.isfile('data/src_pts.npy'):
			fourpoints = np.load('data/src_pts.npy')
		img = cv2.imread('data/Final2.png')
		warped, m = four_point_transform_whole_image(img, pts_from_file=True)
		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.imshow("result", warped)
		cv2.waitKey(0)


