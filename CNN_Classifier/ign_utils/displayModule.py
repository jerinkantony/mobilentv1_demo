import cv2
import numpy as np

from img_utils import display_image
from draw_utils_ign import determine_text_size
from img_utils import draw_rect_from_xywh, get_padding_on_boxList


class displayAnnotations():
	def __init__(self):
		#default params
		self.color = (255, 255, 255)
		self.thickness = 4
		self.align_left = True
		self.font = cv2.FONT_HERSHEY_SIMPLEX 
		self.side_margin = 50
		self.offsetval = None
		self.posList = []
		self.left_counter = 1 
		self.right_counter = 1 
		self.txt_height = None
	def add_image(self, image):
		self.clear_all()
		self.image = image
		self.image_raw = image.copy()
		self.ht, self.wd = self.image.shape[:2]
		self.txt_height = determine_text_size(self.ht)
		
	def add_text(self, text, align_left=None, color=None, thickness=None, offset=None):
		if align_left is None:align_left = self.align_left
		if color is None:color = self.color
		if thickness is None:thickness = self.thickness
		
		textSize = cv2.getTextSize(text, fontFace=self.font, fontScale=self.txt_height, thickness=self.thickness)

		if offset is not None:
			x_pos, y_pos = offset
			
		elif align_left:
			y_pos = 50*(self.left_counter)
			x_pos = self.side_margin  
			self.left_counter+=1
		else:
			y_pos  = 50*self.right_counter
			x_pos = self.wd - textSize[0][0]-self.side_margin
			self.right_counter+=1
		cv2.putText(self.image, text, (x_pos, y_pos), self.font, self.txt_height, color, self.thickness, cv2.LINE_AA)     
		# return self.image
	
	def add_boxes(self, boxList, txtList=[], padval=False, color=(255, 0, 0), thickness=6, xywh=True, hfactor=2):
		if len(boxList) == 0:
			return 
		if padval:
			boxList = get_padding_on_boxList(boxList, padval)
		
		draw_rect_from_xywh(self.image, boxList, color=color, xywh=xywh, thickness=thickness)
		
		for index, box in enumerate(boxList):
			x, y, w, h = box
			offsetPos = (x + self.wd-(2*padval), y + h//hfactor)
			if txtList:
				text =  txtList[index]
				self.add_text(text, offset=offsetPos, color=color)
	
	def add_box(self, box, padval=False, color=(255, 0, 0), xywh=True):
		
		if len(box):
			if padval:
				box = get_padding_on_boxList(box, padval)
			draw_rect_from_xywh(self.image, box, color=color, xywh=xywh)

	def clear_text(self):
		self.image = self.image_raw
	
	def clear_all(self):
		
		self.posList = []
		self.left_counter = 1 
		self.right_counter = 1 
		
if __name__ == '__main__':
	img = np.zeros((512,1080,3), np.uint8)
	disp_obj = displayAnnotations()
	disp_obj.add_image(img)
	disp_obj.add_text('Press A to Add Box')
	disp_obj.add_text('Press B to Add Box')
	disp_obj.add_text('Press C to Add Box')
	disp_obj.add_text('Press D to Add Box', align_left=False, color=(0, 255, 0))
	disp_obj.add_text('Press E to Add Box', align_left=False, offset=(500, 500), color=(255, 255, 0))
	# disp_obj.clear_text()
	display_image(disp_obj.image, waitkey=0)
