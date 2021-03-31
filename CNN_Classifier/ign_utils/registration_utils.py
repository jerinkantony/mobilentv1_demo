#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
{Description}
{Licence_info}
"""
# Generic
import os
import os.path as osp

# Other libs
import cv2
import numpy as np
import time
# Own libs


class REGISTRATION(object):
	def __init__(self,):
		self.template = None  # Template image
		self.M = None
		self.mode = 'key_brute'
		self.method = 'bruteforce' #bruteforce or flann
		self.width = None
		self.height = None
		self.orb_detector = cv2.ORB_create(1000)
		self.kp1, self.des1 = None, None
	def setTemplate(self, img=None):
		if img is not None:
			if type(img) == str:
				assert (osp.isfile(img)
						is True), ('Invalid template path...!! {}'.format(img))
				self.template = cv2.imread(img)
			else:
				self.template = img
			self.readTemplate()
			self.kp1, self.des1 = self.orb_detector.detectAndCompute(self.template, None)
			#cv2.imshow('TEMPLATE', self.template)
			#cv2.waitKey(30)

	def readTemplate(self,):
		self.height, self.width = self.template.shape[:2]

	def save(self, img=None, path=None):
		cv2.imwrite(path, img)

	def fuse(self, bg=None, fg=None, alpha=0.5, grayscale=True):
		fg = cv2.resize(fg, bg.shape[:2][::-1])
		if grayscale:
			bg_zeros = np.zeros_like(bg)
			fg_zeros = np.zeros_like(bg)
			fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
			bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
			bg_zeros[:, :, 1] = fg
			fg_zeros[:, :, 2] = bg
			fused = cv2.addWeighted(bg_zeros, alpha, fg_zeros, 1-alpha, 0)
		else:
			fused = cv2.addWeighted(bg, alpha, fg, 1-alpha, 0)
		return fused

	def imgRegistartionKey_flannBased(self, img1=None, img2=None):
		tic = time.time()
		# Convert to grayscale.
		img1_copy = img1.copy()
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		height, width = img2.shape

		# Create ORB detector with 5000 features.
		# orb_detector = cv2.ORB_create(1000)
		# orb_detector = cv2.SIFT()

		# kp1, des1  = self.orb_detector.detectAndCompute(img1, None)
		kp2, des2 = self.kp1, self.des1
		kp1, des1 = self.orb_detector.detectAndCompute(img1, None)

		FLANN_INDEX_LSH = 6
		index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=20,
							key_size=12, multi_probe_level=1)  # 2
		search_params = dict(checks=50)  # or pass empty dictionary

		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, des2, k=2)
		# store all the good matches as per Lowe's ratio test.
		good = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				good.append(m)
		p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
		registrationIndex = len(good) / 5000

		# Find the homography matrix.
		homographyinv, maskinv = cv2.findHomography(p2, p1, cv2.RANSAC)
		homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

		# Use this matrix to transform the
		# colored image wrt the reference image.
		transformed_img = cv2.warpPerspective(img1_copy,
											  homography, (width, height))
		toc = time.time()
		print("Time Taken for method '{}' is '{}'".format(self.method, str(toc - tic)))

		return transformed_img, homographyinv, registrationIndex

	def imgRegistartionKey_bruteforceBased(self, img1=None, img2=None):
		tic = time.time()
		# Convert to grayscale.
		img1_copy = img1.copy()
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		height, width = img2.shape
		
		# Create ORB detector with 5000 features.
		# orb_detector = cv2.ORB_create(1000)
		#orb_detector = cv2.SIFT()

		
		# kp1, des1  = self.orb_detector.detectAndCompute(img1, None)
		kp2, des2 = self.kp1, self.des1
		kp1, des1 = self.orb_detector.detectAndCompute(img1, None)

		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = matcher.match(des1, des2)  # d1: query, d2: template


		# Sort matches on the basis of their Hamming distance.
		matches.sort(key=lambda x: x.distance)

		# Take the top 90 % matches forward.
		matches = matches[:int(len(matches)*90)]
		no_of_matches = len(matches)
		print("\n"*10, "Registration Index", no_of_matches/1000 , "\n"*10,)
		registrationIndex = no_of_matches/1000
		# Define empty matrices of shape no_of_matches * 2.
		p1 = np.zeros((no_of_matches, 2))
		p2 = np.zeros((no_of_matches, 2))

		for i in range(len(matches)):
			p1[i, :] = kp1[matches[i].queryIdx].pt
			p2[i, :] = kp2[matches[i].trainIdx].pt

		# Find the homography matrix.
		homographyinv, maskinv = cv2.findHomography(p2, p1, cv2.RANSAC)
		homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

		# Use this matrix to transform the
		# colored image wrt the reference image.
		transformed_img = cv2.warpPerspective(img1_copy,
											  homography, (width, height))
		toc = time.time()
		print("Time Taken for method '{}' is '{}'".format(self.method, str(toc -tic)))

		return transformed_img, homographyinv, registrationIndex

	def imgRegistartionEcc(self, im1=None, im2=None):
		# Convert images to grayscale
		im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
		im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

		# Find size of image1
		sz = im1.shape

		# Define the motion model
		warp_mode = cv2.MOTION_TRANSLATION

		# Define 2x3 or 3x3 matrices and initialize the matrix to identity
		if warp_mode == cv2.MOTION_HOMOGRAPHY:
			warp_matrix = np.eye(3, 3, dtype=np.float32)
		else:
			warp_matrix = np.eye(2, 3, dtype=np.float32)

		# Specify the number of iterations.
		number_of_iterations = 5000

		# Specify the threshold of the increment
		# in the correlation coefficient between two iterations
		termination_eps = 1e-10

		# Define termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
					number_of_iterations,  termination_eps)

		# Run the ECC algorithm. The results are stored in warp_matrix.
		(cc, warp_matrix) = cv2.findTransformECC(
			im2_gray, im1_gray, warp_matrix, warp_mode, criteria)  # im1_gray: query, im2_gray: template

		if warp_mode == cv2.MOTION_HOMOGRAPHY:
			# Use warpPerspective for Homography
			im1_aligned = cv2.warpPerspective(
				im1, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
		else:
			# Use warpAffine for Translation, Euclidean and Affine
			im1_aligned = cv2.warpAffine(
				im1, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
		return im1_aligned

	def apply(self, img=None):
		if self.mode == 'ecc':
			img = self.imgRegistartionEcc(img, self.template)
		elif self.mode == 'key_flann':
			img,transform, registrationIndex = self.imgRegistartionKey_flannBased(img, self.template)
		elif self.mode == 'key_brute':
			img,transform, registrationIndex = self.imgRegistartionKey_bruteforceBased(img, self.template)
		return img, transform, registrationIndex


if __name__=="__main__":
	source_folder = "data/registrationexample"
	templateImg = ''
	queryImg = ''
	for files in os.listdir(source_folder):
		if files.startswith('template'):
			templateImg = os.path.join(source_folder, files)
		else:
			queryImg = os.path.join(source_folder, files)
	print(templateImg, queryImg)
	t_img = cv2.imread(templateImg)	
	q_img = cv2.imread(queryImg)
	REGISTRATION_obj = REGISTRATION()
	REGISTRATION_obj.setTemplate(templateImg)
	registeredImg, transformMat, registrationIndex = REGISTRATION_obj.apply(q_img)
	fusedImg = REGISTRATION_obj.fuse(
					bg=t_img, fg=registeredImg)
	cv2.namedWindow("QueryImg", cv2.WINDOW_NORMAL)
	cv2.namedWindow("TemplateImg", cv2.WINDOW_NORMAL)
	cv2.namedWindow("FusedImg", cv2.WINDOW_NORMAL)
	cv2.imshow("QueryImg", q_img)
	cv2.imshow("TemplateImg", t_img)
	cv2.imshow("FusedImg", fusedImg)
	cv2.waitKey(0)
	
			


