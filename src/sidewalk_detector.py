#!/usr/bin/env python
import cv2
import copy
import rospy
import rosbag
import colorsys
import cv2.cv as cv
import numpy as np

from math import floor
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String


class SidewalkDetector():
	def __init__(self):
		# subscribe to the color image
		self.img_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
		
		# cv bridge 
		self.bridge = CvBridge()
		
		# create a sidewalk histogram list
		self.sidewalk_his = []

		# A threshold for identifying a single pixel belongs to sidewalk
		self.sidewalk_thre = 0.8

		# publishers		
		self.image_pub = rospy.Publisher('/sidewalk_detector/color/image_raw', Image, queue_size=10)
		self.pc_pub = rospy.Publisher('/sidewalk_detector/depth/points_in', PointCloud2, queue_size=10)
	
	def image_callback(self, ros_image):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
		except CvBridgeError as e:
			print(e)

		# the images are upside down	
		(self.h, self.w) = cv_image.shape[:2]
		center = (self.w / 2, self.h / 2)
		M = cv2.getRotationMatrix2D(center, 180, 1.0)
		cv_image = cv2.warpAffine(cv_image, M, (self.w, self.h))
		
		# identified martix
		id_img = np.zeros((cv_image.shape[0],cv_image.shape[1]))

		# identified image
		self.img = copy.copy(cv_image)

		# prepare the initial histogram
		if self.sidewalk_his == []:
			self.get_init_histogram(cv_image)
		else:
			self.update_sidewalk_his(cv_image)
			for i in range(0, self.h):
				for j in range(0, self.w):
					is_sidewalk = self.identify(cv_image[i, j])
					if is_sidewalk:
						id_img[i, j] = 1
					else:
						id_img[i, j] = 0

			self.update_bacground_his(cv_image, id_img)

		for i in range(0, self.h):
			for j in range(0, self.w):
				if id_img[i, j] == 1:
					self.img[i, j, 0] = 255
					self.img[i, j, 1] = 0
					self.img[i, j, 2] = 0

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "rgb8"))
		except CvBridgeError as e:
			print(e)

	def identify(self, pixel):
		(r, g, b) = (pixel[0]/255.0, pixel[1]/255.0, pixel[2]/255.0)
		(h, s, v) = colorsys.rgb_to_hsv(r, g, b)
		x, y = self.normalize(h, s)

		fre_s = []
		for his in self.sidewalk_his:
			fre_s.append(his[x, y] / self.wh * self.ww)

		# frequncy in sidewalk histograms
		fre_s = max(fre_s)
		
		# frequency in background histogram
		fre_b = self.background_his[x, y] / self.bp  

		# probability of a single pixel belong to sidewalk
		prob_side = fre_s / (fre_s + fre_b)

		if prob_side > self.sidewalk_thre:
			return True
		else:
			return False

	def get_init_histogram(self, cv_image):
		# window size
		self.wh = 80
		self.ww = 200

		# get the sidewalk frame
		self.mid = self.w / 2
		self.sidewalk_frame = cv_image[(self.h - self.wh) : self.h, (self.mid - self.ww / 2) : (self.mid + self.ww / 2)]

		# sidewalk histogram
		self.sidewalk_his.append(np.zeros((10, 10, 1)))
		for i in range(0, self.wh):
			for j in range(0, self.ww):
				(r, g, b) = (self.sidewalk_frame[i, j, 0]/255.0, self.sidewalk_frame[i, j, 1]/255.0, self.sidewalk_frame[i, j, 2]/255.0)
				(h, s, v) = colorsys.rgb_to_hsv(r, g, b)
				x, y = self.normalize(h, s)
				self.sidewalk_his[0][x, y, 0] += 1

		# background histogram
		self.background_his = np.zeros((10, 10, 1))

		# background total pixel counts
		self.bp = 0

		for i in range(0, self.h):
			for j in range(0, self.w):
				if self.h - self.wh < i < self.h and (self.mid - self.ww / 2) < j < (self.mid + self.ww / 2):
					continue
				(r, g, b) = (cv_image[i, j, 0]/255.0, cv_image[i, j, 1]/255.0, cv_image[i, j, 2]/255.0)
				(h, s, v) = colorsys.rgb_to_hsv(r, g, b)
				x, y = self.normalize(h, s)
				self.bp += 1
				self.background_his[x, y, 0] += 1

	def update_sidewalk_his(self, cv_image):
		self.sidewalk_frame = cv_image[(self.h - self.wh) : self.h, (self.mid - self.ww / 2) : (self.mid + self.ww / 2)]
		if len(self.sidewalk_his) < 4:
			self.sidewalk_his.append(np.zeros((10, 10, 1)))
			for i in range(0, self.wh):
				for j in range(0, self.ww):
					(r, g, b) = (self.sidewalk_frame[i, j, 0]/255.0, self.sidewalk_frame[i, j, 1]/255.0, self.sidewalk_frame[i, j, 2]/255.0)
					(h, s, v) = colorsys.rgb_to_hsv(r, g, b)
					x, y = self.normalize(h, s)
					self.sidewalk_his[-1][x, y, 0] += 1
		else:
			new_his = np.zeros((10, 10, 1))
			for i in range(0, self.wh):
				for j in range(0, self.ww):
					(r, g, b) = (self.sidewalk_frame[i, j, 0]/255.0, self.sidewalk_frame[i, j, 1]/255.0, self.sidewalk_frame[i, j, 2]/255.0)
					(h, s, v) = colorsys.rgb_to_hsv(r, g, b)
					x, y = self.normalize(h, s)
					new_his[x, y, 0] += 1
			dis = []
			for i in range(0, len(self.sidewalk_his)):
				dis.append(self.normdis(new_his, self.sidewalk_his[i]))
			minimum = dis.index(min(dis))
			print 'minimum', minimum
			del(self.sidewalk_his[0])
			self.sidewalk_his.append(new_his)

	def update_bacground_his(self, cv_image, id_img):
		self.bp = 0
		for i in range(0, self.h):
			for j in range(0, self.w):
				if id_img[i, j] == 1:
					continue
				(r, g, b) = (cv_image[i, j, 0]/255.0, cv_image[i, j, 1]/255.0, cv_image[i, j, 2]/255.0)
				(h, s, v) = colorsys.rgb_to_hsv(r, g, b)
				x, y = self.normalize(h, s)
				self.bp += 1
				if self.bp > 3000:
					break
				self.background_his[x, y, 0] += 1

	def normdis(self, his1, his2):
		dis = 0
		if his1.shape != his2.shape:
			print(e)
		for i in range(his1.shape[0]):
			for j in range(his2.shape[1]):
				dis += (his1[i, j] - his2[i, j]) ** 2
		return dis

	def normalize(self, h, s):
		x = int(floor(h * 10))
		y = int(floor(s * 10))
		if x > 1:
			x = 1
		if y > 1:
			y = 1
		return x, y

if __name__ == '__main__':
    rospy.init_node('sidewalk_detector')
    
    # create a detector
    sidewalk_detector = SidewalkDetector()

    while not rospy.is_shutdown():
        pass

	rospy.spin()