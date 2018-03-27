import numpy as np
import copy
import sys
from ale_python_interface import ALEInterface
import cv2
import time
import random

class emulator:
	def __init__(self, rom_name, vis):
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt("max_mum_frames_per_episode")
		self.ale.setInt("random_seed", 123)
		self.ale.setInt("frame_skip", 4)
		self.ale.loadROM('roms/' + rom_name)
		self.legal_actions = self.ale.getMinimalActionSet()
		self.action_map = dict()
		for i in range(len(self.legal_actions)):
			self.action_map[self.legal_actions[i]] = i
		
		print self.legal_actions
		self.screen_width, self.screen_height = self.ale.getScreenDims()
		print("width/height: "+ str(self.screen_width) + "/" + str(self.screen_height))
		self.vis = vis
		if vis:
			cv2.startWindowThread()
			cv2.namedWindow("preview")
			
	def get_image(self):
		# numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
		# self.ale.getScreenRGB(numpy_surface)
		# image = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
		image = self.ale.getScreenRGB()
		image = np.reshape(image, (self.screen_height, self.screen_width, 3))
		return image
	
	def newGame(self):
		self.ale.reset_game()
		return self.get_image(), 0, False
	
	def next(self, action_indx):
		reward = self.ale.act(action_indx)
		nextstate = self.get_image()
		if self.vis:
			cv2.imshow('preview', nextstate)
		return nextstate, reward, self.ale.game_over()
	
	def train(self):
		for episode in range(10):
			total_reward = 0
			frame_number = 0
			while not self.ale.game_over():
				a = self.legal_actions[random.randrange(len(self.legal_actions))]
				# Apply an action and get the resulting reward
				reward = self.ale.act(a);
				total_reward += reward
				screen = self.ale.getScreenRGB()
				screen = np.array(screen).reshape([self.screen_height, self.screen_width, -1])
				frame_number = self.ale.getEpisodeFrameNumber()
				cv2.imshow("screen", screen/255.0)
				cv2.waitKey(0)
				
			self.ale.saveScreenPNG("test_"+str(frame_number)+".png")
			print('Episode %d ended with score: %d' % (episode, total_reward))
			print('Frame number is : ', frame_number)
			self.ale.reset_game()

