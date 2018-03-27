import numpy as np
import tensorflow as tf
import cv2

class Preprocess:
	def __init__(self):
		"""
		:param params: essential parameters for network generation
		:param meta: meta == True, create meta_controller. meta == False, create controller network
		"""
		self.network_name = 'qnet_preprocess'
		self.sess = tf.Session()
		self.input = tf.placeholder('float', shape=[1, 160, 160, 1])
		
		# maxpooling 2x2
		self.output = tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
		                             strides=[1, 2, 2, 1], padding='SAME')
		# ,pool_size=[2,2],strides=2,padding='SAME',data_format='channels_last'
	def process(self, input):
		input = cv2.cvtColor(input,cv2.COLOR_RGB2GRAY)
		feed_dict= {self.input: input}
		output = self.sess.run(self.output, feed_dict=feed_dict)
		return output