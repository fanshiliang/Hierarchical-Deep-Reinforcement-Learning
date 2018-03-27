from h_DQN import *
from storage import *
from emulator import *
import tensorflow as tf
import numpy as np
import time
from ale_python_interface import ALEInterface
import cv2
from scipy import misc
import gc # garbage collector

gc.enable()

params = {
	'ckpt_file':None,
	'num_episodes': 250000,
	'rms_decay':0.99,
	'rms_eps':1e-6,
	'db_size': 1000000,
	'batch': 32,
	'num_act': 0,
	'input_dims' : [210, 160, 3],
	'input_dims_proc' : [84, 84, 4],
	'episode_max_length': 100000,
	'learning_interval': 1,
	'eps': 1.0,
	'eps_step':1000000,
	'discount': 0.95,
	'lr': 0.0002,
	'save_interval':20000,
	'train_start':100,
	'eval_mode':False
}

class HRL:
	def __init__(self, params):
		print 'initializing......'
		self.params = params
		self.sess = tf.Session()
		self.storage = storage(self.params['db_size'], self.params['input_dims_proc'])
		self.engine = emulator(rom_name='breakout.bin',vis=True)
		self.params['num_act'] = len(self.engine.legal_actions)
		self.build_nets()
		self.Q_global = 0
		self.cost_disp = 0
	
	def build_nets(self):
		print 'building networks'
		self.meta_net = h_DQN(self.params, 'meta')
		self.control_net = h_DQN(self.params, 'control')
	
	def start(self):
		print 'start training......'
		count = self.control_net.sess.run(self.control_net.global_step)
		print 'global step = ' + str(count)
		local_count = 0
		for numeps in range(self.params['num_episodes']):
			self.Q_global = 0
			state_proc = np.zeros([84,84,4])
			state_proc_old = None
			action = None
			terminal = None
			delay = 0
			state = self.engine.newGame()
			
