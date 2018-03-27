import zmq
import json
import tensorflow as tf
import cv2
import random
import numpy as np
import math
from preprocess import *
from TransitionTable import *
from h_DQN import *

# ZMQ server
port = 5550
context = zmq.Context()
print "Connecting to server..."
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:%s" % port)
meta_controller = {
	# Model backups
	'action_num': 6,
	'input_dimension': [84,84,4],
	'load_file': None,
	'save_file': None,
	'save_interval' : 10000,
	
	# Training parameters
	'train_start': 5000,    # Episodes before training starts
	'batch_size': 32,       # Replay memory batch size
	'mem_size': 100000,     # Replay memory size
	
	'discount': 0.95,       # Discount rate (gamma value)
	'lr': .0002,            # Learning reate
	# 'rms_decay': 0.99,    # RMS Prop decay (switched to adam)
	# 'rms_eps': 1e-6,      # RMS Prop epsilon (switched to adam)
	
	# Epsilon value (epsilon-greedy)
	'eps': 1.0,             # Epsilon start value
	'eps_final': 0.1,       # Epsilon end value
	'eps_step': 10000      # Epsilon steps between start and end (linear)
}
controller = {
	# Model backups
	'action_num': 18,
	'input_dimension': [84,84,4],
	'load_file': None,
	'save_file': None,
	'save_interval' : 10000,
	
	# Training parameters
	'train_start': 5000,    # Episodes before training starts
	'batch_size': 32,       # Replay memory batch size
	'mem_size': 100000,     # Replay memory size
	
	'discount': 0.95,       # Discount rate (gamma value)
	'lr': .0002,            # Learning reate
	# 'rms_decay': 0.99,    # RMS Prop decay (switched to adam)
	# 'rms_eps': 1e-6,      # RMS Prop epsilon (switched to adam)
	
	# Epsilon value (epsilon-greedy)
	'eps': 1.0,             # Epsilon start value
	'eps_final': 0.1,       # Epsilon end value
	'eps_step': 10000      # Epsilon steps between start and end (linear)
}
class agent:
	def __init__(self, args):
		self.network_meta = h_DQN(params=meta_controller, meta=True)
		self.network = h_DQN(params=controller, meta=False)
		self.subgoal_dims = args['subgoal_dims']
		self.use_distance = args['use_distance']
		self.max_reward = args['max_reward']
		self.min_reward = args['min_reward']
		self.rescale_r = args['rescale_r']
		self.max_r = 1
		# self.prep = Preprocess()
		self.meta_args = {
			'n_actions' : 6,
			'stateDim' : 7056,
			'numActions' : 6,
			'maxSize' :  50000,
			'histType' : "linear",
			'histLen': 4,
			'histSpacing' : 1,
			'nonTermProb' : 1,
			'bufferSize' : 512,
			'subgoal_dims' : self.subgoal_dims
		}
		self.meta_transitions = TransitionTable(self.meta_args)
		
		self.transition_args = {
			'n_actions' : 18,
			'stateDim' : 7056,
			'histLen' : 4,
			'numActions' : 6,
			'maxSize' : 200000,
			'histType' : 'linear',
			'histSpacing' : 1,
			'nonTermProb': 1,
			"bufferSize" : 512,
			'subgoal_dims': self.subgoal_dims
			
		}
		self.transition = TransitionTable(self.transition_args)
		
		self.numSteps = 0  #Number of perceived states.
		self.lastState = None
		self.lastAction = None
		self.lastSubgoal = None
		
		self.subgoal_success = [0] * 8
		self.subgoal_total = [0] * 8
		self.global_subgoal_success = [0] * 8
		self.global_subgoal_total = [0] * 8
		
		self.subgoal_seq = []
		self.global_subgoal_seq = []
		
		# to keep track of dying position
		self.deathPosition = None
		self.DEATH_THRESHOLD = 15
		self.ignoreState = None
		self.metaignoreState = None
		
		# Q-learning parameters  ##########need more modification
		self.dynamic_discount = 0.99
		self.discount = 0.99 #Discount factor.
		self.discount_internal = 0.99 #Discount factor for internal rewards
		self.update_freq = 4
		
		# epsilon annealing
		self.ep_start   = 1
		self.ep         = self.ep_start # Exploration probability.
		self.ep_end     = 0.1
		self.ep_endt    = 1000000

		# Number of points to replay per learning step.
		self.n_replay =  1
		# Number of steps after which learning starts.
		self.learn_start = 50000
		self.meta_learn_start = 1000
		
		self.lastTerminal = None
		self.minibatch_size = 256
		self.metanumSteps = 0
		self.metalastState = None
		self.metalastAction = None
		self.metalastSubgoal = None
		self.metalastTerminal = None
		self.v_avg = 0 # V running average.
		self.tderr_avg = 0 # TD error running average.

		self.q_max = 1
		self.r_max = 1
	
	def preprocess(self, observation):
		print "observation shape: ", observation.shape
		observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
		observation = observation[26:110,:]
		print "observation after preprocess: ", observation.shape
		return np.reshape(observation, (84, 84, 1))
		
	def get_objects(self, state):
		cv2.imwrite('tmp_'+str(port)+'.png', state)
		socket.send("")
		msg = socket.recv()
		while msg == None:
			msg = socket.recv()
		print "message in get_objects: ", msg
		object_list = json.loads(msg)
		self.objects = object_list
		return object_list
	
	def pick_subgoal(self, state, metareward, terminal, testing, testing_ep):
		print "state shape: ",state.shape
		objects = self.get_objects(state)
		print "objects: ", objects
		subg = np.copy(objects[1]) * 0
		print "subg: ", subg
		ftrvec =  np.zeros(len(objects)*self.subgoal_dims)
		print "ftrvec: ", ftrvec
		ftrvec = np.concatenate((subg, ftrvec))
		print "ftrvec2: ", ftrvec
		# state = state[50:,:,:]
		# state = np.array([state])
		
		# set preprocess !!!!!!!
		state = self.preprocess(state)
		
		print "State shape: ", state.shape
		# print "State", state
		self.meta_transitions.add_recent_state(state[0],terminal,ftrvec)
		
		# Store transition s, a, r, s'
		if self.metalastState and not testing:
			self.meta_transitions.add(self.metalastState, self.metalastAction, np.array([metareward, metareward+0]), self.metalastTerminal, ftrvec, priority)
		
		curState, subgoal = self.meta_transitions.get_recent()
		print "curstate shape: ", curState.shape
		curState = curState.reshape([1, 4, 84, 84])
		
		# select action
		actionIndex = 1
		qfunc = None
		if not terminal:
			print "subgoal: ", subgoal
			actionIndex, qfunc = self.e_Greedy('meta', self.network_meta, curState, testing_ep, subgoal, self.metalastAction)
		
		self.meta_transitions.add_recent_action(actionIndex)
		
		# do some Q-learning updates
		if self.metanumSteps > self.meta_learn_start and not testing and self.metanumSteps % self.update_freq == 0:
			for i in range(self.n_replay):
				self.qLearnMinibatch(self.network_meta, self.target_network_meta, self.meta_transitions,\
									 self.dw_meta, self.w_meta, self.g_meta, self.g2_meta, self.tmp_meta,\
									 self.deltas_meta, false, self.meta_args.n_actions, true)
		if not testing:
			self.metanumSteps = self.metanumSteps + 1
			self.metalastState = np.copy(state)
			self.metalastAction = actionIndex
			self.metalastTerminal = terminal
		
		if self.meta_args['n_actions'] == 6:
			index = actionIndex + 2
		else:
			index = actionIndex + 5
		
		print "index: ", index
		subg = objects[index]
		
		if not terminal:
			self.subgoal_total[index] += 1
			self.global_subgoal_total[index] += 1
		
		ftrvec = np.zeros(len(objects)*self.subgoal_dims)
		ftrvec[index] = 1
		ftrvec[-1] = index  ################# might have some problem
		
		if terminal:
			self.global_subgoal_seq.append(self.subgoal_seq)
			self.subgoal_seq = []
		else:
			self.subgoal_seq.append(index)
		
		return np.concatenate((subg, ftrvec))
		
	def e_Greedy(self, mode, network, state, testing_ep, subgoal, lastsubgoal):
		# handle the learn start
		print "enter e_greedy: ", mode
		if mode == 'meta':
			learn_start = self.meta_learn_start
		else:
			learn_start = self.learn_start
		if testing_ep:
			self.ep = testing_ep
		else:
			self.ep = (self.ep_end + max(0, (self.ep_start - self.ep_end) *\
			                                  (self.ep_endt - max(0, self.numSteps - learn_start))/self.ep_endt))
		subgoal_id = subgoal[-1]
		if mode != 'meta' and subgoal_id != 6 and subgoal_id != 8:
			self.ep = 0.1
		
		n_actions = None
		if mode == 'meta':
			n_actions = self.meta_args['n_actions']
		else:
			n_actions = self.transition_args['n_actions']
		print "enter e_greedy"
		# epsilon greedy
		self.ep = -1 ## To do: delete this after testing network
		if random.uniform(0,1) < self.ep:
			if mode == 'meta':
				chosen_act = random.randint(0, n_actions-1)
				while chosen_act == lastsubgoal:
					chosen_act = random.randint(0, n_actions-1)
				return chosen_act, None
			else:
				return random.randInt(0, n_actions-1), None
		else:
			return self.greedy(network, n_actions, state, subgoal, lastsubgoal)

	def greedy(self, network, n_actions, state, subgoal, lastsubgoal):
		# turn single state into minibatch. Needed for convolutional nets
		if state.ndim == 2:
			state = state.reshape(1, state.shape[0], state.shape[1])
		if network.network_name == 'meta_controller':
			param = meta_controller
		else:
			param = controller
		subgoal = subgoal.reshape(1, self.subgoal_dims*9)
		print "State in greedy: ", state.shape
		print "subgoal in greedy", subgoal.shape
		# Q value from network
		print "state reshape: ", np.reshape(state,
		                                    (1, param['input_dimension'][0], param['input_dimension'][1], param['input_dimension'][2])).shape
		
		Q_pred = network.sess.run(
			network.y,
			feed_dict = {network.x: np.reshape(state,
			                                     (1, param['input_dimension'][0], param['input_dimension'][1], param['input_dimension'][2])),
			             network.q_t: np.zeros(1),
			             network.actions: np.zeros((1, param['action_num'])),
			             network.terminals: np.zeros(1),
			             network.rewards: np.zeros(1)})[0]
		maxq = Q_pred[0]
		besta = [0]
		if lastsubgoal == 0:
			maxq = q[1]
			besta = [1]
		for a in range(1, n_actions):
			if a != lastsubgoal:
				if Q_pred[a] > maxq:
					besta = [a]
					maxq = Q_pred[a]
				elif Q_pred[a] == maxq:
					besta.append(a)
		r = random.randint(0, len(besta)-1)
		
		print "besta[r]: ", besta[r]
		print 'Q_pred: ', Q_pred
		return besta[r], Q_pred
		
	def isGoalReached(self, subgoal, objects):
		agent = objects[0]
		
		# subgoal include both subgoal and all objects
		dist = math.sqrt(np.power((subgoal[0] - agent[0]),2) + np.power((subgoal[1] - agent[1]),2))
		# just a small threshold to indicat when agent meets subgoal
		if dist < 9:
			print 'subgoal reached ID: ', subgoal[-1]
			subg = subgoal[0:self.subgoal_dims]
			self.subgoal_success[int(subgoal[-1])] = self.subgoal_success[int(subgoal[-1])] + 1
			self.global_subgoal_success[int(subgoal[-1])] = self.global_subgoal_success[int(subgoal[-1])] + 1
			return True
		else:
			return False
	
	def intrinsic_reward(self, subgoal, objects):
		agent = objects[0]
		reward = 0
		if self.lastSubgoal and np.sum(np.absolute(self.lastSubgoal[2:self.subgoal_dims]-subgoal[2:self.subgoal_dims])):
			dist1 = math.sqrt(np.power((subgoal[0]-agent[0]),2) + np.power((subgoal[1]-agent[1]),2))
			dist2 = math.sqrt(np.power((self.lastSubgoal[0]-self.lastobjects[0][0]),2) + np.power((self.lastSubgoal[1]-self.lastobjects[0][1]),2))
			reward = dist2 - dist1
		else:
			reward = 0
		
		if not self.use_distance:
			# no intrinsic reward except for reaching the subgoal
			reward = 0
		return reward
		
	def perceive(self, subgoal, reward, rawstate, terminal, testing=False, testing_ep=None):
		# process state
		state = self.preprocess(rawstate)
		objects = self.get_objects(rawstate)
		
		if terminal:
			self.deathPosition = objects[0][0:2]
		goal_reached = self.isGoalReached(subgoal, objects)
		print "goal_reached: ", goal_reached
		intrinsic_reward = self.intrinsic_reward(subgoal, objects)
		print "intrinsic reward: ", intrinsic_reward
		
		if terminal:
			intrinsic_reward = intrinsic_reward - 200
		
		# penality for non-move
		intrinsic_reward = intrinsic_reward - 0.1
		
		if goal_reached:
			intrinsic_reward = intrinsic_reward + 50
		
		if self.max_reward:
			reward = min(reward, self.max_reward)
		if self.min_reward:
			reward = max(reward, self.min_reward)
		if self.rescale_r:
			self.r_max = max(self.r_max, reward)
			
		self.transition.add_recent_state(state, terminal, subgoal)
		
		# store transition s, a, r, s'
		if self.lastState and not testing and self.lastSubgoal:
			if self.ignoreState:
				self.ignoreState = None
			else:
				self.transition.add(self.lastState, self.lastAction, np.array([reward, intrinsic_reward])\
				                    ,self.lastTerminal, self.lastSubgoal)
		
		curState, subgoal = self.transition.get_recent()
		curState = curState.reshape([1, 4, 84, 84])
		
		actionIndex = 0
		qfunc = None
		if not terminal:
			actionIndex, qfunc = self.e_Greedy('lower', self.network, curState, testing_ep, subgoal, None)
		
		self.transition.add_recent_action(actionIndex)
		
		# Q_learning updates
		if self.numSteps > self.learn_start and not testing and self.numSteps % self.update_freq == 0:
			for i in range(0, self.n_replay):
				self.qLearnMinibatch(self.network, self.target_network, self.transition)
			
		if not testing:
			self.numSteps = self.numSteps + 1
		self.lastState = np.copy(state)
		self.lastAction = actionIndex
		self.lastTerminal = terminal
		
		if not terminal:
			self.lastSubgoal = subgoal
			if self.deathPosition:
				currentPosition = objects[0][0:2]
				if math.sqrt(np.power(currentPosition[0]-self.deathPosition[0], 2)+np.power(currentPosition[1]-self.deathPosition[1], 2)) < self.DEATH_THRESHOLD:
					self.lastSubgoal = None
				else:
					self.deathPosition = None
					self.ignoreState = 1
					
		self.lastobjects = objects
		
		# copy update target network
		# TODO
		if not terminal:
			return actionIndex, goal_reached, reward, reward+intrinsic_reward, qfunc
		else:
			return 0, goal_reached, reward, reward+intrinsic_reward, qfunc
		
	
	def qLearnMinibatch(self, network, target_network, tran_table,):
		s, a, r, s2, term, subgoals, subgoals2 = tran_table.sample(self.minibatch_size)
		if external_r:
			r = r[0]
			subgoals[]
		