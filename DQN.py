import numpy as np
import tensorflow as tf

class DQN:
	def __init__(self, params, name):
		self.params = params
		self.network_name = 'qnet '+name
		self.sess = tf.Session()
		self.q_t = tf.placeholder('float',[None], name=self.network_name+'_q_t')
		self.rewards = tf.placeholder('float',[None], name=self.network_name+'_reward')
		self.actions = tf.placeholder('float', [None,params['action_num']], name=self.network_name+'_actions')
		self.x = tf.placeholder('float', [None, params['input_dimension'][0], params['input_dimensition'][1],
		                                  params['input_dimension'][2]], name=self.network_name + '_x')
		self.terminals = tf.placeholder('float', [None], name=self.network_name+'_terminals')
		
		# conv 1
		layer_name = 'conv1' ;  filter_size = 8 ; channels = 4 ; filters = 32 ; stride = 4
		self.w1 = tf.Variable(tf.random_normal([filter_size, filter_size, channels, filters], stddev=0.01),
		                      name=self.network_name+'_'+layer_name+'_weight')
		self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name+"_"+layer_name+'_bias')
		self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',
		                       name=self.network_name+'_'+layer_name+'_convs')
		self.o1 = tf.nn.relu(tf.add(self.c1, self.b1), name=self.network_name+'_'+layer_name+'_activations')
		
		# conv2
		layer_name = 'conv2' ; filter_size = 4 ; channels = 32 ; filters = 64 ; stride = 2
		self.w2 = tf.Variable(tf.random_normal([filter_size, filter_size, channels, filters], stddev=0.01),
		                      name=self.network_name+'_'+layer_name+'_weights')
		self.b2 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name+'_'+layer_name+'_bias')
		self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',
		                       name=self.network_name+'_'+layer_name+'_convs')
		self.o2 = tf.nn.relu(tf.add(self.c2, self.b2), name=self.network_name+'_'+layer_name+'_activations')
		
		# conv 3
		layer_name = 'conv3' ; filter_size = 3 ; channels = 64 ; filters = 64 ; stride = 1
		self.w3 = tf.Variable(tf.random_normal([filter_size, filter_size, channels, filters], stddev=0.01),
		                      name=self.network_name+'_'+layer_name+'_weights')
		self.b3 = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name+'_'+layer_name+'_activations')
		self.c3 = tf.nn.conv2d(self.o2, self.w3, strides=[1, stride, stride, 1], padding='SAME',
		                       name=self.network_name+'_'+layer_name+'_convs')
		self.o3 = tf.nn.relu(tf.add(self.c3, self.b3), name=self.network_name+'_'+layer_name+'_activations')
		
		# flat
		o3_shape = self.o3.get_shape().as_list()
		
		# fully connected layer
		layer_name = 'fc1' ; hiddens = 256 ; dim = o3_shape[1]*o3_shape[2]*o3_shape[3]
		self.o3_flat = tf.reshape(self.o3, [-1, dim], name=self.network_name+'_'+layer_name+'_input_flat')
		self.w4 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01),
		                      name=self.network_name+'_'+layer_name+'_weight')
		self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name+'_'+layer_name+'bias')
		self.ip4 = tf.add(tf.matmul(self.o3_flat, self.w4), self.b4, name=self.network+'_'+layer_name+'_outputs')
		self.o4 = tf.nn.relu(self.ip3, name=self.network_name+'_'+layer_name+'_activations')
		
		# fully connected layer 2
		layer_name = 'fc2' ; hiddens = params['num_act'] ; dim = 256
		self.w5 = tf.Variable(tf.random_normal([dim, hiddens], stddev=0.01), name=self.network_name+'_'+layer_name+'_weights')
		self.b5 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name=self.network_name+'_'+layer_name+'_bias')
		self.y = tf.add(tf.matmul(self.o4, self.w5), self.b5, name=self.network_name+'_'+layer_name+'_output')
		
		# Q, cost
		self.discount = tf.constant(self.params['discount'])
		self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.descount, self.q_t)))
		self.Q_pred = tf.reduce_sum(tf.multiply(self.y, self.actions), reduction_indices=1)
		self.loss = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred),2))
		
		# save or load file
		if self.params['load_file'] is not None:
			self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]), name='global_step', trainable=False)
		else:
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
		
		# optimizer
		self.optim = tf.train.AdamOptimizer(self.params['lr']).minimize(self.loss, global_step=self.global_step)
		self.saver = tf.train.Saver(max_to_keep=0)
		
		self.sess.run(tf.global_variables_initializer())
		
		if self.params['load_file'] is not None:
			print('Loading checkpoint...')
			self.saver.restore(self.sess, self.params['load_file'])
		
	def train(self, bat_s, bat_a, bat_t, bat_n, bat_r):
		'''
		:param bat_s: current state
		:param bat_a: action
		:param bat_t: terminal
		:param bat_n: next state
		:param bat_r: rewards
		:return:
		'''
		feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.rewards: bat_r, self.actions: bat_a, self.terminals: bat_t}
		q_t = self.sess.run(self.y, feed_dict=feed_dict)
		q_t = np.amax(q_t, axis=1)
		feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals: bat_t, self.rewards: bat_r}
		_,cnt,loss = self.sess.run([self.optim, self.global_step, self.loss],feed_dict=feed_dict)
		return cnt,loss
	
	def save_session(self, filename):
		self.saver.save(self.sess.filename)