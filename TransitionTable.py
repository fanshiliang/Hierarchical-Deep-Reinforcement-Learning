import numpy as np
import random


class TransitionTable:
	def __init__(self, args):
		self.stateDim = args['stateDim']
		self.numActions = args['numActions']
		self.histLen = args['histLen']
		self.maxSize = args['maxSize']
		self.bufferSize = args['bufferSize']
		self.histType = "linear"
		self.histSpacing = args['histSpacing']
		self.zeroFrames = 1
		self.nonTermProb = args['nonTermProb']
		self.nonEventProb =  1
		self.numEntries = 0
		self.insertIndex = 0
		self.histIndices = {}
		
		if self.histType == 'linear':
			# history is the last histlen frames
			self.recentMemSize = self.histSpacing * self.histLen
			for i in range(1, self.histLen+1):
				self.histIndices[i] = i*self.histSpacing
		
		self.s = np.zeros([self.maxSize, 84, 84], dtype='float') # image dimensions
		self.a = np.zeros(self.maxSize, dtype='float')
		self.t = np.zeros(self.maxSize, dtype='float')
		self.r = np.zeros(self.maxSize, dtype='float')
		self.subgoal_dims = args['subgoal_dims'] * 9    ## total number of objects
		self.subgoal = np.zeros([self.maxSize, self.subgoal_dims])
		self.action_encodings = np.zeros([self.numActions,self.numActions],dtype='int')
		np.fill_diagonal(self.action_encodings, 1)
		
		self.recent_s = np.array([])
		self.recent_a = np.array([])
		self.recent_t = np.array([])
		self.recent_subgoal = np.array([])
		
		s_size = self.stateDim*self.histLen
		self.buf_a = np.zeros(self.bufferSize,dtype='float')
		self.buf_r = np.zeros([self.bufferSize,2],dtype='float')
		self.buf_term = np.zeros(self.bufferSize,dtype='float')
		self.buf_s = np.zeros([self.bufferSize, s_size],dtype='float')
		self.buf_s2     = np.zeros([self.bufferSize, s_size],dtype='float')
		self.buf_subgoal = np.zeros([self.bufferSize, self.subgoal_dims],dtype='float')
		self.buf_subgoal2 = np.zeros([self.bufferSize, self.subgoal_dims],dtype='float')
  
	def reset(self):
		self.numEntries = 0
		self.insertIndex = 0
	
	def size(self):
		return self.numEntries
	
	def empty(self):
		return self.numEntries == 0
	
	def fill_buffer(self):
		self.buf_index = 1
		for buf_index in range(1, self.bufferSize):
			s, a, r, s2, term, subgoal, subgoal2 = self.sample_one(1)
			self.buf_s[buf_index] = s[:]
			self.buf_a[buf_index] = a
			self.buf_subgoal[buf_index] = subgoal
			self.buf_subgoal2[buf_index] = subgoal2
			self.buf_r[buf_index] = r
			self.buf_s2[buf_index] = s2[:]
			self.buf_term[buf_ind] = term
		self.buf_s = self.buf_s / 255
		self.buf_s2 = self.buf_s2 / 255
	
	def sample_one(self):
		valid = False
		while not valid:
			index = random.randint(2, self.numEntries-self.recentMemSize)
			if self.t[index+self.recentMemSize-1] == 0:
				valid = True
			if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 \
					and random.uniform(0,1) > self.nonTermProb:
				valid = False
			if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 \
					and self.r[index+self.recentMemSize-1] \
					and random.uniform(0,1) > self.nonTermProb:
				valid = False
				
		return self.get(index)
	
	def sample(self, b_size):
		if b_size != None:
			batch_size = b_size
		else:
			batch_size = 1
		if not self.buf_index or self.buf_index + batch_size -1 > self.bufferSize:
			self.fill_buffer()
		index = self.buf_index
		self.buf_index = self.buf_index + batch_size
		range = [index, index+batch_size-1]
		return buf_s[range[0]:range[1]], buf_a[range[0]:range[1]], buf_r[range[0]:range[1]], buf_s2[range[0]:range[1]],\
		       buf_term[range[0]:range[1]], buf_subgoal[range[0]:range[1]], buf_subgoal2[range[0]:range[1]]
	
	def concatFrames(self,index, use_recent):
		if use_recent:
			l_s, l_t, l_subgoal = self.recent_s, self.recent_t, self.recent_subgoal[-1]
		else:
			l_s, l_t, l_subgoal = self.s, self.t, self.subgoal[index]
		fullstate = np.copy(l_s[1])
		fullstate = np.zeros([self.histLen, 84, 84 ,1], dtype='float')
		print "fullstate shape : ", fullstate.shape
		
		# zero out frames from all but the most recent episode
		zero_out = False
		episode_start = self.histLen
		
		print "histlen: ", self.histLen
		print "histIndices : ", self.histIndices
		for i in range(self.histLen-1, 0, -1):
			if not zero_out:
				print "lower range: ", index+self.histIndices[i]-1
				print "upper range: ", index+self.histIndices[i+1]-1
				for j in range(index+self.histIndices[i]-1, index+self.histIndices[i+1]-1):
					if l_t[j] == 1:
						zero_out = True
						break
			
			if zero_out:
				fullstate[i] = fullstate[i] * 0
			else:
				episode_start = i
		
		if self.zeroFrames == 0:
			episode_start = 1
		
		for i in range(episode_start-1, self.histLen):
			fullstate[i] = np.copy(l_s[index+self.histIndices[i]-1])
		
		return fullstate, l_subgoal
		
	def concatActions(self, index, use_recent):
		act_hist = np.zeros([self.histlen, self,numActions], dtype='float')
		if use_recent:
			l_a, l_t = np.copy(self.recent_a), np.copy(self.recent_t)
		else:
			l_a, l_t = np.copy(self.a), np.copy(self.t)
		zero_out = False
		episode_start = self.histLen
		
		for i in range(self.histLen, 0, -1):
			if not zero_out:
				for j in range(index+self.histIndices[i]-1, index+self.histIndices[i+1]-1):
					if l_t[j] == 1:
						zero_out = True
						break
			
			if zero_out:
				act_hist[i] = act_hist[i] * 0
			else:
				episode_start = i
		
		if self.zeroFrames == 0:
			episode_start = 1
		
		for i in range(episode_start, self.histlen+1):
			act_hist[i] = self.action_encodings[l_a[index+self.histIndices[i]-1]]
	
	def get_recent(self):
		fullstate, subgoal = self.concatFrames(1, True)
		return fullstate / 255, subgoal
	
	def get(self,index):
		s, subgoal = self.concatFrames(index, False)
		s2, subgoal2 = self.concatFrames(index+1, False)
		ar_index = index+self.recentMemSize-1
		return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1], self.subgoal[ar_index], self.subgoal[ar_index+1]

	def add(self, l_s, l_a, l_r, l_term, l_subgoal):
		if self.numEntries < self.maxSize:
			self.numEntries = self.numEntries + 1
		 # always insert at next index then wrap around
		self.insertIndex = self.insertIndex + 1
		# overwrite oldest experience once at capacity
		if self.insertIndex > self.maxSize:
			self.insertIndex = 1
		
		# overwrite (s,a,r,t) at insertIndex
		self.s[self.insertIndex] = np.copy(l_s) * 255
		self.a[self.insertIndex] = l_a
		self.r[self.insertIndex] = l_r
		self.subgoal[self.insertIndex] = l_subgoal
		if term:
			self.t[self.insertIndex] = 1
		else:
			self.t[self.insertIndex] = 0
	
	def add_recent_state(self, s, term, subgoal):
		l_s = np.copy(s) * 255
		l_subgoal = np.copy(subgoal)
		self.recent_s = self.recent_s.tolist()
		self.recent_subgoal = self.recent_subgoal.tolist()
		self.recent_t = self.recent_t.tolist()
		if len(self.recent_s) == 0:
			for i in range(1, self.recentMemSize+1):
				self.recent_s.append(np.copy(s)*0)
				self.recent_t.append(1)
				self.recent_subgoal.append(np.copy(subgoal)*0)
		self.recent_s.append(l_s.tolist())
		self.recent_subgoal.append(l_subgoal.tolist())
		if term:
			self.recent_t.append(1)
		else:
			self.recent_t.append(0)
		
		# keep recentMensize
		if len(self.recent_s) > self.recentMemSize:
			self.recent_s.pop(0)
			self.recent_t.pop(0)
			self.recent_subgoal.pop(0)
		
		self.recent_s = np.array(self.recent_s)
		self.recent_t = np.array(self.recent_t)
		self.recent_subgoal = np.array(self.recent_subgoal)
	
	def add_recent_action(self,a):
		self.recent_a = self.recent_a.tolist()
		if len(self.recent_a) == 0:
			for i in range(self.recentMemSize):
				self.recent_a.append(1)
		self.recent_a.append(a)
		if len(self.recent_a) > self.recentMemSize:
			self.recent_a.pop(0)
		
		self.recent_a = np.array(self.recent_a)
		