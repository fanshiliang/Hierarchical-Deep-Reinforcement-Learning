from emulator import *
from agent import *
from storage import *
import time

# param setting section
STEPTHROUGH = True # print information after every action
META_AGENT = True
SUBGOAL_SCREEN = True
MAX_STEPS_EPISODE = 5000
steps = 100000   # total steps for training
step = 0
start_time = time.asctime(time.localtime(time.time()))
reward_counts = []
episode_counts = []
time_history = {}
v_history = []
qmax_history = []
td_history = []
reward_history = []
total_reward = None
nrewards = None
nepisodes = None
episode_reward = None

agent_params = {
	'subgoal_dims' : 7,
	'use_distance' : True,
	'max_reward' : 1000,
	'min_reward' : -1000,
	'rescale_r' : True
}

# param setting section ending

# training initial section
ag = agent(agent_params)
emu = emulator("montezuma_revenge.bin", False)
rawstate = emu.newGame()
reward = 0
terminal = False
learn_start = ag.learn_start
time_history[1] = 0
# initial section ending

if META_AGENT:
	subgoal = ag.pick_subgoal(rawstate, 0, False, False, None)
	

action_list = ['no-op', 'fire', 'up', 'right', 'left', 'down', 'up-right','up-left','down-right','down-left',
               'up-fire', 'right-fire','left-fire', 'down-fire','up-right-fire','up-left-fire',
               'down-right-fire', 'down-left-fire']

death_counter = 0 # handle bug in MZ

episode_step_counter = 0
metareward = 0
SAVE_NET_EXIT = False
cum_metareward = 0
numepisodes = 0

while step < steps:
	step = step + 1
	subgoal_screen = np.copy(rawstate)
	print "subgoal_screen: ", subgoal_screen.shape
	
	# mark subgoal region in the image
	if SUBGOAL_SCREEN:
		for row in range(int(30+subgoal[0]-5), int(30+subgoal[0]+6)):
			for col in range(int(subgoal[1]-5), int(subgoal[1]+6)):
				subgoal_screen[row][col] = 1
	
	action_index, isGoalReached, reward_ext, reward_tot, qfunc = ag.perceive(subgoal, reward, subgoal_screen, terminal)
	metareward = metareward + reward_ext
	if STEPTHROUGH:
		print("Reward Ext", reward_ext)
		print("Reward Tot", reward_tot)
		print("Q-func")
		if qfunc:
			for i in range(len(action_list)):
				print(action_list[i], qfunc[i])
		print("Action", action_index)
	
	# end of game after door opens
	if metareward > 100:
		terminal = true
		death_counter = 4
		
	# game over, get next game
	if not terminal and episode_step_counter < MAX_STEPS_EPISODE:
		screen, reward, terminal = emu.next(action_index)
		if not terminal:
			screen, temp_reward, terminal = emu.next(0)
		reward = reward + temp_reward
		episode_step_counter = episode_step_counter + 1
		prev_Q = qfunc
	else:
		death_counter = death_counter + 1
		if META_AGENT:
			subgoal = ag.pick_subgoal(screen, metareward, true, false, 0)
			if metareward > 0:
				print("MetaR: ", metareward)
			cum_metareward = cum_metareward + metareward
			metareward = 0
		# start a new game
		screen, reward, terminal = emu.newGame()
		
		if death_counter == 5:
			screen, reward, terminal = emu.newGame()
			death_counter = 0
			numepisodes = numepisodes + 1
		
		new_game = True
		isGoalReached = True  # new game so reset goal
		episode_step_counter = 0
	
	if isGoalReached:
		if metareward > 0:
			print("METAREWARD: ", metareward, "| subgoal: ", subgoal[-1])
		subgoal = ag.pick_subgoal(screen, metareward, terminal, false, 0)
		cum_metareward = cum_metareward + metareward
		metareward = 0
		isGoalReached = False
	