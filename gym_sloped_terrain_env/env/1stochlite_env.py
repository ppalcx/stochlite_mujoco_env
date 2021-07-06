import numpy as np
import gym
from gym import spaces
import gym_sloped_terrain.envs.trajectory_generator as trajectory_generator
import math
import random
from collections import deque
import pybullet
import gym_sloped_terrain.envs.bullet_client as bullet_client
import pybullet_data
import gym_sloped_terrain.envs.planeEstimation.get_terrain_normal as normal_estimator
import matplotlib.pyplot as plt
from utils.logger import DataLog
import os
import gym_sloped_terrain.envs.pybullet_func as Simulator
#import gym_sloped_terrain.envs.mujoco_func_env as Simulator


# LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
# KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076] #hip
# KNEE_CONSTRAINT_POINT_LEFT = [0.0,0.0,-0.077] #knee
RENDER_HEIGHT = 720 
RENDER_WIDTH = 960 
PI = np.pi

class StochliteEnv(gym.Env):

	def __init__(self,
				 render = False,
				 on_rack = False,
				 gait = 'trot',
				 phase =   [0, PI, PI,0],#[FR, FL, BR, BL] 
				 action_dim = 20,
				 end_steps = 1000,
				 stairs = False,
				 downhill =False,
				 seed_value = 100,
				 wedge = False,
				 IMU_Noise = False,
				 deg = 5): # deg = 5

		self.robot=Simulator.StochliteLoader(render=render, on_rack=on_rack)
		print(self.robot)
		self._is_stairs = stairs
		self._is_wedge = wedge
		self._is_render = render
		self._on_rack = on_rack
		self.rh_along_normal = 0.24

		self.seed_value = seed_value
		random.seed(self.seed_value)

		# if self._is_render:
		# 	self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
		# else:
		# 	self._pybullet_client = bullet_client.BulletClient()

		# self._theta = 0

		self._frequency = 2.5 # originally 2.5, changing for stability
		self.termination_steps = end_steps
		self.downhill = downhill

		#PD gains
		self._kp = 400
		self._kd = 10

		self.dt = 0.01
		self._frame_skip = 50
		self._n_steps = 0
		self._action_dim = action_dim

		self._obs_dim = 11

		self.action = np.zeros(self._action_dim)

		self._last_base_position = [0, 0, 0]
		self.last_rpy = [0, 0, 0]
		self._distance_limit = float("inf")

		self.current_com_height = 0.25 # 0.243
		
		#wedge_parameters
		self.wedge_start = 0.5 
		self.wedge_halflength = 2

		if gait is 'trot':
			phase = [0, PI, PI, 0]
		elif gait is 'walk':
			phase = [0, PI, 3*PI/2 ,PI/2]
		self._trajgen = trajectory_generator.TrajectoryGenerator(gait_type=gait, phase=phase)
		self.inverse = False
		self._cam_dist = 1.0
		self._cam_yaw = 0.0
		self._cam_pitch = 0.0

		self.avg_vel_per_step = 0
		self.avg_omega_per_step = 0

		self.linearV = 0
		self.angV = 0
		self.prev_vel=[0,0,0]
		self.prev_ang_vels = [0, 0, 0] # roll_vel, pitch_vel, yaw_vel of prev step
		self.total_power = 0

		self.x_f = 0
		self.y_f = 0

		self.clips=7

		self.friction = 0.6
		# self.ori_history_length = 3
		# self.ori_history_queue = deque([0]*3*self.ori_history_length, 
		#                             maxlen=3*self.ori_history_length)#observation queue

		self.step_disp = deque([0]*100, maxlen=100)
		self.stride = 5

		self.incline_deg = deg
		self.incline_ori = 0

		self.prev_incline_vec = (0,0,1)

		self.terrain_pitch = []
		self.add_IMU_noise = IMU_Noise

		self.INIT_POSITION =[0,0,0.3] # [0,0,0.3], Spawning stochlite higher to remove initial drift
		self.INIT_ORIENTATION = [0, 0, 0, 1]

		self.support_plane_estimated_pitch = 0
		self.support_plane_estimated_roll = 0

		self.pertub_steps = 0
		self.x_f = 0
		self.y_f = 0

		## Gym env related mandatory variables
		self._obs_dim = 10 #[roll, pitch, roll_vel, pitch_vel, yaw_vel, SP roll, SP pitch, cmd_xvel, cmd_yvel, cmd_avel]
		observation_high = np.array([np.pi/2] * self._obs_dim)
		observation_low = -observation_high
		self.observation_space = spaces.Box(observation_low, observation_high)

		action_high = np.array([1] * self._action_dim)
		self.action_space = spaces.Box(-action_high, action_high)

		self.commands = np.array([0, 0, 0]) #Joystick commands consisting of cmd_x_velocity, cmd_y_velocity, cmd_ang_velocity
		self.max_linear_xvel = 0.5 #0.4, made zero for only ang vel # calculation is < 0.2 m steplength times the frequency 2.5 Hz
		self.max_linear_yvel = 0.25 #0.25, made zero for only ang vel # calculation is < 0.14 m times the frequency 2.5 Hz
		self.max_ang_vel = 3.5 #considering less than pi/2 steer angle # less than one complete rotation in one second
		self.max_steplength = 0.2 # by the kinematic limits of the robot
		self.max_steer_angle = PI/2 #plus minus PI/2 rads
		self.max_x_shift = 0.1 #plus minus 0.1 m
		self.max_y_shift = 0.14 # max 30 degree abduction
		self.max_z_shift = 0.1 # plus minus 0.1 m
		self.max_incline = 15 # in deg
		self.robot_length = 0.334 # measured from stochlite
		self.robot_width = 0.192 # measured from stochlite

		# self.robot.hard_reset()
		#print('render',render, on_rack)

		self.Set_Randomization(default=True, idx1=2, idx2=2)

		self.logger = DataLog()

		if(self._is_stairs):
			boxHalfLength = 0.1
			boxHalfWidth = 1
			boxHalfHeight = 0.015
			sh_colBox = self.robot.create_collision_shape(self.robot.geom_box(),halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
			boxOrigin = 0.3
			n_steps = 15
			self.stairs = []
			for i in range(n_steps):
				step =self.robot.create_multi_body(baseMass=0,baseCollisionShapeIndex = sh_colBox,basePosition = [boxOrigin + i*2*boxHalfLength,0,boxHalfHeight + i*2*boxHalfHeight],baseOrientation=[0.0,0.0,0.0,1])
				self.stairs.append(step)
				self.robot.change_dynamics(step, -1, lateralFriction=0.8)

	def reset(self):
       
		self.robot.reset(self)
		#self._n_steps = 0

	def updateCommands(self, num_plays, episode_length):
		ratio = num_plays/episode_length
		if num_plays < 0.2 * episode_length:
			self.commands = [0, 0, 0]
		elif num_plays < 0.8 * episode_length:
		 	self.commands = np.array([self.max_linear_xvel, self.max_linear_yvel, self.max_ang_vel])*ratio
		else:
			self.commands = [self.max_linear_xvel, self.max_linear_yvel, self.max_ang_vel]
			# self.commands = np.array([self.max_linear_xvel, self.max_linear_yvel, self.max_ang_vel])*ratio

	def Set_Randomization(self, default = True, idx1 = 0, idx2=0, idx3=2, idx0=0, idx11=0, idxc=2, idxp=0, deg = 5, ori = 0): # deg = 5, changed for stochlite
		'''
		This function helps in randomizing the physical and dynamics parameters of the environment to robustify the policy.
		These parameters include wedge incline, wedge orientation, friction, mass of links, motor strength and external perturbation force.
		Note : If default argument is True, this function set above mentioned parameters in user defined manner
		'''
		if default:
			frc=[0.5,0.6,0.8]
			# extra_link_mass=[0,0.05,0.1,0.15]
			cli=[5.2,6,7,8]
			# pertub_range = [0, -30, 30, -60, 60]
			self.pertub_steps = 150 
			self.x_f = 0
			# self.y_f = pertub_range[idxp]
			self.incline_deg = deg + 2*idx1
			# self.incline_ori = ori + PI/12*idx2
			self.new_fric_val =frc[idx3]
			self.friction = self.robot.SetFootFriction(self.new_fric_val)
			# self.FrontMass = self.SetLinkMass(0,extra_link_mass[idx0])
			# self.BackMass = self.SetLinkMass(11,extra_link_mass[idx11])
			self.clips = cli[idxc]

		else:
			avail_deg = [5, 7, 9, 11] # [5, 7, 9, 11] # [5,7,9,11], changed for stochlite
			# avail_ori = [-PI/2, PI/2]
			# extra_link_mass=[0,.05,0.1,0.15]
			# pertub_range = [0, -30, 30, -60, 60]
			cli=[5,6,7,8]
			self.pertub_steps = 150 #random.randint(90,200) #Keeping fixed for now
			self.x_f = 0
			# self.y_f = pertub_range[random.randint(0,2)]
			self.incline_deg = avail_deg[random.randint(0, 2)]
			# self.incline_ori = avail_ori[random.randint(0, 1)] #(PI/12)*random.randint(0, 4) #resolution of 15 degree, changed for stochlite
			self.new_fric_val = np.round(np.clip(np.random.normal(0.6,0.08),0.55,0.8),2)
			self.friction = self.robot.SetFootFriction(self.new_fric_val)
			# i=random.randint(0,3)
			# self.FrontMass = self.SetLinkMass(0,extra_link_mass[i])
			# i=random.randint(0,3)
			# self.BackMass = self.SetLinkMass(11,extra_link_mass[i])
			self.clips = np.round(np.clip(np.random.normal(6.5,0.4),5,8),2)

	def randomize_only_inclines(self, default=True, idx1=0, idx2=0, deg = 5, ori = 0): # deg = 5, changed for stochlite
		'''
		This function only randomizes the wedge incline and orientation and is called during training without Domain Randomization
		'''
		if default:
			self.incline_deg = deg + 2 * idx1
			# self.incline_ori = ori + PI / 12 * idx2

		else:
			avail_deg = [5, 7, 9, 11] # [5, 7, 9, 11]
			# avail_ori = [-PI/2, PI/2]
			self.incline_deg = avail_deg[random.randint(0, 2)]
			# self.incline_ori = avail_ori[random.randint(0, 1)] #(PI / 12) * random.randint(0, 4)  # resolution of 15 degree


	def boundYshift(self, x, y):
		'''
		This function bounds Y shift with respect to current X shift
		Args:
			 x : absolute X-shift
			 y : Y-Shift
		Ret :
			  y : bounded Y-shift
		'''
		if x > 0.5619:
			if y > 1/(0.5619-1)*(x-1):
				y = 1/(0.5619-1)*(x-1)
		return y


	def getYXshift(self, yx):
		'''
		This function bounds X and Y shifts in a trapezoidal workspace
		'''
		y = yx[:4]
		x = yx[4:]
		for i in range(0,4):
			y[i] = self.boundYshift(abs(x[i]), y[i])
			y[i] = y[i] * 0.038
			x[i] = x[i] * 0.0418
		yx = np.concatenate([y,x])
		return yx

	def transform_action(self, action):
		'''
		Transform normalized actions to scaled offsets
		Args:
			action : 15 dimensional 1D array of predicted action values from policy in following order :
					 [(X-shifts of FL, FR, BL, BR), (Y-shifts of FL, FR, BL, BR),
					  (Z-shifts of FL, FR, BL, BR), (Augmented cmd_vel Vx, Vx, Wz)]
		Ret :
			action : scaled action parameters

		Note : The convention of Cartesian axes for leg frame in the codebase follows this order, Z points up, X forward and Y right.
		'''
		action = np.clip(action,-1,1)

		# X-Shifts scaled down by 0.1
		action[:4] = action[:4] * 0.1

		# Y-Shifts scaled down by 0.1 and offset of 0.05m abd outside added to the respective leg, in case of 0 state or 0 policy.
		action[4] = action[4] * 0.1 + 0.05  
		action[5] = action[5] * 0.1 - 0.05
		action[6] = action[6] * 0.1 + 0.05
		action[7] = action[7] * 0.1 - 0.05

		# X-Shifts scaled down by 0.1
		action[8:12] = action[8:12] * 0.1

		action[12:] = action[12:]

		# print('Scaled Action in env', action)

		return action

	
	def step(self, action):
		'''
		function to perform one step in the environment
		Args:
			action : array of action values
		Ret:
			ob 	   : observation after taking step
			reward     : reward received after taking step
			done       : whether the step terminates the env
			{}	   : any information of the env (will be added later)
		'''
		action = self.transform_action(action)
		
		self.do_simulation(action, n_frames = self._frame_skip)

		ob = self.GetObservation()
		reward, done = self._get_reward()
		return ob, reward, done,{}

	def CurrentVelocities(self):
		'''
		Returns robot's linear and angular velocities
		Ret:
			radial_v  : linear velocity
			current_w : angular velocity
		'''
		current_w = self.robot.GetBaseAngularVelocity()[2]
		current_v = self.robot.GetBaseLinearVelocity()
		radial_v = math.sqrt(current_v[0]**2 + current_v[1]**2)
		return radial_v, current_w


	def do_simulation(self, action, n_frames):
		'''
		Converts action parameters to corresponding motor commands with the help of a elliptical trajectory controller
		'''  
		self.action = action
		prev_motor_angles = self.robot.GetMotorAngles()
		ii = 0

		leg_m_angle_cmd = self._trajgen.generate_trajectory(action, prev_motor_angles, self.dt)
		
		m_angle_cmd_ext = np.array(leg_m_angle_cmd)

		m_vel_cmd_ext = np.zeros(12)

		force_visualizing_counter = 0

		for _ in range(n_frames):
			ii = ii + 1
			applied_motor_torque = self.robot._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
			self.robot.step_simulation()

			if self._n_steps >=self.pertub_steps and self._n_steps <= self.pertub_steps + self.stride:
				force_visualizing_counter += 1
				if(force_visualizing_counter%7==0):
					self.robot.apply_Ext_Force(self.x_f,self.y_f,visulaize=True,life_time=0.1)
				else:
					self.robot.apply_Ext_Force(self.x_f,self.y_f,visulaize=False)

	
		contact_info = self.robot.get_foot_contacts()
		pos, ori = self.robot.GetBasePosAndOrientation()
		#print('ori',ori)
		# Camera follows robot in the debug visualizer
		self.robot.reset_debug_visualizer_camera(self._cam_dist, 10, -10, pos)

		Rot_Mat = self.robot.get_matrix_from_quaternion(ori)
		# print('Rot_Mat is',Rot_Mat)
		Rot_Mat = np.array(Rot_Mat)
		#print('Rot_Mat',Rot_Mat)
		Rot_Mat = np.reshape(Rot_Mat,(3,3))

		plane_normal, self.support_plane_estimated_roll, self.support_plane_estimated_pitch = normal_estimator.vector_method_Stochlite(self.prev_incline_vec, contact_info, self.robot.GetMotorAngles(), Rot_Mat)
		self.prev_incline_vec = plane_normal

		motor_torque = self.robot._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
		motor_vel = self.robot.GetMotorVelocities()
		self.total_power = self.power_consumed(motor_torque, motor_vel)
		# print('power per step', self.total_power)

		# Data Logging
		# log_dir = os.getcwd()
		# self.logger.log_kv("Robot_roll", pos[0])
		# self.logger.log_kv("Robot_pitch", pos[1])
		# self.logger.log_kv("SP_roll", self.support_plane_estimated_roll)
		# self.logger.log_kv("SP_pitch", self.support_plane_estimated_pitch)
		# self.logger.save_log(log_dir + '/experiments/logs_sensors')

		# print("estimate", self.support_plane_estimated_roll, self.support_plane_estimated_pitch)
		# print("incline", self.incline_deg)

		self._n_steps += 1



	def _termination(self, pos, orientation):
		'''
		Check termination conditions of the environment
		Args:
			pos 		: current position of the robot's base in world frame
			orientation : current orientation of robot's base (Quaternions) in world frame
		Ret:
			done 		: return True if termination conditions satisfied
		'''
		done = False
		RPY = self.robot.get_euler_from_quaternion(orientation)

		if self._n_steps >= self.termination_steps:
			done = True
		else:
			if abs(RPY[0]) > math.radians(30):
				print('Oops, Robot about to fall sideways! Terminated', RPY[0])
				done = True

			if abs(RPY[1])>math.radians(35):
				print('Oops, Robot doing wheely! Terminated', RPY[1])
				done = True

			if pos[2] > 0.7:
				print('Robot was too high! Terminated', pos[2])
				done = True

		return done

	def _get_reward(self):
		'''
		Calculates reward achieved by the robot for RPY stability, torso height criterion and forward distance moved on the slope:
		Ret:
			reward : reward achieved
			done   : return True if environment terminates
			
		'''
		wedge_angle = self.incline_deg*PI/180
		robot_height_from_support_plane = 0.25 # walking height of stochlite	
		pos, ori = self.robot.GetBasePosAndOrientation()

		RPY_orig = self.robot.get_euler_from_quaternion(ori)
		RPY = np.round(RPY_orig, 4)

		current_height = round(pos[2], 5)
		self.current_com_height = current_height
		# standing_penalty=3
	
		desired_height = (robot_height_from_support_plane)/math.cos(wedge_angle) + math.tan(wedge_angle)*((pos[0])*math.cos(self.incline_ori)+ 0.5)

		#Need to re-evaluate reward functions for slopes, value of reward function after considering error should be greater than 0.5, need to tune

		roll_reward = np.exp(-45 * ((RPY[0]-self.support_plane_estimated_roll) ** 2))
		pitch_reward = np.exp(-45 * ((RPY[1]-self.support_plane_estimated_pitch) ** 2))
		# yaw_reward = np.exp(-40 * (RPY[2] ** 2)) # np.exp(-35 * (RPY[2] ** 2)) increasing reward for yaw correction
		# height_reward = 20 * np.exp(-800 * (desired_height - current_height) ** 2)
		height_reward = 20 * np.exp(-100 * abs(desired_height - current_height))
		power_reward = np.exp(-self.total_power)

		x = pos[0]
		y = pos[1]
		# yaw = RPY[2]
		x_l = self._last_base_position[0]
		y_l = self._last_base_position[1]
		self._last_base_position = pos
		# self.last_yaw = RPY[2]

		step_distance_x = abs(x - x_l)
		step_distance_y = abs(y - y_l)
		# step_x_reward = -100 * step_distance_x
		# step_y_reward = -100 * step_distance_y

		x_velocity = self.robot.GetBaseLinearVelocity()[0] #(x - x_l)/self.dt
		y_velocity = self.robot.GetBaseLinearVelocity()[1] #(y - y_l)/self.dt
		ang_velocity = self.robot.GetBaseAngularVelocity()[2] #(yaw - self.last_yaw)/self.dt
		roll_vel = self.robot.GetBaseAngularVelocity()[0]
		pitch_vel = self.robot.GetBaseAngularVelocity()[1]

		cmd_translation_vel_reward = 15 * np.exp(-5 * ((x_velocity - self.max_linear_xvel*self.commands[0])**2 + (y_velocity - self.max_linear_yvel*self.commands[1])**2))
		cmd_rotation_vel_reward = 10 * np.exp(-1 * (ang_velocity - self.max_ang_vel*self.commands[2])**2)
		roll_vel_reward = 2 * np.exp(-100 * (roll_vel ** 2))
		pitch_vel_reward = 2 * np.exp(-100 * (pitch_vel ** 2))

		done = self._termination(pos, ori)
		if done:
			reward = 0
		else:
			reward = round(pitch_reward, 4) + round(roll_reward, 4) + round(height_reward, 4)+ round(cmd_translation_vel_reward, 4) + round(cmd_rotation_vel_reward, 4)\
					 + round(roll_vel_reward, 4) + round(pitch_vel_reward, 4)# - 10000 * round(step_distance_x, 4) - 10000 * round(step_distance_y, 4)\
					#+ round(power_reward, 4) + round(yaw_reward, 4) - 100 * round(step_distance_x, 4) - 100 * round(step_distance_y, 4)\

		# reward_info = [0, 0, 0, 0, 0]
		# reward_info[0] = self.commands[0]
		# reward_info[1] = self.commands[1]
		# reward_info[2] = x_velocity
		# reward_info[3] = y_velocity
		# reward_info[4] = reward_info[4] + reward
		'''
		#Penalize for standing at same position for continuous 150 steps
		self.step_disp.append(step_distance_x)
	
		if(self._n_steps>150):
			if(sum(self.step_disp)<0.035):
				reward = reward-standing_penalty
		'''

		# Testing reward function
		reward_info = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		done = self._termination(pos, ori)
		if done:
			reward = 0
		else:
			reward_info[0] = round(roll_reward, 4)
			reward_info[1] = round(pitch_reward, 4)
			reward_info[2] = round(height_reward, 4)
			reward_info[3] = round(roll_vel_reward, 4)
			reward_info[4] = round(pitch_vel_reward, 4)
			reward_info[5] = round(cmd_translation_vel_reward, 4)
			reward_info[6] = round(cmd_rotation_vel_reward, 4)
			reward_info[7] = -10000 * round(step_distance_x, 4)
			reward_info[8] = -10000 * round(step_distance_y, 4)
			reward_info[9] = round(power_reward, 4)
			reward_info[10] = self.support_plane_estimated_roll
			reward_info[11] = self.support_plane_estimated_pitch

			reward = round(pitch_reward, 4) + round(roll_reward, 4) + round(height_reward, 4)+ round(cmd_translation_vel_reward, 4) + round(cmd_rotation_vel_reward, 4)\
					 + round(roll_vel_reward, 4) + round(pitch_vel_reward, 4)# - 10000 * round(step_distance_x, 4) - 10000 * round(step_distance_y, 4)\
					#+ round(power_reward, 4) + round(yaw_reward, 4) - 100 * round(step_distance_x, 4) - 100 * round(step_distance_y, 4)\

		reward_info[12] = reward

		return reward_info, done

		# return reward, done

	def add_noise(self, sensor_value, SD = 0.04):
		'''
		Adds sensor noise of user defined standard deviation in current sensor_value
		'''
		noise = np.random.normal(0, SD, 1)
		sensor_value = sensor_value + noise[0]
		return sensor_value

	def power_consumed(self, motor_torque, motor_vel):
		'''
		Calculates total power consumed (sum of all motor powers) as a multiplication of torque and velocity
		'''
		total_power = 0
		for torque, vel in zip(motor_torque, motor_vel):
			power = torque * vel
			total_power += abs(power)
		return total_power

	def GetObservation(self):
		'''
		This function returns the current observation of the environment for the interested task
		Note:- obs and state vectors are different. obs vector + cmd vels = state vector.
		Ret:
			obs : [roll, pitch, roll_vel, pitch_vel, yaw_vel, SP roll, SP pitch]
		'''
		pos, ori = self.robot.GetBasePosAndOrientation()
		RPY_vel = self.robot.GetBaseAngularVelocity()
		RPY = self.robot.get_euler_from_quaternion(ori)
		RPY = np.round(RPY, 5)
		RPY_noise = []

		for val in RPY:
			if(self.add_IMU_noise):
				val = self.add_noise(val)
			RPY_noise.append(val)

		obs = [RPY[0], RPY[1], RPY_vel[0], RPY_vel[1], RPY_vel[2], self.support_plane_estimated_roll, self.support_plane_estimated_pitch]
		self.last_rpy = RPY

		return obs
	
	def render(self):
        
		self.robot.render()


