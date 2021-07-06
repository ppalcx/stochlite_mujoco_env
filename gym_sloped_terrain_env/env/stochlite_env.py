"""
Python programme to create a framework of the Stoch2 gym environment
"""
import os
import numpy as np
import gym
import math
import random
import matplotlib.pyplot as plt
import time

from collections import deque
from gym import utils, spaces
from utils.logger import DataLog

import gym_sloped_terrain.envs.bullet_client as bullet_client
import gym_sloped_terrain.envs.walking_controller as walking_controller
import gym_sloped_terrain.envs.planeEstimation.get_terrain_normal as normal_estimator
import gym_sloped_terrain.envs.rotation as rotation

 
"""Uncomment one of the lines below to access the desired physics simulator"""
import envs.mujoco_func_env as Simulator
# import envs.pybullet_func_env as Simulator


###############################################################
#	CLASS StochliteEnv DEFINED HERE
###############################################################

class StochliteEnv(gym.Env):

	def __init__(self, render=False, on_rack=False):
		"""
		Initialize the class variables, set the directory path, and load the initial trajectiories.
		Define the action-space and the state-space, along with its dimensions.
		"""
		self._is_render = render
		self._on_rack = on_rack

		self.robot = Simulator.StochliteLoader(frame_skip = 5)

		self._walkcon = walking_controller.WalkingController(gait_type='trot',
															spine_enable = False,
															planning_space = 'polar_task_space',
															left_to_right_switch = True,
															frequency=self.robot.frequency,
															zero_desired_velocity = True)

		##Gym env related mandatory variables
		observation_high = np.array([10.0] * self.robot.obs_dim)
		observation_low = -observation_high
		self.observation_space = spaces.Box(observation_low, observation_high)

		action_high = np.array([1] * self.robot.action_dim)
		self.action_space = spaces.Box(-action_high, action_high)

		self.robot.hard_reset(render, on_rack)


	def Set_Randomization(self, default = True, idx1 = 0, idx2=0, idx3=2, idx0=0, idx11=0, idxc=2, idxp=0, deg = 0, ori = 0): # deg = 5, changed for stochlite
		'''
		This function helps in randomizing the physical and dynamics parameters of the environment to robustify the policy.
		These parameters include wedge incline, wedge orientation, friction, mass of links, motor strength and external perturbation force.
		Note : If default argument is True, this function set above mentioned parameters in user defined manner
		'''
		if default:
			frc=[0.55,0.6,0.8]
			# extra_link_mass=[0,0.05,0.1,0.15]
			cli=[5.2,6,7,8]
			# pertub_range = [0, -30, 30, -60, 60]
			self.pertub_steps = 150 
			self.x_f = 0
			# self.y_f = pertub_range[idxp]
			self.incline_deg = deg + 2*idx1
			# self.incline_ori = ori + PI/12*idx2
			self.new_fric_val =frc[idx3]
			self.friction = self.SetFootFriction(self.new_fric_val)
			# self.FrontMass = self.SetLinkMass(0,extra_link_mass[idx0])
			# self.BackMass = self.SetLinkMass(11,extra_link_mass[idx11])
			self.clips = cli[idxc]

		else:
			avail_deg = [0, 0, 0, 0] # [5, 7, 9, 11] # [5,7,9,11], changed for stochlite
			# extra_link_mass=[0,.05,0.1,0.15]
			# pertub_range = [0, -30, 30, -60, 60]
			cli=[5,6,7,8]
			self.pertub_steps = 150 #random.randint(90,200) #Keeping fixed for now
			self.x_f = 0
			# self.y_f = pertub_range[random.randint(0,2)]
			self.incline_deg = avail_deg[random.randint(0, 3)]
			# self.incline_ori = (PI/12)*random.randint(0, 4) #resolution of 15 degree, changed for stochlite
			# self.new_fric_val = np.round(np.clip(np.random.normal(0.6,0.08),0.55,0.8),2)
			self.friction = self.SetFootFriction(0.8) #(self.new_fric_val)
			# i=random.randint(0,3)
			# self.FrontMass = self.SetLinkMass(0,extra_link_mass[i])
			# i=random.randint(0,3)
			# self.BackMass = self.SetLinkMass(11,extra_link_mass[i])
			self.clips = np.round(np.clip(np.random.normal(6.5,0.4),5,8),2)

	def randomize_only_inclines(self, default=True, idx1=0, idx2=0, deg = 0, ori = 0): # deg = 5, changed for stochlite
		'''
		This function only randomizes the wedge incline and orientation and is called during training without Domain Randomization
		'''
		if default:
			self.incline_deg = deg + 2 * idx1
			# self.incline_ori = ori + PI / 12 * idx2

		else:
			avail_deg = [0, 0, 0, 0] # [5, 7, 9, 11]
			self.incline_deg = avail_deg[random.randint(0, 3)]
			# self.incline_ori = (PI / 12) * random.randint(0, 4)  # resolution of 15 degree


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
			action : 20 dimensional 1D array of predicted action values from policy in following order :
					 [(step lengths of FR, FL, BR, BL), (steer angles of FR, FL, BR, BL),
					  (Y-shifts of FR, FL, BR, BL), (X-shifts of FR, FL, BR, BL),
					  (Z-shifts of FR, FL, BR, BL)]
		Ret :
			action : scaled action parameters
		Note : The convention of Cartesian axes for leg frame in the codebase follow this order, Y points up, X forward and Z right. 
		       While in research paper we follow this order, Z points up, X forward and Y right.
		'''
		'''
		The previous action transform
		action = np.clip(action, -1, 1)
		action[:4] = (action[:4] + 1)/2					# Step lengths are positive always
		action[:4] = action[:4] *2 * 0.068  				# Max step length = 2x0.068
		action[4:8] = action[4:8] * PI/2				# PHI can be [-pi/2, pi/2]
		action[8:12] = (action[8:12]+1)/2				# Y-shifts are positive always
		action[8:16] = self.getYXshift(action[8:16])
		action[16:20] = action[16:20]*0.035				# Max allowed Z-shift due to abduction limits is 3.5cm
		action[17] = -action[17]
		action[19] = -action[19]
		'''

		# Note: just check if the getYXShift function is required  

		#This is to switch the z_shift and y_shift
		#this is being done as we are taking the 
		# cordinate system for the walking controler 
		# where all the cordinate system are using right 
		# hand rule  
		# action_temp = action[8:12]

		# replasing the y_shift with z_shift
		# The order is fr,fl,br,bl 
		# Ensure that in the walking controller the order 
		# of the legs are taken care of 
		# we are negating the values as the convention for 
		# the walking controller is with right hand rule 
		# forwards: +x, upwards +z, left: +y
		# action[8]  = -action[16]
		# action[9]  = action[17]
		# action[10] = -action[18]
		# action[11] = action[19]

		# #replacing the z_shift with the y_shift 
		# action[16] = action_temp[0]
		# action[17] = action_temp[1]
		# action[18] = action_temp[2]
		# action[19] = action_temp[3]

		'''
		Action changed
        action[:4] -> step_length fl fr bl br
        action[4:8] -> steer angle 
        action[8:12] -> x_shift fl fr bl br
        action[12:16] -> y_shift fl fr bl br
        action[16:20] -> z_shift fl fr bl br
		'''

		# X, Y, Z shifts required for ang vel calculations
		# action[8:12] = action[8:12]
		# action[12:16] = action[12:16] * self.max_y_shift
		# action[16:20] = action[16:20] * self.max_z_shift

		# step_length_offset = self.max_steplength * math.sqrt(self.commands[0]**2 + self.commands[1]**2)/math.sqrt(self.max_linear_xvel**2 + self.max_linear_yvel**2)

		lvel_step_length_offset = math.sqrt(self.commands[0]**2 + self.commands[1]**2) / self._frequency # math.sqrt(self.max_linear_xvel**2 + self.max_linear_yvel**2) * math.sqrt(self.commands[0]**2 + self.commands[1]**2) / self._frequency
		lvel_steer_angle_offset = math.atan2(self.commands[1], self.commands[0]) # self.max_steer_angle * self.commands[1]

		# Robot footprint
		# robot_fp_len = self.robot_length
		# robot_fp_wid = self.robot_width +  

		# sp_pitch_offset = self.max_x_shift * np.degrees(self.support_plane_estimated_pitch) / self.max_incline
		# sp_roll_offset = self.max_z_shift * np.degrees(self.support_plane_estimated_roll) / self.max_incline
		# print("Sp pitch", np.degrees(self.support_plane_estimated_pitch))
		# print("pitch offset", sp_pitch_offset)

		action = np.clip(action,-1,1)

		action[:4] = action[:4] * self.max_steplength # Limiting step length # * 0.35 # (action[:4]+1)/2 * 0.4 * self.max_steplength considering neg step lengths #+ 0.15 # Step lengths are positive always
		
		action[:4] = action[:4] + lvel_step_length_offset # Max step length = 2x0.1 =0.2

		action[4:8] = action[4:8] * self.max_steer_angle # Limiting steer angle

		action[4:8] = action[4:8] + lvel_steer_angle_offset 

		action[8:12] = action[8:12] #+ sp_pitch_offset #- 0.08

		action[12:16] = action[12:16] * self.max_y_shift #+ 0.04 # np.clip(action[12:16], -0.14, 0.14)  #the Y_shift can be +/- 0.14 from the leg zero position, max abd angle = +/- 30 deg 

		# action[12] = -action[12]
		# action[14] = -action[14]

		action[16:20] = action[16:20] * self.max_z_shift

		# action[16] = action[16] * self.max_z_shift + sp_roll_offset
		# action[17] = action[17] * self.max_z_shift - sp_roll_offset # Z_shift can be +/- 0.1 from z center
		# action[18] = action[18] * self.max_z_shift + sp_roll_offset
		# action[19] = action[19] * self.max_z_shift - sp_roll_offset

		# print("in env", action)

		return action

	def CurrentVelocities(self):
		'''
		Returns robot's linear and angular velocities
		Ret:
			radial_v  : linear velocity
			current_w : angular velocity
		'''
		current_w = self.GetBaseAngularVelocity()[2]
		current_v = self.GetBaseLinearVelocity()
		radial_v = math.sqrt(current_v[0]**2 + current_v[1]**2)
		return radial_v, current_w		

	def step(self, action, callback=None):
		"""
		The function call to get the next action from the current state
        Args:
        --- action : An array present in the action space of the environment
        --- callback : A function sent as argument to initiate callbacks
        Returns:
        --- observation : The observation array on simulating the action for a fixed time-step
        --- reward : The reward value for the given action
        --- done : A flag indicating whether the episode has terminated or not
        --- info : A dictionary containing the rewards and the penalty incurred
		"""
		energy_spent_per_step, cost_reference = self.do_simulation(action, frame_skip = self.robot.frame_skip, callback=callback)
		
		position, orientation = self.robot.GetBasePositionAndOrientation()
		observation = self.GetObservation()
		done, penalty = self.EpisodeTermination(position, orientation)
		reward = self.GetReward(action, energy_spent_per_step, cost_reference, penalty)

		if done:
			self.reset()

		return observation, reward, done, dict(reward_run=reward, reward_ctrl=-penalty)



	def GetObservation(self):
		"""
		Concatenate the position and orientation (Observation_space dim = 7)
		Returns:
        --- observation : An array present in the observation space of the environment
		"""
		pos, ori = self.robot.GetBasePositionAndOrientation()
		return np.concatenate([pos,ori]).ravel() 
		

	def EpisodeTermination(self, position, orientation):
		"""
		Function to terminate the current episode given the conditions
		Args:
        --- position : The position of the robot in the Cartesian coordinates
        --- orientation : A quaternion storing the orientation of the robot
        Returns:
        --- done : A flag indicating whether the episode has terminated or not
		--- penalty : The penalty incurred during episode termination
		"""
		done = False
		penalty = 0
		rot_mat = self.robot.GetMatrixFromQuaternion(orientation)
		local_up = rot_mat[6:]

		# stop episode after ten steps
		if self.robot.n_steps >= 1000:
			done = True
			print('%s steps finished. Terminated' % self.robot.n_steps)
			penalty = 0
		else:
			if np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.3:
				print('Oops, Robot about to fall! Terminated')
				done = True
				penalty = penalty + 0.1
			if position[2] < 0.04:
				print('Robot was too low! Terminated')
				done = True
				penalty = penalty + 0.5
			if position[2] > 0.3:
				print('Robot was too high! Terminated')
				done = True
				penalty = penalty + 0.6

		if done and self.robot.n_steps <= 2:
			penalty = 3

		return done, penalty


	def GetReward(self, action, energy_spent_per_step, cost_reference, penalty):
		
		'''
		Calculates reward achieved by the robot for RPY stability, torso height criterion and forward distance moved on the slope:
		Ret:
			reward : reward achieved
			done   : return True if environment terminates
			
		'''
		wedge_angle = self.incline_deg*PI/180
		robot_height_from_support_plane = 0.25 # walking height of stochlite	
		pos, ori = self.GetBasePosAndOrientation()

		RPY_orig = rotation.quat2euler(ori)
		RPY = np.round(RPY_orig, 4)

		current_height = round(pos[2], 5)
		self.current_com_height = current_height
		# standing_penalty=3
	
		desired_height = (robot_height_from_support_plane)/math.cos(wedge_angle) + math.tan(wedge_angle)*((pos[0])*math.cos(self.incline_ori)+ 0.5)

		#Need to re-evaluate reward functions for slopes, value of reward function after considering error should be greater than 0.5, need to tune

		roll_reward = np.exp(-45 * ((RPY[0]-self.support_plane_estimated_roll) ** 2))
		pitch_reward = np.exp(-45 * ((RPY[1]-self.support_plane_estimated_pitch) ** 2))
		yaw_reward = np.exp(-40 * (RPY[2] ** 2)) # np.exp(-35 * (RPY[2] ** 2)) increasing reward for yaw correction
		height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

		# x = pos[0]
		# y = pos[1]
		# yaw = RPY[2]
		# x_l = self._last_base_position[0]
		# y_l = self._last_base_position[1]
		self._last_base_position = pos
		self.last_yaw = RPY[2]

		# step_distance_x = (x - x_l)
		# step_distance_y = abs(y - y_l)

		x_velocity = self.GetBaseLinearVelocity()[0] #(x - x_l)/self.dt
		y_velocity = self.GetBaseLinearVelocity()[1] #(y - y_l)/self.dt
		ang_velocity = self.GetBaseAngularVelocity()[2] #(yaw - self.last_yaw)/self.dt

		cmd_translation_vel_reward = np.exp(-50 * ((x_velocity - self.commands[0])**2 + (y_velocity - self.commands[1])**2))
		cmd_rotation_vel_reward = np.exp(-50 * (ang_velocity - self.commands[2])**2)

		done = self._termination(pos, ori)
		if done:
			reward = 0
		else:
			reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4)\
					 + round(height_reward,4) + round(cmd_translation_vel_reward, 4) + round(cmd_rotation_vel_reward, 4)#- 100 * round(step_distance_x, 4) - 100 * round(step_distance_y, 4)\

		reward_info = [0, 0, 0, 0, 0]
		reward_info[0] = self.commands[0]
		reward_info[1] = self.commands[1]
		reward_info[2] = x_velocity
		reward_info[3] = y_velocity
		reward_info[4] = reward_info[4] + reward
		'''
		#Penalize for standing at same position for continuous 150 steps
		self.step_disp.append(step_distance_x)
	
		if(self._n_steps>150):
			if(sum(self.step_disp)<0.035):
				reward = reward-standing_penalty
		'''
		# Testing reward function

		# reward_info = [0, 0, 0, 0, 0, 0, 0, 0, 0]
		# done = self._termination(pos, ori)
		# if done:
		# 	reward = 0
		# else:
		# 	reward_info[0] = round(roll_reward, 4)
		# 	reward_info[1] = round(pitch_reward, 4)
		# 	reward_info[2] = round(yaw_reward, 4)
		# 	reward_info[3] = round(height_reward, 4)
		# 	reward_info[4] = 100 * round(step_distance_x, 4)
		# 	reward_info[5] = -50 * round(step_distance_y, 4)
		# 	reward_info[6] = self.support_plane_estimated_roll
		# 	reward_info[7] = self.support_plane_estimated_pitch

		# 	reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4)\
		# 			 + round(height_reward,4) + 100 * round(step_distance_x, 4) - 50 * round(step_distance_y, 4)

		# reward_info[8] = reward

		return reward_info, done
		
	'''
		
		"""
		Function to call the rewards
		Args:
        --- action : An array present in the action space of the environment
        --- energy_spent_per_step : Total energy spent in the n frames of simulation
		--- cost_reference : Reference cost for the gait
		--- penalty : The penalty incurred during episode termination
        Returns:
        --- reward : The reward value for the given action
		"""
		current_base_position, current_base_orientation = self.robot.GetBasePositionAndOrientation() 
		distance_travelled = current_base_position[0] - self.robot.last_base_position[0]
		self.robot.xpos_previous = current_base_position
		
		costreference_reward = np.exp(-2*(0 - cost_reference)**2)
		reward = distance_travelled - penalty - 0.01 * energy_spent_per_step + 0.5 * cost_reference
		return reward
	'''



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
		RPY = self._pybullet_client.getEulerFromQuaternion(orientation)

		if self._n_steps >= self.termination_steps:
			done = True
		else:
			if abs(RPY[0]) > math.radians(30):
				print('Oops, Robot about to fall sideways! Terminated')
				done = True

			if abs(RPY[1])>math.radians(35):
				print('Oops, Robot doing wheely! Terminated')
				done = True

			if pos[2] > 0.7:
				print('Robot was too high! Terminated')
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
		pos, ori = self.GetBasePosAndOrientation()

		RPY_orig = rotation.quat2euler(ori)
		RPY = np.round(RPY_orig, 4)

		current_height = round(pos[2], 5)
		self.current_com_height = current_height
		# standing_penalty=3
	
		desired_height = (robot_height_from_support_plane)/math.cos(wedge_angle) + math.tan(wedge_angle)*((pos[0])*math.cos(self.incline_ori)+ 0.5)

		#Need to re-evaluate reward functions for slopes, value of reward function after considering error should be greater than 0.5, need to tune

		roll_reward = np.exp(-45 * ((RPY[0]-self.support_plane_estimated_roll) ** 2))
		pitch_reward = np.exp(-45 * ((RPY[1]-self.support_plane_estimated_pitch) ** 2))
		yaw_reward = np.exp(-40 * (RPY[2] ** 2)) # np.exp(-35 * (RPY[2] ** 2)) increasing reward for yaw correction
		height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

		# x = pos[0]
		# y = pos[1]
		# yaw = RPY[2]
		# x_l = self._last_base_position[0]
		# y_l = self._last_base_position[1]
		self._last_base_position = pos
		self.last_yaw = RPY[2]

		# step_distance_x = (x - x_l)
		# step_distance_y = abs(y - y_l)

		x_velocity = self.GetBaseLinearVelocity()[0] #(x - x_l)/self.dt
		y_velocity = self.GetBaseLinearVelocity()[1] #(y - y_l)/self.dt
		ang_velocity = self.GetBaseAngularVelocity()[2] #(yaw - self.last_yaw)/self.dt

		cmd_translation_vel_reward = np.exp(-50 * ((x_velocity - self.commands[0])**2 + (y_velocity - self.commands[1])**2))
		cmd_rotation_vel_reward = np.exp(-50 * (ang_velocity - self.commands[2])**2)

		done = self._termination(pos, ori)
		if done:
			reward = 0
		else:
			reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4)\
					 + round(height_reward,4) + round(cmd_translation_vel_reward, 4) + round(cmd_rotation_vel_reward, 4)#- 100 * round(step_distance_x, 4) - 100 * round(step_distance_y, 4)\

		reward_info = [0, 0, 0, 0, 0]
		reward_info[0] = self.commands[0]
		reward_info[1] = self.commands[1]
		reward_info[2] = x_velocity
		reward_info[3] = y_velocity
		reward_info[4] = reward_info[4] + reward
		'''
		#Penalize for standing at same position for continuous 150 steps
		self.step_disp.append(step_distance_x)
	
		if(self._n_steps>150):
			if(sum(self.step_disp)<0.035):
				reward = reward-standing_penalty
		'''
		# Testing reward function

		# reward_info = [0, 0, 0, 0, 0, 0, 0, 0, 0]
		# done = self._termination(pos, ori)
		# if done:
		# 	reward = 0
		# else:
		# 	reward_info[0] = round(roll_reward, 4)
		# 	reward_info[1] = round(pitch_reward, 4)
		# 	reward_info[2] = round(yaw_reward, 4)
		# 	reward_info[3] = round(height_reward, 4)
		# 	reward_info[4] = 100 * round(step_distance_x, 4)
		# 	reward_info[5] = -50 * round(step_distance_y, 4)
		# 	reward_info[6] = self.support_plane_estimated_roll
		# 	reward_info[7] = self.support_plane_estimated_pitch

		# 	reward = round(yaw_reward, 4) + round(pitch_reward, 4) + round(roll_reward, 4)\
		# 			 + round(height_reward,4) + 100 * round(step_distance_x, 4) - 50 * round(step_distance_y, 4)

		# reward_info[8] = reward

		return reward_info, done
	
	
	def add_noise(self, sensor_value, SD = 0.04):
		'''
		Adds sensor noise of user defined standard deviation in current sensor_value
		'''
		noise = np.random.normal(0, SD, 1)
		sensor_value = sensor_value + noise[0]
		return sensor_value

	def render(self):
		"""
		Function to render the environment simulations
		"""
		self.robot.RenderModel()


	def reset(self):
		"""
		Set the initial conditions of the environemnt, return the observation
		"""
		self.robot.ResetTheEnv()
		self.robot.n_steps = 0
			  
		return self.GetObservation()


	def CostReferenceGait(self, theta, q):
		"""
		Performs forward kinematics, calculates the error involved
		Args:
		--- theta : The theta which was initialized
		--- q : The Motor angles
		Returns:
		--- ls_error : The cost reference values
		"""
		ls_error = np.linalg.norm(self.action - self.robot._action_ref)
		return ls_error

	def GetDesiredMotorAngles(self):
		"""
		Function to get the desired motor angles from the walking controller
		"""
		_, leg_m_angle_cmd, _, _ = self._walkcon.transform_action_to_motor_joint_command(self.robot.theta,self.action)
		return leg_m_angle_cmd
