import os
import numpy as np
import gym
from gym import spaces
import math
import random
from collections import deque
import matplotlib.pyplot as plt
from utils.logger import DataLog
from mujoco_py.generated import const
import time


import gym_sloped_terrain.envs.bullet_client as bullet_client
import gym_sloped_terrain.envs.walking_controller as walking_controller
import gym_sloped_terrain.envs.planeEstimation.get_terrain_normal as normal_estimator



# LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
# KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076] #hip
# KNEE_CONSTRAINT_POINT_LEFT = [0.0,0.0,-0.077] #knee

RENDER_HEIGHT = 720 
RENDER_WIDTH = 960 
PI = np.pi
no_of_points = 100


def constrain_theta(theta):
	theta = np.fmod(theta, 2*no_of_points)
	if(theta < 0):
		theta = theta + 2*no_of_points
	return theta

class StochliteEnv(mujoco_env.MujocoEnv,utils.EzPickle):

	def __init__(self,
				 render = False,
				 on_rack = False,
				 gait = 'trot',
				 phase =   [0, no_of_points, no_of_points,0],#[FR, FL, BR, BL] 
				 action_dim = 20,
				 end_steps = 1000,
				 stairs = False,
				 downhill =False,
				 seed_value = 100,
				 wedge = False,
				 IMU_Noise = False,
				 deg = 5): # deg = 5

		self._is_stairs = stairs
		self._is_wedge = wedge
		self._is_render = render
		self._on_rack = on_rack
		self.rh_along_normal = 0.24

		self.seed_value = seed_value
		random.seed(self.seed_value)

		if self._is_render:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
		else:
			self._pybullet_client = bullet_client.BulletClient()

		self._theta = 0

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
		self.last_yaw = 0
		self._distance_limit = float("inf")

		self.current_com_height = 0.25 # 0.243
		
		#wedge_parameters
		self.wedge_start = 0.5 
		self.wedge_halflength = 2

		if gait is 'trot':
			phase = [0, no_of_points, no_of_points, 0]
		elif gait is 'walk':
			phase = [0, no_of_points, 3*no_of_points/2 ,no_of_points/2]
		self._walkcon = walking_controller.WalkingController(gait_type=gait, phase=phase)
		self.inverse = False
		self._cam_dist = 1.0
		self._cam_yaw = 0.0
		self._cam_pitch = 0.0

		self.avg_vel_per_step = 0
		self.avg_omega_per_step = 0

		self.linearV = 0
		self.angV = 0
		self.prev_vel=[0,0,0]

		self.x_f = 0
		self.y_f = 0

		self.clips=7

		self.friction = 0.6
		self.ori_history_length = 3
		self.ori_history_queue = deque([0]*3*self.ori_history_length, 
		                            maxlen=3*self.ori_history_length)#observation queue

		self.step_disp = deque([0]*100, maxlen=100)
		self.stride = 5

		self.incline_deg = deg
		self.incline_ori = 0

		self.prev_incline_vec = (0,0,1)

		self.terrain_pitch = []
		self.add_IMU_noise = IMU_Noise

		self.support_plane_estimated_pitch = 0
		self.support_plane_estimated_roll = 0

		self.pertub_steps = 0
		self.x_f = 0
		self.y_f = 0

		

		self.state_pos = StatePositionIndex()
		self.reset_pos = InitPosition()
		self.state_vel =StateVelocityIndex()
		self.actuaor = ActuatorIndex()
		self.limits = JoinLimit()

		mujoco_env.MujocoEnv.__ini__(self, stoch23.xml, 1)
		utils.EzPickle.__init__(self)







		## Gym env related mandatory variables
		self._obs_dim = 3*self.ori_history_length + 2 #[r,p,y]x previous time steps, suport plane roll and pitch
		observation_high = np.array([np.pi/2] * self._obs_dim)
		observation_low = -observation_high
		self.observation_space = spaces.Box(observation_low, observation_high)

		action_high = np.array([1] * self._action_dim)
		self.action_space = spaces.Box(-action_high, action_high)

		self.commands = np.array([0, 0, 0]) #Joystick commands consisting of cmd_x_velocity, cmd_y_velocity, cmd_ang_velocity
		self.max_linear_xvel = 0.35 # calculation is < 0.2 m steplength times the frequency 2.5 Hz
		self.max_linear_yvel = 0 #0.25, made zero for only x direction # calculation is < 0.14 m times the frequency 2.5 Hz
		self.max_ang_vel = 0 #6, made zero for only x direction # less than one complete rotation in one second
		self.max_steplength = 0.2 # by the kinematic limits of the robot
		self.max_x_shift = 0.1 #plus minus 0.1 m
		self.max_y_shift = 0.14 # max 30 degree abduction
		self.max_z_shift = 0.1 # plus minus 0.1 m
		self.max_incline = 15 # in deg

		self.hard_reset()

		self.Set_Randomization(default=True, idx1=2, idx2=2)

		self.logger = DataLog()

		if(self._is_stairs):
			boxHalfLength = 0.1
			boxHalfWidth = 1
			boxHalfHeight = 0.015
			sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
			boxOrigin = 0.3
			n_steps = 15
			self.stairs = []
			for i in range(n_steps):
				step =self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,basePosition = [boxOrigin + i*2*boxHalfLength,0,boxHalfHeight + i*2*boxHalfHeight],baseOrientation=[0.0,0.0,0.0,1])
				self.stairs.append(step)
				self._pybullet_client.changeDynamics(step, -1, lateralFriction=0.8)


	def hard_reset(self):
		'''
		Function to
		1) Set simulation parameters which remains constant throughout the experiments
		2) load urdf of plane, wedge and robot in initial conditions
		'''
		self._pybullet_client.resetSimulation()
		self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
		self._pybullet_client.setTimeStep(self.dt/self._frame_skip)

		self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
		self._pybullet_client.changeVisualShape(self.plane,-1,rgbaColor=[1,1,1,0.9])
		self._pybullet_client.setGravity(0, 0, -9.8)

		if self._is_wedge:

			wedge_halfheight_offset = 0.01

			self.wedge_halfheight = wedge_halfheight_offset + 1.5 * math.tan(math.radians(self.incline_deg)) / 2.0
			self.wedgePos = [0, 0, self.wedge_halfheight]
			self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

			if not (self.downhill):
				wedge_model_path = "gym_sloped_terrain/envs/Wedges/uphill/urdf/wedge_" + str(
					self.incline_deg) + ".urdf"

				self.INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler(
					[math.radians(self.incline_deg) * math.sin(self.incline_ori),
					 -math.radians(self.incline_deg) * math.cos(self.incline_ori), 0])

				self.robot_landing_height = wedge_halfheight_offset + 0.28 + math.tan(
					math.radians(self.incline_deg)) * abs(self.wedge_start)

				# self.INIT_POSITION = [self.INIT_POSITION[0], self.INIT_POSITION[1], self.robot_landing_height]
				self.INIT_POSITION = [-0.1, 0, self.robot_landing_height]

			else:
				wedge_model_path = "gym_sloped_terrain/envs/Wedges/downhill/urdf/wedge_" + str(
					self.incline_deg) + ".urdf"

				self.robot_landing_height = wedge_halfheight_offset + 0.28 + math.tan(
					math.radians(self.incline_deg)) * 1.5

				self.INIT_POSITION = [0, 0, self.robot_landing_height]  # [0.5, 0.7, 0.3] #[-0.5,-0.5,0.3]

				self.INIT_ORIENTATION = [0, 0, 0, 1]

			self.wedge = self._pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)

			self.SetWedgeFriction(0.7)


		model_path = 'gym_sloped_terrain/envs/robots/stochlite/stochlite_description/urdf/stochlite_urdf.urdf'
		self.stochlite = self._pybullet_client.loadURDF(model_path, self.INIT_POSITION,self.INIT_ORIENTATION)

		self._joint_name_to_id, self._motor_id_list  = self.BuildMotorIdList()

		num_legs = 4
		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=True)
		self.ResetPoseForAbd()

		if self._on_rack:
			self._pybullet_client.createConstraint(
				self.stochlite, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
				[0, 0, 0], [0, 0, 0], [0, 0, 0.4])

		self._pybullet_client.resetBasePositionAndOrientation(self.stochlite, self.INIT_POSITION, self.INIT_ORIENTATION)
		self._pybullet_client.resetBaseVelocity(self.stochlite, [0, 0, 0], [0, 0, 0])

		self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
		self.SetFootFriction(self.friction)
		# self.SetLinkMass(0,0)
		# self.SetLinkMass(11,0)

	def reset_standing_position(self):
		num_legs = 4
		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=False, standstilltorque=10)
		self.ResetPoseForAbd()

		# Conditions for standstill
		for i in range(300):
			self._pybullet_client.stepSimulation()

		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=False, standstilltorque=0)


	def reset(self):
		'''
		This function resets the environment 
		Note : Set_Randomization() is called before reset() to either randomize or set environment in default conditions.
		'''
		self._theta = 0
		self._last_base_position = [0, 0, 0]
		self.commands = [0, 0, 0]
		self.last_yaw = 0
		self.inverse = False

		if self._is_wedge:
			self._pybullet_client.removeBody(self.wedge)

			wedge_halfheight_offset = 0.01

			self.wedge_halfheight = wedge_halfheight_offset + 1.5 * math.tan(math.radians(self.incline_deg)) / 2.0
			self.wedgePos = [0, 0, self.wedge_halfheight]
			self.wedgeOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0, self.incline_ori])

			if not (self.downhill):
				wedge_model_path = "gym_sloped_terrain/envs/Wedges/uphill/urdf/wedge_" + str(self.incline_deg) + ".urdf"

				self.INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler(
					[math.radians(self.incline_deg) * math.sin(self.incline_ori),
					 -math.radians(self.incline_deg) * math.cos(self.incline_ori), 0])

				self.robot_landing_height = wedge_halfheight_offset + 0.28 + math.tan(math.radians(self.incline_deg)) * abs(self.wedge_start)

				# self.INIT_POSITION = [self.INIT_POSITION[0], self.INIT_POSITION[1], self.robot_landing_height]
				self.INIT_POSITION = [-1, 0, self.robot_landing_height]


			else:
				wedge_model_path = "gym_sloped_terrain/envs/Wedges/downhill/urdf/wedge_" + str(self.incline_deg) + ".urdf"

				self.robot_landing_height = wedge_halfheight_offset + 0.28 + math.tan(math.radians(self.incline_deg)) * 1.5

				self.INIT_POSITION = [0, 0, self.robot_landing_height]  # [0.5, 0.7, 0.3] #[-0.5,-0.5,0.3]

				self.INIT_ORIENTATION = [0, 0, 0, 1]


			self.wedge = self._pybullet_client.loadURDF(wedge_model_path, self.wedgePos, self.wedgeOrientation)
			self.SetWedgeFriction(0.7)

		self._pybullet_client.resetBasePositionAndOrientation(self.stochlite, self.INIT_POSITION, self.INIT_ORIENTATION)
		self._pybullet_client.resetBaseVelocity(self.stochlite, [0, 0, 0], [0, 0, 0])
		self.reset_standing_position()

		self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
		self._n_steps = 0
		return self.GetObservation()


	def updateCommands(self, num_plays, episode_length):
		ratio = num_plays/episode_length
		if num_plays < 0.2 * episode_length:
			self.commands = [0, 0, 0]
		# elif num_plays < 0.6 * episode_length:
		# 	self.commands = [0.5 * self.max_linear_xvel, 0.5 * self.max_linear_yvel, 0.5 * self.max_ang_vel]
		else:
			# self.commands = [self.max_linear_xvel, self.max_linear_yvel, self.max_ang_vel]
			self.commands = np.array([self.max_linear_xvel, self.max_linear_yvel, self.max_ang_vel])*ratio

	def apply_Ext_Force(self, x_f, y_f,link_index= 1,visulaize = False,life_time=0.01):
		'''
		function to apply external force on the robot
		Args:
			x_f  :  external force in x direction
			y_f  : 	external force in y direction
			link_index : link index of the robot where the force need to be applied
			visulaize  :  bool, whether to visulaize external force by arrow symbols
			life_time  :  life time of the visualization
 		'''
		force_applied = [x_f,y_f,0]
		self._pybullet_client.applyExternalForce(self.stochlite, link_index, forceObj=[x_f,y_f,0],posObj=[0,0,0],flags=self._pybullet_client.LINK_FRAME)
		f_mag = np.linalg.norm(np.array(force_applied))

		if(visulaize and f_mag != 0.0):
			point_of_force = self._pybullet_client.getLinkState(self.stochlite, link_index)[0]
			
			lam = 1/(2*f_mag)
			dummy_pt = [point_of_force[0]-lam*force_applied[0],
			 			point_of_force[1]-lam*force_applied[1],
			 			point_of_force[2]-lam*force_applied[2]]
			self._pybullet_client.addUserDebugText(str(round(f_mag,2))+" N",dummy_pt,[0.13,0.54,0.13],textSize=2,lifeTime=life_time)
			self._pybullet_client.addUserDebugLine(point_of_force,dummy_pt,[0,0,1],3,lifeTime=life_time)


	def SetLinkMass(self,link_idx,mass=0):
		'''
		Function to add extra mass to front and back link of the robot

		Args:
			link_idx : link index of the robot whose weight to need be modified
			mass     : value of extra mass to be added

		Ret:
			new_mass : mass of the link after addition
		Note : Presently, this function supports addition of masses in the front and back link only (0, 11)
		'''
		link_mass = self._pybullet_client.getDynamicsInfo(self.stochlite,link_idx)[0]
		if(link_idx==0):
			link_mass = mass  # mass + 1.1
			self._pybullet_client.changeDynamics(self.stochlite, 0, mass=link_mass)
		elif(link_idx==11):
			link_mass = mass  # mass + 1.1
			self._pybullet_client.changeDynamics(self.stochlite, 11, mass=link_mass)

		return link_mass
		

	def getlinkmass(self,link_idx):
		'''
		function to retrieve mass of any link
		Args:
			link_idx : link index of the robot
		Ret:
			m[0] : mass of the link
		'''
		m = self._pybullet_client.getDynamicsInfo(self.stochlite,link_idx)
		return m[0]
	

	def Set_Randomization(self, default = True, idx1 = 0, idx2=0, idx3=2, idx0=0, idx11=0, idxc=2, idxp=0, deg = 5, ori = 0): # deg = 5, changed for stochlite
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
			avail_deg = [5, 7, 9, 11] # [5,7,9,11], changed for stochlite
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

	def randomize_only_inclines(self, default=True, idx1=0, idx2=0, deg = 5, ori = 0): # deg = 5, changed for stochlite
		'''
		This function only randomizes the wedge incline and orientation and is called during training without Domain Randomization
		'''
		if default:
			self.incline_deg = deg + 2 * idx1
			# self.incline_ori = ori + PI / 12 * idx2

		else:
			avail_deg = [5, 7, 9, 11] # [5, 7, 9, 11]
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
		step_length_offset = self.max_steplength * math.sqrt(self.commands[0]**2 + self.commands[1]**2)/math.sqrt(self.max_linear_xvel**2 + self.max_linear_yvel**2)

		sp_pitch_offset = self.max_x_shift * np.degrees(self.support_plane_estimated_pitch) / self.max_incline
		# sp_roll_offset = self.max_z_shift * np.degrees(self.support_plane_estimated_roll) / self.max_incline
		# print("Sp pitch", np.degrees(self.support_plane_estimated_pitch))
		# print("pitch offset", sp_pitch_offset)

		action = np.clip(action,-1,1)

		action[:4] = (action[:4]+1)/2*0.4*self.max_steplength  #+ 0.15 	# Step lengths are positive always
		
		action[:4] = action[:4] + step_length_offset  # Max step length = 2x0.1 =0.2

		action[8:12] = action[8:12] + sp_pitch_offset #- 0.08

		action[12:16] = action[12:16] * self.max_y_shift + 0.04 # np.clip(action[12:16], -0.14, 0.14)  #the Y_shift can be +/- 0.14 from the leg zero position, max abd angle = +/- 30 deg 

		action[12] = -action[12]
		action[14] = -action[14]

		action[16:20] = action[16:20] * self.max_z_shift

		# action[16] = action[16] * self.max_z_shift + sp_roll_offset
		# action[17] = action[17] * self.max_z_shift - sp_roll_offset # Z_shift can be +/- 0.1 from z center
		# action[18] = action[18] * self.max_z_shift + sp_roll_offset
		# action[19] = action[19] * self.max_z_shift - sp_roll_offset

		# print("in env", action)

		return action

	def get_foot_contacts(self):
		'''
		Retrieve foot contact information with the supporting ground and any special structure (wedge/stairs).
		Ret:
			foot_contact_info : 8 dimensional binary array, first four values denote contact information of feet [FR, FL, BR, BL] with the ground
			while next four with the special structure. 
		'''
		foot_ids = [5,2,11,8]
		foot_contact_info = np.zeros(8)

		for leg in range(4):
			contact_points_with_ground = self._pybullet_client.getContactPoints(self.plane, self.stochlite, -1, foot_ids[leg])
			# print(contact_points_with_ground)
			if len(contact_points_with_ground) > 0:
				foot_contact_info[leg] = 1
				# print("Contact", foot_contact_info[leg])

			if self._is_wedge:
				contact_points_with_wedge = self._pybullet_client.getContactPoints(self.wedge, self.stochlite, -1, foot_ids[leg])
				if len(contact_points_with_wedge) > 0:
					foot_contact_info[leg+4] = 1
					# print("Contact", foot_contact_info[leg])

			if self._is_stairs:
				for steps in self.stairs:
					contact_points_with_stairs = self._pybullet_client.getContactPoints(steps, self.stochlite, -1,
																					   foot_ids[leg])
					if len(contact_points_with_stairs) > 0:
						foot_contact_info[leg + 4] = 1

		# print(foot_contact_info)
		# foot_contact_info = [1, 1, 1, 1]
		return foot_contact_info


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
		current_w = self.GetBaseAngularVelocity()[2]
		current_v = self.GetBaseLinearVelocity()
		radial_v = math.sqrt(current_v[0]**2 + current_v[1]**2)
		return radial_v, current_w


	def do_simulation(self, action, n_frames):
		'''
		Converts action parameters to corresponding motor commands with the help of a elliptical trajectory controller
		'''
		omega = 2 * no_of_points * self._frequency  
		self.action = action
		ii = 0

		leg_m_angle_cmd = self._walkcon.run_elliptical_Traj_Stochlite(self._theta,action)

		self._theta = constrain_theta(omega * self.dt + self._theta)
		
		m_angle_cmd_ext = np.array(leg_m_angle_cmd)

		m_vel_cmd_ext = np.zeros(12)

		force_visualizing_counter = 0

		for _ in range(n_frames):
			ii = ii + 1
			applied_motor_torque = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
			self._pybullet_client.stepSimulation()

			if self._n_steps >=self.pertub_steps and self._n_steps <= self.pertub_steps + self.stride:
				force_visualizing_counter += 1
				if(force_visualizing_counter%7==0):
					self.apply_Ext_Force(self.x_f,self.y_f,visulaize=True,life_time=0.1)
				else:
					self.apply_Ext_Force(self.x_f,self.y_f,visulaize=False)

	
		contact_info = self.get_foot_contacts()
		pos, ori = self.GetBasePosAndOrientation()

		Rot_Mat = self._pybullet_client.getMatrixFromQuaternion(ori)
		Rot_Mat = np.array(Rot_Mat)
		Rot_Mat = np.reshape(Rot_Mat,(3,3))

		plane_normal, self.support_plane_estimated_roll, self.support_plane_estimated_pitch = normal_estimator.vector_method_Stochlite(self.prev_incline_vec, contact_info, self.GetMotorAngles(), Rot_Mat)
		self.prev_incline_vec = plane_normal

		log_dir = os.getcwd()
		self.logger.log_kv("Robot_roll", pos[0])
		self.logger.log_kv("Robot_pitch", pos[1])
		self.logger.log_kv("SP_roll", self.support_plane_estimated_roll)
		self.logger.log_kv("SP_pitch", self.support_plane_estimated_pitch)
		self.logger.save_log(log_dir + '/experiments/logs_sensors')

		# print("estimate", self.support_plane_estimated_roll, self.support_plane_estimated_pitch)
		# print("incline", self.incline_deg)

		self._n_steps += 1

	def render(self, mode="rgb_array", close=False):
		if mode != "rgb_array":
			return np.array([])

		base_pos, _ = self.GetBasePosAndOrientation()
		view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
				cameraTargetPosition=base_pos,
				distance=self._cam_dist,
				yaw=self._cam_yaw,
				pitch=self._cam_pitch,
				roll=0,
				upAxisIndex=2)
		proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
				fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
				nearVal=0.1, farVal=100.0)
		(_, _, px, _, _) = self._pybullet_client.getCameraImage(
				width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
				projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

		rgb_array = np.array(px).reshape(RENDER_WIDTH, RENDER_HEIGHT, 4)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

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

		RPY_orig = self._pybullet_client.getEulerFromQuaternion(ori)
		RPY = np.round(RPY_orig, 4)

		current_height = round(pos[2], 5)
		self.current_com_height = current_height
		standing_penalty=3
	
		desired_height = (robot_height_from_support_plane)/math.cos(wedge_angle) + math.tan(wedge_angle)*((pos[0])*math.cos(self.incline_ori)+ 0.5)

		#Need to re-evaluate reward functions for slopes, value of reward function after considering error should be greater than 0.5, need to tune

		roll_reward = np.exp(-45 * ((RPY[0]-self.support_plane_estimated_roll) ** 2))
		pitch_reward = np.exp(-45 * ((RPY[1]-self.support_plane_estimated_pitch) ** 2))
		yaw_reward = np.exp(-40 * (RPY[2] ** 2)) # np.exp(-35 * (RPY[2] ** 2)) increasing reward for yaw correction
		height_reward = np.exp(-800 * (desired_height - current_height) ** 2)

		x = pos[0]
		y = pos[1]
		yaw = RPY[2]
		x_l = self._last_base_position[0]
		y_l = self._last_base_position[1]
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

		# reward_info = [0, 0, 0]
		# reward_info[0] = self.commands[0]
		# reward_info[1] = x_velocity
		# reward_info[2] = reward_info[2] + reward
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

		return reward, done

	def _apply_pd_control(self, motor_commands, motor_vel_commands):
		'''
		Apply PD control to reach desired motor position commands
		Ret:
			applied_motor_torque : array of applied motor torque values in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA]
		'''
		qpos_act = self.GetMotorAngles()
		qvel_act = self.GetMotorVelocities()
		applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)

		applied_motor_torque = np.clip(np.array(applied_motor_torque), -self.clips, self.clips)
		applied_motor_torque = applied_motor_torque.tolist()

		for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
			self.SetMotorTorqueById(motor_id, motor_torque)
		return applied_motor_torque

	def add_noise(self, sensor_value, SD = 0.04):
		'''
		Adds sensor noise of user defined standard deviation in current sensor_value
		'''
		noise = np.random.normal(0, SD, 1)
		sensor_value = sensor_value + noise[0]
		return sensor_value

	def GetObservation(self):
		'''
		This function returns the current observation of the environment for the interested task
		Ret:
			obs : [R(t-2), P(t-2), Y(t-2), R(t-1), P(t-1), Y(t-1), R(t), P(t), Y(t), estimated support plane (roll, pitch) ]
		'''
		pos, ori = self.GetBasePosAndOrientation()
		RPY = self._pybullet_client.getEulerFromQuaternion(ori)
		RPY = np.round(RPY, 5)

		for val in RPY:
			if(self.add_IMU_noise):
				val = self.add_noise(val)
			self.ori_history_queue.append(val)

		obs = np.concatenate((self.ori_history_queue, [self.support_plane_estimated_roll,self.support_plane_estimated_pitch])).ravel()

		return obs

	def GetMotorAngles(self):
		'''
		This function returns the current joint angles in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
		motor_ang = [self._pybullet_client.getJointState(self.stochlite, motor_id)[0] for motor_id in self._motor_id_list]
		# print("motor angles", motor_ang)
		return motor_ang

	def GetMotorVelocities(self):
		'''
		This function returns the current joint velocities in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
		motor_vel = [self._pybullet_client.getJointState(self.stochlite, motor_id)[1] for motor_id in self._motor_id_list]
		return motor_vel

	def GetBasePosAndOrientation(self):
		'''
		This function returns the robot torso position(X,Y,Z) and orientation(Quaternions) in world frame
		'''
		position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.stochlite))
		return position, orientation

	def GetBaseAngularVelocity(self):
		'''
		This function returns the robot base angular velocity in world frame
		Ret: list of 3 floats
		'''
		basevelocity= self._pybullet_client.getBaseVelocity(self.stochlite)
		return basevelocity[1]

	def GetBaseLinearVelocity(self):
		'''
		This function returns the robot base linear velocity in world frame
		Ret: list of 3 floats
		'''
		basevelocity= self._pybullet_client.getBaseVelocity(self.stochlite)
		return basevelocity[0]

	def SetFootFriction(self, foot_friction):
		'''
		This function modify coefficient of friction of the robot feet
		Args :
		foot_friction :  desired friction coefficient of feet
		Ret  :
		foot_friction :  current coefficient of friction
		'''
		FOOT_LINK_ID = [2,5,8,11]
		for link_id in FOOT_LINK_ID:
			self._pybullet_client.changeDynamics(
			self.stochlite, link_id, lateralFriction=foot_friction)
		return foot_friction

	def SetWedgeFriction(self, friction):
		'''
		This function modify friction coefficient of the wedge
		Args :
		foot_friction :  desired friction coefficient of the wedge
		'''
		self._pybullet_client.changeDynamics(
			self.wedge, -1, lateralFriction=friction)

	def SetMotorTorqueById(self, motor_id, torque):
		'''
		function to set motor torque for respective motor_id
		'''
		self._pybullet_client.setJointMotorControl2(
				  bodyIndex=self.stochlite,
				  jointIndex=motor_id,
				  controlMode=self._pybullet_client.TORQUE_CONTROL,
				  force=torque)

	def BuildMotorIdList(self):
		'''
		function to map joint_names with respective motor_ids as well as create a list of motor_ids
		Ret:
		joint_name_to_id : Dictionary of joint_name to motor_id
		motor_id_list	 : List of joint_ids for respective motors in order [FLH FLK FRH FRK BLH BLK BRH BRK FLA FRA BLA BRA ]
		'''
		num_joints = self._pybullet_client.getNumJoints(self.stochlite)
		joint_name_to_id = {}
		for i in range(num_joints):
			joint_info = self._pybullet_client.getJointInfo(self.stochlite, i)
			# print(joint_info[0], joint_info[1])
			joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

		# print(joint_info)
		# print(joint_info[1].decode("UTF-8"))

		MOTOR_NAMES = [ "fl_hip_joint",
						"fl_knee_joint",
						"fr_hip_joint",
						"fr_knee_joint",
						"bl_hip_joint",
						"bl_knee_joint",
						"br_hip_joint",
						"br_knee_joint",
						"fl_abd_joint",
						"fr_abd_joint",
						"bl_abd_joint",
						"br_abd_joint"]

		motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

		return joint_name_to_id, motor_id_list

	def ResetLeg(self, leg_id, add_constraint, standstilltorque=10):
		'''
		function to reset hip and knee joints' state
		Args:
			 leg_id 		  : denotes leg index
			 add_constraint   : bool to create constraints in lower joints of five bar leg mechanisim
			 standstilltorque : value of initial torque to set in hip and knee motors for standing condition
		'''
		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["fl_hip_joint"],  # motor
            targetValue=0.772001, targetVelocity=0)

		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["fl_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["fr_hip_joint"],
            targetValue=0.772001, targetVelocity=0)
		
		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["fr_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["bl_hip_joint"],  # motor
            targetValue=0.772001, targetVelocity=0)

		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["bl_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["br_hip_joint"],
            targetValue=0.772001, targetVelocity=0)

		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["br_hip_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["fl_knee_joint"],  # motor
            targetValue=-1.4129, targetVelocity=0)

		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["fl_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["fr_knee_joint"],
            targetValue=-1.4129, targetVelocity=0)

		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["fr_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["bl_knee_joint"],  # motor
            targetValue=-1.4129, targetVelocity=0)

		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["bl_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["br_knee_joint"],
            targetValue=-1.4129, targetVelocity=0)

		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["br_knee_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )


	def ResetPoseForAbd(self):
		'''
		Reset initial conditions of abduction joints
		'''
		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["fl_abd_joint"],
            targetValue=0, targetVelocity=0)

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["fr_abd_joint"],
            targetValue=0, targetVelocity=0)

		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["bl_abd_joint"],
            targetValue=0, targetVelocity=0)
			
		self._pybullet_client.resetJointState(
            self.stochlite,
            self._joint_name_to_id["br_abd_joint"],
            targetValue=0, targetVelocity=0)

        # Set control mode for each motor and initial conditions
		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["fl_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )
		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["fr_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )
		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["bl_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )
		self._pybullet_client.setJointMotorControl2(
            bodyIndex=self.stochlite,
            jointIndex=(self._joint_name_to_id["br_abd_joint"]),
            controlMode=self._pybullet_client.VELOCITY_CONTROL,
            force=0,
            targetVelocity=0
        )




