	import numpy as np


    def  (self):
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

		# MOTOR_NAMES = [ "fl_hip_joint",
		# 				"fl_knee_joint",
		# 				"fr_hip_joint",
		# 				"fr_knee_joint",
		# 				"bl_hip_joint",
		# 				"bl_knee_joint",
		# 				"br_hip_joint",
		# 				"br_knee_joint",
		# 				"fl_abd_joint",
		# 				"fr_abd_joint",
		# 				"bl_abd_joint",
		# 				"br_abd_joint"]



        Motor_NAMES=["FR_abd",
                     "FR_hip",
                     "FR_knee",
                     "FR_foot",
                     "FL_abd",
                     "FL_hip",
                     "FL_knee",
                     "FL_foot",
                     "BR_abd",
                     "BR_hip",
                     "BR_knee",
                     "BR_foot",
                     "BL_abd",
                     "BL_hip",
                     "BL_knee",
                     "BL_foot"]
        
        
        
        
        

        # Motor_NAMES=["FR_abd_joint",
        #              "FR_hip_joint",
        #              "FR_knee_joint",
        #              "FR_foot_joint",
        #              "FL_abd_joint",
        #              "FL_hip_joint",
        #              "FL_knee_joint",
        #              "FL_foot_joint",
        #              "BR_abd_joint",
        #              "BR_hip_joint",
        #              "BR_knee_joint",
        #              "BR_foot_joint",
        #              "BL_abd_joint",
        #              "BL_hip_joint",
        #              "BL_knee_joint",
        #              "BL_foot_joint"]
		motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

		return joint_name_to_id, motor_id_list