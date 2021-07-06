import numpy as np
from gym import utils
import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class StochliteEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='stoch25.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_pose=self.sim.data.qpos[0]
        #print(x_pose)
        x_position_before = self.sim.data.qpos[0]

        # badeid= self.sim.model._geom_name2id['wedge5']
        #print('bad',badeid)
        #print(x_position_before)
        # print('bodybodyid', bodyid)
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #print(observation)
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info
    
    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()[7:]
        #since half cheetah is a planar constrained robot.. the sim.data.qpos will have 
        #[ base_x, base_z , base_pitch , + 6 joint angles]..thus 9 values
        #for quadruped the orientation will come as quaternion so 3(xyz)+4(quaternion)+12(joint angles)
        #print("position", position)
        velocity = self.sim.data.qvel.flat.copy()
        #print("velocity is", velocity)
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

if __name__ == "__main__":
    e = StochliteEnv()
    e.reset()
    
    # import time
    # step_time=[]
    
    for i in range(10000):
        
        action = e.action_space.sample()
       # print ("the action", action)
        # action =np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        # action =np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # action =np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # action =np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5,0.5,0.5,0.5, 0.5])

        #print(action.shape)
        e.step(action)
        e.render()
        print ('render')
        print("outer",i)
        """  
        if i==1 :
            start_time=time.time()
            e.step(action)
            # print('i_step',i)
            print('time_step_outer_loop',(time.time()-start_time))
            e.render()
        else:
            e.step(action)
            # print('i_step',i)
            e.render()
        """