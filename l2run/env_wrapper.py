import numpy as np
import gym
from gym.spaces import Box
import gym
from osim.env import L2RunEnv
import os
from osim.env.utils.mygym import convert_to_gym
from itertools import product
import opensim
import math

class L2RunEnvWrapper(gym.Wrapper):

    def __init__(self, env, full=False, action_repeat=5, fail_reward=-0.2, 
                 exclude_centering_frame=False):
        """
        Initialize the environment:
        Parameters:
        - full: uses as observation vector the full observation vector
        - skipFrame : How many frame to skip every action
        - exclude_centering_frame: put or not the pelvis x and y in obs vector
                                   (obs are centered wrt pelvis)
        """
        gym.Wrapper.__init__(self, env)
        env.reset()
        self.full = full
        self.env = env
        self.action_repeat = action_repeat
        self.fail_reward = fail_reward
        self.exclude_centering_frame = exclude_centering_frame
        self.env_step = 0
        if self.full:
            self.get_observation = self.get_observation_full
        else:
            self.get_observation = self.get_observation_basic

        self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )
        self.observation_space = convert_to_gym(self.observation_space)
        

    def reset(self, **kwargs):
        self.env_step = 0
        self.env.reset(**kwargs)
        return self.get_observation()

    def seed(self, seed):
        self.env.seed(seed)

    def step(self, action):
        total_reward = 0.
        for _ in range(self.action_repeat):
            observation, _, done, _ = self.env.step(action)
            reward = self.reward()
            observation = self.get_observation()
            total_reward += reward
            self.env_step += 1
            if done:
                if self.env_step < 1000:  # hardcoded
                    total_reward += self.fail_reward
                break

        #total_reward *= self.reward_scale
        return observation, total_reward, done, None


    def is_done(self):
        state_desc = self.env.get_state_desc()
        return state_desc["body_pos"]["pelvis"][1] < 0.7

    def get_state_desc(self):
        return self.env.get_state_desc()

    ## Values in the observation vector
    def get_observation_basic(self):
        """
        Returns the basic observation vector with positions and velocities
        TODO: check zeros at the end of observation vector
        """
        state_desc = self.env.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []

        # obtain pelvis x,y coordinates, that will be used for centering of body 
        # poses x y. add it to obs vec only if required
        pelvis = state_desc["body_pos"]["pelvis"][0:2]
        if not self.exclude_centering_frame:
            res += pelvis

        for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r", "ground_pelvis"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]

        # center body parts poses in pelvis reference
        for body_part in ["head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            res += [state_desc["body_pos"][body_part][i] - pelvis[i] for i in range(2)]
        # center in pelvis reference also the center of mass
        res += [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res += state_desc["misc"]["mass_center_vel"]

        # strenght of left and right psoas, nex obstacle distance x from pelvis, y of the 
        # center relative to the ground, radius
        # here are set to 0
        # res += [0]*5 TODO

        return res

    ## Values in the observation vector
    def get_observation_full(self):
        """
        Returns the full observation vector with positions, velocities, accelerations and muscles
        To do: check zeros at the end of observation vector
        """
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
            # if self.prosthetic and body_part in ["toes_r","talus_r"]:
            #     res += [0] * 9
            #     continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            # store pelvis pose in pelvis var and use it for centering of 
            # body poses. If asked also add pelvis to obs vector
            if body_part == "pelvis":
                pelvis = cur
                # add pelvis to the observation vector
                if not self.exclude_centering_frame:
                    res += cur
                # otherwise keep only velocities and acc of pelvis
                else:
                    res += cur[2:]
            # if it is not pelvis, then brings everithing in the pelvis 
            # reference system
            else:
                cur_upd = cur
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
                res += cur

        for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in state_desc["muscles"].keys():
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res

    def get_observation_space_size(self):
        return len(self.get_observation())

    def reward(self):
        state_desc = self.env.get_state_desc()
        prev_state_desc = self.env.get_prev_state_desc()
        # velocity + small reward for each timestep
        # TODO: maybe we can use joint_vel ground_pelvis
        if not prev_state_desc:
            return 0
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26): 
            lig = opensim.CoordinateLimitForce.safeDownCast(self.env.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.env.osim_model.state) ** 2
        penalty = math.sqrt(lig_pen) * 10e-8
        speed = state_desc["joint_pos"]["ground_pelvis"][1] - prev_state_desc["joint_pos"]["ground_pelvis"][1]
        return speed - penalty + 0.001

    # utility methods that can be used outside for implementing actions flip
    # returns all the names of the observation vector values in order
    # (depending on full observation space or base observation space)
    # rigth and left strings are put in flippable features of the obs space
    def get_observation_names(self):
        if self.full:
            names = [body_part + "_" + var for (body_part, var) in product(['pelvis', 'head', 'torso', 'toes_left', 'toes_right', 'talus_left', 'talus_right'], ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'rz', 'vrz', 'arz'])]
            names += [body_part + "_" + var for (body_part, var) in product(['ankle_left', 'ankle_right', 'back', 'hip_left', 'hip_right', 'knee_left', 'knee_right'], ['rz', 'vrz', 'arz'])]
            names += ['center_of_mass' + var for var in ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'ofg']]
        else:
            names = ["pelvis_x", "pelvis_y"]
            names += [joint + "_" + var for (joint, var) in product(["hip_left","hip_right","knee_left","knee_right","ankle_left","ankle_right"], ["rz", "vrz"])]
            names += ["ground_pelvis_rot", "ground_pelvis_x", "ground_pelvis_y", "ground_pelvis_vel_rot", "ground_pelvis_vel_x", "ground_pelvis_vel_y"]
            names += [body_part + "_" + var for (body_part, var) in product(["head", "torso", "toes_left", "toes_right", "talus_left", "talus_right"], ["x", "y"])]
            names += ["com_x", "com_y", "com_vel_x", "com_vel_y"]
        # if exclude_centering_frame need to remove x and y of pelvis (first 2 el)
        if self.exclude_centering_frame:
            names = names[2:]
        assert len(names) == self.get_observation_space_size()
        return names
            
def create_environment(visualize, full, action_repeat, fail_reward, exclude_centering_frame):
    env = L2RunEnv(visualize=visualize)
    env = L2RunEnvWrapper(env, full, action_repeat, fail_reward, exclude_centering_frame)
    return env