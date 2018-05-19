import numpy as np
import gym
from gym.spaces import Box
import gym
from osim.env import L2RunEnv
import os
from osim.env.utils.mygym import convert_to_gym

class L2RunEnvWrapper(gym.Wrapper):

    def __init__(self, env, full=False, action_repeat=5, fail_reward=-0.2, **kwargs):
        """
        Initialize the environment:
        Parameters:
        - full: uses as observation vector the full observation vector
        - skipFrame : How many frame to skip every action
        """
        gym.Wrapper.__init__(self, env)
        env.reset()
        self.full = full
        self.env = env
        self.action_repeat = action_repeat
        self.fail_reward = fail_reward
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

    def step(self, action):
        total_reward = 0.
        for _ in range(self.action_repeat):
            observation, reward, done, _ = self.env.step(action)
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
        To do: check zeros at the end of observation vector
        """
        state_desc = self.env.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        res += state_desc["joint_pos"]["ground_pelvis"]
        res += state_desc["joint_vel"]["ground_pelvis"]

        for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r",]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]

        for body_part in ["head", "pelvis", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            res += state_desc["body_pos"][body_part][0:2]

        res = res + state_desc["misc"]["mass_center_pos"] + state_desc["misc"]["mass_center_vel"]

        res += [0]*5

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
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
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
        if not prev_state_desc:
            return 0
        return state_desc["joint_pos"]["ground_pelvis"][1] - prev_state_desc["joint_pos"]["ground_pelvis"][1]


def create_environment(action_repeat,full=True, **kwargs):
    env = L2RunEnv(visualize=False)

    env = L2RunEnvWrapper(env, full=full, action_repeat=action_repeat, **kwargs)

    return env