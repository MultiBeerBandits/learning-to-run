from baselines.ddpg.memory import Memory
from itertools import product
import numpy as np

class ReplayBufferFlip(Memory):

    def __init__(self, limit, action_shape, observation_shape, flip_state, state_description):
        super(ReplayBufferFlip, self).__init__(limit, action_shape, observation_shape)
        self.flip_state = flip_state
        self.left_idx = self.get_idx(state_description, "_left")
        self.right_idx = self.get_idx(state_description, "_right")
        self.action_space_size = action_shape[0]

    def sample(self, batch_size):
        buff = super.sample(batch_size)
        if self.flip_state:
            buff['obs0'] = np.hstack((buff['obs0'], self.swap_states(buff['obs0'])))
            buff['obs1'] = np.hstack((buff['obs1'], self.swap_states(buff['obs1'])))
            buff['rewards'] = np.hstack((buff['rewards'], buff['rewards']))
            buff['actions'] = np.hstack((buff['actions'], self.swap_actions(buff['actions'])))
            buff['terminals1'] = np.hstack((buff['terminals1'], buff['terminals1']))
        return buff

    def env_features_names(self):
        names = ['pelvis_' + var for var in ['y', 'vx', 'vy', 'ax', 'ay', 'rz', 'vrz', 'arz']]
        names += [body_part + "_" + var for (body_part, var) in product(['head', 'torso', 'toes_left', 'toes_right', 'talus_left', 'talus_right'], ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'rz', 'vrz', 'arz'])]
        names += [body_part + "_" + var for (body_part, var) in product(['ankle_left', 'ankle_right', 'back', 'hip_left', 'hip_right', 'knee_left', 'knee_right'], ['rz', 'vrz', 'arz'])]
        names += ['center_of_mass' + var for var in ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'ofg']]
        return names

    def  get_idx(self, desc, side):
        names = self.env_features_names()
        idx = [i for i, el in enumerate(names) if side in el]
        return idx

    def swap_states(self, states):
        new_states = states.copy()
        left = states[:, self.left_idx]
        right = states[:, self.right_idx]
        new_states[:, self.left_idx] = right
        new_states[:, self.right_idx] = left
        return new_states

    def swap_actions(self, actions):
        new_actions = np.zeros_like(actions)
        new_actions[:, : self.action_space_size // 2] = actions[:, self.action_space_size // 2 :]
        new_actions[:, self.action_space_size // 2 :] = actions[:, : self.action_space_size // 2]
        return new_actions

