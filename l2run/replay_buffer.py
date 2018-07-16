from baselines.ddpg.memory import Memory
import numpy as np


class ReplayBufferFlip(Memory):

    def __init__(self, limit, flip_state, obs_vector_names, action_shape, obs_shape):
        super(ReplayBufferFlip, self).__init__(limit, action_shape, obs_shape)
        self.flip_state = flip_state
        self.obs_vector_names = obs_vector_names
        self.left_idx = self.get_idx("_left")
        self.right_idx = self.get_idx("_right")
        self.action_space_size = action_shape[0]

    def sample(self, batch_size):
        if self.flip_state:
            batch_size = batch_size // 2
        buff = super(ReplayBufferFlip, self).sample(batch_size)
        if self.flip_state:
            buff['obs0'] = np.vstack(
                (buff['obs0'], self.swap_states(buff['obs0'])))
            buff['obs1'] = np.vstack(
                (buff['obs1'], self.swap_states(buff['obs1'])))
            buff['rewards'] = np.vstack((buff['rewards'], buff['rewards']))
            buff['actions'] = np.vstack(
                (buff['actions'], self.swap_actions(buff['actions'])))
            buff['terminals1'] = np.vstack(
                (buff['terminals1'], buff['terminals1']))
        return buff

    def get_idx(self, side):
        # get the idx of names that contains side ("left" od "right")
        idx = [i for i, el in enumerate(self.obs_vector_names) if side in el]
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
        new_actions[:, : self.action_space_size //
                    2] = actions[:, self.action_space_size // 2:]
        new_actions[:, self.action_space_size //
                    2:] = actions[:, : self.action_space_size // 2]
        return new_actions
