from baselines.ddpg.memory import Memory


class ReplayBuffer(Memory):

    def __init__(self, limit, action_shape, observation_shape, flip_state):
        super(limit, action_shape, observation_shape)
        self.flip_state = flip_state

    def sample(self, batch_size):
        # TODO need to override to implement flipped actions
        pass