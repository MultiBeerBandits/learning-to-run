from baselines.policy.mlp_policy import MlpPolicy
from baselines.envs.lqg1d import LQG1D
import numpy as np
import baselines.common.tf_util as U

sess = U.single_threaded_session()
sess.__enter__()

env = LQG1D()

pi = MlpPolicy("pi",env.observation_space,env.action_space,hid_size=1,
               num_hid_layers=1,use_bias=False)

be = MlpPolicy("be",env.observation_space,env.action_space,hid_size=1,
               num_hid_layers=0,use_bias=False)
s = np.array([[1.],[2.],[1.]])

a = np.array([[0.],[1.],[-1.]])

r = np.array([[1.],[0.],[1.]])