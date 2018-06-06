import tensorflow as tf
from baselines.ddpg.ddpg import DDPG
from baselines import logger
import baselines.common.tf_util as U
from baselines.common import set_global_seeds
import numpy as np
import time
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from multiprocessing.pool import Pool
from multiprocessing import Process, Queue, Event
from env_wrapper import create_environment
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
import os
import os.path
import time


def test(env, actor, critic, memory, normalize_observations, gamma, reward_scale, nb_episodes,
         episode_length, checkpoint_dir):

    # Initialize DDPG agent (target network and replay buffer)
    agent = DDPG(actor, critic, memory, env.observation_space.shape,
                 env.action_space.shape, gamma=gamma,
                 normalize_observations=normalize_observations,
                 reward_scale=reward_scale)

    # We need max_action because the NN output layer is a tanh.
    # So we must scale it back.
    max_action = env.action_space.high

    # Start testing loop
    with U.single_threaded_session() as sess:
        agent.initialize(sess)

        # setup saver
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

        # restore all
        print("restoring variables")
        # Add ops to save and restore all the variables.
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        step_times = []
        for eval_episode in range(nb_episodes):
            print("Evaluating episode {}...".format(eval_episode))
            obs = env.reset()
            for t in range(episode_length):

                # Select action a_t without noise
                a_t, _ = agent.pi(
                    obs, apply_param_noise=False, apply_action_noise=False, compute_Q=False)
                assert a_t.shape == env.action_space.shape

                # Execute action a_t and observe reward r_t and next state s_{t+1}
                start_step_time = time.time()
                obs, r_t, eval_done, info = env.step(
                    max_action * a_t)
                end_step_time = time.time()
                step_time = end_step_time - start_step_time
                step_times.append(step_time)

                if eval_done:
                    print("  Episode done!")
                    obs = env.reset()
                    break
        print("Average step time: ", np.mean(step_times))