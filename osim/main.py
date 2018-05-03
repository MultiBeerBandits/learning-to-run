import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

#import baselines.ddpg.training as training
import training

from model import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

from osim.env.osim import L2RunEnv
import tensorflow as tf
from mpi4py import MPI


def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):

    param_noise = None
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)


    # Create envs.
    env = L2RunEnv(visualize=True)
    eval_env = None

    # Parse noise_type
    nb_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(nb_actions), sigma=np.ones(nb_actions))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape)
    actor = Actor(nb_actions, layer_norm=layer_norm)
    critic = Critic(layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(
        rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=env, action_noise=action_noise,
                   actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    #boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    # with default settings, perform 1M steps total
    parser.add_argument('--nb_epochs', type=int, default=500)
    parser.add_argument('--nb_episodes', type=int, default=20)
    # per epoch cycle and MPI worker
    parser.add_argument('--nb_train_steps', type=int, default=50)
    parser.add_argument('--nb_eval_episodes', type=int, default=5)
    # per epoch cycle and MPI worker
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=20)
    # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs *
               args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    run(**args)