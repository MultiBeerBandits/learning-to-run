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
import testing

from model import Actor, Critic
#from baselines.ddpg.memory import Memory
from replay_buffer import ReplayBufferFlip
from baselines.ddpg.noise import *

from env_wrapper import create_environment
import tensorflow as tf
from mpi4py import MPI


def run(seed, noise_type, layer_norm, evaluation, flip_state,
        full, action_repeat, fail_reward, exclude_centering_frame, **kwargs):

    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if kwargs['num_timesteps'] is not None:
        assert(kwargs['num_timesteps'] == kwargs['nb_epochs'] *
               kwargs['nb_epoch_cycles'] * kwargs['nb_rollout_steps'])
    param_noise = None
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Main env
    env = create_environment(False, full, action_repeat, fail_reward, exclude_centering_frame)
    env.reset()
    eval_env = None

    # Parse noise_type
    nb_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(nb_actions), sigma=0.2*np.ones(nb_actions), theta=0.1)

    # Configure components.
    memory = ReplayBufferFlip(int(5e6),
                              flip_state,
                              env.get_observation_names(),
                              env.action_space.shape,
                              env.observation_space.shape)
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
    
    del kwargs['func']
    del kwargs['num_timesteps']
    training.train(env=env, action_noise=action_noise,
                   actor=actor, critic=critic, memory=memory,
                   visualize=False, full=full, action_repeat=action_repeat,
                   fail_reward=fail_reward, exclude_centering_frame=exclude_centering_frame,
                    **kwargs)

    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

def test(seed, layer_norm, full, action_repeat, fail_reward, exclude_centering_frame, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    # Main env
    env = create_environment(True, full, action_repeat, fail_reward, exclude_centering_frame)
    env.reset()
    eval_env = None

    # Parse noise_type
    nb_actions = env.action_space.shape[-1]

    # Configure components.
    memory = ReplayBufferFlip(int(5e6),
                              False,
                              env.get_observation_names(),
                              env.action_space.shape,
                              env.observation_space.shape)
    actor = Actor(nb_actions, layer_norm=layer_norm)
    critic = Critic(layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(
        rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    del kwargs['func']
    testing.test(env=env, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()

def build_train_args(sub_parsers):
    parser = sub_parsers.add_parser('train')
    parser.set_defaults(func=run)
    #boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    #boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    # parser.add_argument('--actor-lr', type=float, default=1e-4)
    # parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    # with default settings, perform 1M steps total
    parser.add_argument('--nb-epochs', type=int, default=500)
    parser.add_argument('--nb-epoch-cycles', type=int, default=1)
    parser.add_argument('--nb-episodes', type=int, default=20)
    # per epoch cycle and MPI worker
    parser.add_argument('--nb-train-steps', type=int, default=50)
    parser.add_argument('--nb-eval-episodes', type=int, default=5)
    # per epoch cycle and MPI worker
    parser.add_argument('--episode-length', type=int, default=100)
    parser.add_argument('--eval-freq', type=int, default=20)
    # save net weights each save-freq
    parser.add_argument('--save-freq', type=int, default=20)
    # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')
    parser.add_argument('--num-timesteps', type=int, default=None)
    # DDPG improvements
    parser.add_argument('--action-repeat', type=int, default=1)
    boolean_flag(parser, 'flip-state', default=False)
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--experiment-name', type=str, default="")
    boolean_flag(parser, 'evaluation', default=False)
    # environment wrapper args
    boolean_flag(parser, 'full', default=False, help="use full observation")
    boolean_flag(parser, 'exclude-centering-frame', default=False,
                 help="exclude pelvis from observation vec")
    parser.add_argument('--fail-reward', type=float, default=-0.2)
    
def build_test_args(sub_parsers):
    parser = sub_parsers.add_parser('test')
    parser.set_defaults(func=test)
    #boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    #boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-observations', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--nb-episodes', type=int, default=20)
    # per epoch cycle and MPI worker
    parser.add_argument('--episode-length', type=int, default=1000)
    # DDPG improvements
    parser.add_argument('--action-repeat', type=int, default=1)
    # environment wrapper args
    boolean_flag(parser, 'full', default=False, help="use full observation")
    boolean_flag(parser, 'exclude-centering-frame', default=False,
                 help="exclude pelvis from observation vec")
    parser.add_argument('--fail-reward', type=float, default=-0.2)
    parser.add_argument('--checkpoint-dir', type=str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_parsers = parser.add_subparsers()
    build_train_args(sub_parsers)
    build_test_args(sub_parsers)
    args = parser.parse_args()
    dict_args = vars(args)
    print("Running with args: ", dict_args)
    logger.configure()
    # Run actual script.
    args.func(**dict_args)
