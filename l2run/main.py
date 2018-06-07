import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

# import baselines.ddpg.training as training
import training
import testing

from model import Actor, Critic
# from baselines.ddpg.memory import Memory
from replay_buffer import ReplayBufferFlip
from baselines.ddpg.noise import *

from env_wrapper import create_environment
import tensorflow as tf
from mpi4py import MPI

from learning_session import LearningSession


def pack_run_params(seed, noise_type, layer_norm, evaluation, flip_state,
                    full, action_repeat, fail_reward, exclude_centering_frame, **kwargs):
    args = kwargs.copy()
    args['seed'] = seed
    args['noise_type'] = noise_type
    args['layer_norm'] = layer_norm
    args['evaluation'] = evaluation
    args['flip_state'] = flip_state
    args['full'] = full
    args['action_repeat'] = action_repeat
    args['fail_reward'] = fail_reward
    args['exclude_centering_frame'] = exclude_centering_frame
    return args


def run(seed, noise_type, layer_norm, evaluation, flip_state,
        full, action_repeat, fail_reward, exclude_centering_frame,
        checkpoint_dir, log_dir, session_path, last_training_step,
        integrator_accuracy, experiment_name, **kwargs):

    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if kwargs['num_timesteps'] is not None:
        assert(kwargs['num_timesteps'] == kwargs['nb_epochs'] *
               kwargs['nb_epoch_cycles'] * kwargs['nb_rollout_steps'])

    tmp_log, tmp_chkpt = get_log_and_checkpoint_dirs(experiment_name)

    if log_dir is None:
        log_dir = tmp_log
    if checkpoint_dir is None:
        checkpoint_dir = tmp_chkpt
    param_noise = None
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Main env
    env = create_environment(False, full, action_repeat,
                             fail_reward, exclude_centering_frame, integrator_accuracy)
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

    # Create LearningSession was passed
    del kwargs['func']
    sess_args = pack_run_params(seed, noise_type, layer_norm, evaluation, flip_state,
                                full, action_repeat, fail_reward, exclude_centering_frame, **kwargs)
    learning_session = LearningSession(
        session_path, checkpoint_dir, log_dir, last_training_step, **sess_args)

    del kwargs['num_timesteps']
    training.train(env=env, action_noise=action_noise,
                   actor=actor, critic=critic, memory=memory,
                   visualize=False, full=full, action_repeat=action_repeat,
                   fail_reward=fail_reward, exclude_centering_frame=exclude_centering_frame,
                   learning_session=learning_session, integrator_accuracy=integrator_accuracy,
                   **kwargs)

    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def test(seed, layer_norm, full, action_repeat, fail_reward, exclude_centering_frame,
         integrator_accuracy, render, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    # Main env
    env = create_environment(render, full, action_repeat,
                             fail_reward, exclude_centering_frame,
                             integrator_accuracy)
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


def resume(**kwargs):
    if 'dump_file' in kwargs:
        # Load session from dump file
        learning_session = LearningSession.from_file(kwargs['dump_file'])
    elif 'session_path' in kwargs:
        # Load most recent session from session path
        learning_session = LearningSession.from_last(kwargs['session_path'])
    else:
        raise ValueError(
            ('Missing required parameter.'
             'Pass either --dump-file or --session-path'))
    # Call run restoring learning session
    session_args = learning_session.args
    session_args['log_dir'] = learning_session.log_dir
    session_args['checkpoint_dir'] = learning_session.checkpoint_dir
    session_args['last_training_step'] = learning_session.last_training_step
    session_args['session_path'] = str(learning_session.session_path)
    run(**session_args)


def build_train_args(sub_parsers):
    parser = sub_parsers.add_parser('train')
    parser.set_defaults(func=run)
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    # boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--min-buffer-length', type=int, default=200)
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
    parser.add_argument('--num-testing-processes', type=int, default=1)
    parser.add_argument('--experiment-name', type=str, default="")
    boolean_flag(parser, 'evaluation', default=False)
    # environment wrapper args
    boolean_flag(parser, 'full', default=False, help="use full observation")
    boolean_flag(parser, 'exclude-centering-frame', default=False,
                 help="exclude pelvis from observation vec")
    parser.add_argument('--fail-reward', type=float, default=-0.2)
    # Reduce accuracy -> Improve efficiency
    # Good value may be 3e-2
    parser.add_argument('--integrator-accuracy', type=float, default=5e-5)
    # restore environment after some trajectories (possible mem leak)
    parser.add_argument('--max-env-traj', type=int, default=100)
    # Learning session args
    parser.add_argument('--session_path', type=str, default="log")
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--last-training_step', type=int, default=0)


def build_test_args(sub_parsers):
    parser = sub_parsers.add_parser('test')
    parser.set_defaults(func=test)
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=True)
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
    parser.add_argument('--integrator-accuracy', type=float, default=5e-5)
    parser.add_argument('--fail-reward', type=float, default=-0.2)
    parser.add_argument('--checkpoint-dir', type=str)


def build_resume_args(sub_parsers):
    parser = sub_parsers.add_parser('resume')
    parser.set_defaults(func=resume)
    parser.add_argument('--session-path', type=float,
                        help='The learning session dumps directory')
    parser.add_argument('--dump-file', type=str,
                        help='The learning session to resume')


def get_log_and_checkpoint_dirs(experiment_name):
    import datetime

    now = datetime.datetime.now()
    log_dir = "tf_logs/" + experiment_name + \
        "-" + now.strftime("%Y%m%d-%H%M%S") + "/"
    checkpoint_dir = "tf_checkpoints/" + experiment_name + \
        "-" + now.strftime("%Y%m%d-%H%M%S")
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpointfile = checkpoint_dir + "/model"
    return log_dir, checkpointfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_parsers = parser.add_subparsers()
    build_train_args(sub_parsers)
    build_test_args(sub_parsers)
    build_resume_args(sub_parsers)

    args = parser.parse_args()
    dict_args = vars(args)
    print("Running with args: ", dict_args)
    logger.configure()
    # Run actual script.
    args.func(**dict_args)
