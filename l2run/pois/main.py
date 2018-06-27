import baselines.common.tf_util as U
import baselines.pgpe.neuronwise_poisnpe_par as multipoisnpe
import numpy as np

from baselines.common import (set_global_seeds, boolean_flag)
from baselines.pgpe.parallel_sampler import ParallelSampler
from baselines.policy.neuronwise_pemlp_policy import MultiPeMlpPolicy
from l2run.env_wrapper import create_environment


'''
Training script for multiprocess POIS algorithm (with multi-layer perceptron
policy) in L2Run OpenSim environment.
'''


def train(batch_size, gamma, horizon, seed, shift, normalize, use_rmax, use_renyi, path, delta, full,
          action_repeat, reward_scale, fail_reward, exclude_centering_frame,
          integrator_accuracy, num_processes):
    DIR = path + '/results/multipoisnpe' + \
        '/seed_' + str(seed)
    import os
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # Maker function for OpenSim L2Run Environment
    def env_maker():
        env = create_environment(False, full, action_repeat, reward_scale,
                                 fail_reward, exclude_centering_frame, integrator_accuracy)
        env.reset()
        env.seed(seed)
        return env

    # Maker function for multiprocess Multi-layer Perceptron Policy
    def pol_maker(name, observation_space, action_space): return MultiPeMlpPolicy(name,
                                                                                  observation_space,
                                                                                  action_space,
                                                                                  hid_layers=[
                                                                                      64, 32, 16],
                                                                                  use_bias=True,
                                                                                  seed=seed)

    sampler = ParallelSampler(
        env_maker, pol_maker, gamma, horizon, np.ravel, batch_size, num_processes, seed)

    sess = U.make_session()
    sess.__enter__()

    set_global_seeds(seed)

    multipoisnpe.learn(env_maker,
                       pol_maker,
                       sampler,
                       gamma=gamma,
                       initial_batch_size=batch_size,
                       task_horizon=horizon,
                       max_iterations=500,
                       save_to=DIR,
                       verbose=2,
                       feature_fun=np.ravel,
                       normalize=normalize,
                       use_rmax=use_rmax,
                       use_renyi=use_renyi,
                       max_offline_ite=20,
                       max_search_ite=30,
                       delta=delta,
                       shift=shift,
                       use_parabola=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--path', help='save here', type=str, default='temp')
    parser.add_argument('--seed', help='RNG seed', type=int, default=10)
    parser.add_argument('--shift', help='Normalize return?',
                        type=int, default=0)
    parser.add_argument(
        '--normalize', help='Normalize weights?', type=int, default=1)
    parser.add_argument(
        '--use_rmax', help='Use Rmax in bound (or var)?', type=int, default=1)
    parser.add_argument(
        '--use_renyi', help='Use Renyi in ESS (or weight norm)?', type=int, default=0)
    parser.add_argument('--delta', help='delta', type=str, default='0.6')
    parser.add_argument('--action-repeat', type=int, default=1)
    boolean_flag(parser, 'full', default=False, help="use full observation")
    boolean_flag(parser, 'exclude-centering-frame', default=False,
                 help="exclude pelvis from observation vec")
    parser.add_argument('--reward-scale', type=float, default=1.0)
    parser.add_argument('--fail-reward', type=float, default=-0.2)
    # Reduce accuracy -> Improve efficiency
    # Good value may be 3e-2
    parser.add_argument('--integrator-accuracy', type=float, default=5e-5)
    parser.add_argument('--num-processes', type=int, default=1)

    args = parser.parse_args()
    delta = float(args.delta)

    # Start training
    train(args.batch_size,
          args.gamma,
          args.horizon,
          args.seed,
          args.shift,
          args.normalize,
          args.use_rmax,
          args.use_renyi,
          args.path,
          delta,
          args.full,
          args.action_repeat,
          args.reward_scale,
          args.fail_reward,
          args.exclude_centering_frame,
          args.integrator_accuracy,
          args.num_processes)
