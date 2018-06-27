from multiprocessing import Process, Queue, Event
from env_wrapper import create_environment
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.memory import Memory
from baselines.common import set_global_seeds

import numpy as np
import baselines.common.tf_util as U
import os
import time
import queue


class RoundRobinTester():
    def __init__(self, num_processes,
                 actor,
                 critic,
                 episode_length,
                 nb_episodes,
                 action_repeat,
                 max_action,
                 gamma,
                 tau,
                 normalize_returns,
                 batch_size,
                 normalize_observations,
                 critic_l2_reg,
                 popart,
                 clip_norm,
                 reward_scale,
                 # environment wrapper parameters
                 full,
                 exclude_centering_frame,
                 integrator_accuracy,
                 max_env_traj,
                 visualize,
                 fail_reward):
        """
        Construct a new RoundRobinTester, spawning num_processes
        TestingWorkers.

        Arguments:
        num_processes -- number of testing workers to spawn
        Others -- parameters forwarded to TestingWorker constructor
        """
        self.inputQs = [Queue() for _ in range(num_processes)]
        self.outputQ = Queue()

        # Build workers...
        self.workers = [TestingWorker(actor,
                                      critic,
                                      episode_length,
                                      nb_episodes,
                                      action_repeat,
                                      max_action,
                                      gamma,
                                      tau,
                                      normalize_returns,
                                      batch_size,
                                      normalize_observations,
                                      critic_l2_reg,
                                      popart,
                                      clip_norm,
                                      reward_scale,
                                      self.inputQs[i],
                                      self.outputQ,
                                      full,
                                      exclude_centering_frame,
                                      integrator_accuracy,
                                      max_env_traj,
                                      visualize,
                                      fail_reward
                                      ) for i in range(num_processes)]
        # And then run them
        for i in range(num_processes):
            self.workers[i].start()

        self.num_processes = num_processes
        self.target = 0

    def test(self, actor_weights, global_step):
        """
        Dispatch a testing job to current target worker.

        Arguments:
        actor_weights -- The parameter vector for Actor net
        global_step -- The current training step
        """
        # First, dispatch task to target worker
        self.inputQs[self.target].put(('test', actor_weights, global_step))

        # Then move the scheduler
        self.target = (self.target + 1) % self.num_processes

    def log_stats(self, stats, logger):
        """
        Collect one result-set from testing workers (if any).
        Then log results.

        Arguments:
        stats -- A EvaluationStatistics object where to log statistics to
        logger -- A baselines.logger object where to dump statistics to
        """
        try:
            results = self.outputQ.get_nowait()
            rewards, step_times, distances, episode_lengths, test_step = results
            print("Collecting results from a testing worker...")
            stats.add_distances(distances)
            stats.add_episode_lengths(episode_lengths)
            stats.add_rewards(rewards)
            stats.add_step_times(step_times)
            combined_stats = {}
            stats.fill_stats(combined_stats)
            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('Step: {}'.format(test_step))
            stats.plot_distance(test_step)
            stats.plot_reward(test_step)
        except queue.Empty:
            # If no results, do nothing
            pass

    def close(self):
        """
        Send an exit signal the testing workers.
        """
        for i in range(self.num_processes):
            self.inputQs[i].put(('exit', None, None))


class TestingWorker(Process):
    def __init__(self,
                 actor,
                 critic,
                 episode_length,
                 nb_episodes,
                 action_repeat,
                 max_action,
                 gamma,
                 tau,
                 normalize_returns,
                 batch_size,
                 normalize_observations,
                 critic_l2_reg,
                 popart,
                 clip_norm,
                 reward_scale,
                 inputQ,
                 outputQ,
                 # environment wrapper parameters
                 full,
                 exclude_centering_frame,
                 integrator_accuracy,
                 max_env_traj,
                 visualize,
                 fail_reward):
        # Invoke parent constructor BEFORE doing anything!!
        Process.__init__(self)
        self.actor = actor
        self.critic = critic
        self.episode_length = episode_length
        self.nb_episodes = nb_episodes
        self.action_repeat = action_repeat
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.normalize_returns = normalize_returns
        self.batch_size = batch_size
        self.normalize_observations = normalize_observations
        self.critic_l2_reg = critic_l2_reg
        self.popart = popart
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale
        self.inputQ = inputQ
        self.full = full
        self.exclude_centering_frame = exclude_centering_frame
        self.visualize = visualize
        self.fail_reward = fail_reward
        self.integrator_accuracy = integrator_accuracy
        self.outputQ = outputQ
        self.max_env_traj = max_env_traj

    def run(self):
        """Override Process.run()"""
        # Create environment
        env = create_environment(action_repeat=self.action_repeat,
                                 full=self.full,
                                 exclude_centering_frame=self.exclude_centering_frame,
                                 visualize=self.visualize,
                                 fail_reward=self.fail_reward,
                                 integrator_accuracy=self.integrator_accuracy)
        nb_actions = env.action_space.shape[-1]

        env.seed(os.getpid())
        set_global_seeds(os.getpid())

        num_traj = 0

        # Allocate ReplayBuffer
        memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                        observation_shape=env.observation_space.shape)

        # Create DPPG agent
        agent = DDPG(self.actor, self.critic, memory, env.observation_space.shape,
                     env.action_space.shape, gamma=self.gamma, tau=self.tau,
                     normalize_returns=self.normalize_returns,
                     normalize_observations=self.normalize_observations,
                     batch_size=self.batch_size, action_noise=None,
                     param_noise=None, critic_l2_reg=self.critic_l2_reg,
                     enable_popart=self.popart, clip_norm=self.clip_norm,
                     reward_scale=self.reward_scale)

        # Build the testing logic fn
        testing_fn = make_testing_fn(
            agent, env, self.episode_length, self.action_repeat, self.max_action, self.nb_episodes)

        # Start TF session
        with U.single_threaded_session() as sess:
            agent.initialize(sess)
            set_parameters = U.SetFromFlat(self.actor.trainable_vars)

            # Start sampling-worker loop.
            while True:
                message, actor_ws, global_step = self.inputQ.get()  # Pop message
                if message == 'test':
                    # Set weights
                    set_parameters(actor_ws)
                    # Do testing
                    rewards, step_times, distances, episode_lengths = testing_fn()
                    self.outputQ.put(
                        (rewards, step_times, distances, episode_lengths, global_step))

                    # update number of trajectories
                    num_traj += self.nb_episodes

                    # restore environment if needed
                    if num_traj >= self.max_env_traj:
                        env.restore()
                        num_traj = 0

                elif message == 'exit':
                    print('[Worker {}] Exiting...'.format(os.getpid()))
                    env.close()
                    break


def make_testing_fn(agent, env, episode_length, action_repeat, max_action, nb_episodes):
    # Define the closure
    def sampling_fn():
        # Sampling logic
        agent.reset()
        obs = env.reset()
        rewards = []
        step_times = []
        distances = []
        episode_lengths = []
        obs = env.reset()
        for n in range(nb_episodes):
            reward_i = 0
            for t in range(episode_length):
                # Select action a_t without noise
                a_t, _ = agent.pi(
                    obs, apply_param_noise=False, apply_action_noise=False, compute_Q=False)
                assert a_t.shape == env.action_space.shape

                # Execute action a_t and observe reward r_t and next state s_{t+1}
                start_step_time = time.time()
                obs, r_t, eval_done, _ = env.step(max_action * a_t)
                end_step_time = time.time()
                step_time = end_step_time - start_step_time
                reward_i += r_t
                step_times.append(step_time)

                if eval_done:
                    print("[Worker {}] Testing Episode done!".format(os.getpid()))
                    episode_lengths.append(t)
                    distances.append(env.get_distance())
                    rewards.append(reward_i)
                    obs = env.reset()
                    break
        print("[Worker {}] Evaluation done".format(os.getpid()))

        return (rewards, step_times, distances, episode_lengths)
    return sampling_fn
