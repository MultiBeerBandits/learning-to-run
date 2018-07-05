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
import queue
from env_wrapper import create_environment
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
import os
import os.path
import time
from round_robin_tester import RoundRobinTester


class EvaluationStatistics:

    def __init__(self, tf_session, tf_writer):
        self._build_tf_graph()
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.distances = []
        self.step_times = []
        self.episode_lengths = []
        self.tf_session = tf_session
        self.tf_writer = tf_writer

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_rewards(self, rewards):
        self.rewards = rewards

    def plot_reward(self, global_step):
        summary = self.tf_session.run(self.summary_reward, feed_dict={
                                      self.tf_rewards: self.rewards})
        self.tf_writer.add_summary(summary, global_step)
        self.rewards.clear()

    def add_actor_loss(self, actor_loss, global_step):
        self.actor_losses.append(actor_loss)
        summary = self.tf_session.run(self.summary_al, feed_dict={
                                      self.tf_actor_loss: actor_loss})
        self.tf_writer.add_summary(summary, global_step)

    def add_critic_loss(self, critic_loss, global_step):
        self.critic_losses.append(critic_loss)
        summary = self.tf_session.run(self.summary_cl, feed_dict={
                                      self.tf_critic_loss: critic_loss})
        self.tf_writer.add_summary(summary, global_step)

    def add_distance(self, distance):
        """
        Stores the distance
        """
        self.distances.append(distance)

    def add_distances(self, distances):
        """
        Stores the distance
        """
        self.distances = distances

    def plot_distance(self, global_step):
        summary_mean = self.tf_session.run(self.summary_dist_mean, feed_dict={
                                           self.tf_distance: self.distances})
        summary_var = self.tf_session.run(self.summary_dist_var, feed_dict={
                                          self.tf_distance: self.distances})
        self.tf_writer.add_summary(summary_mean, global_step)
        self.tf_writer.add_summary(summary_var, global_step)
        self.distances.clear()

    def add_step_time(self, step_time):
        self.step_times.append(step_time)

    def add_step_times(self, step_times):
        self.step_times = step_times

    def add_episode_length(self, episode_length):
        self.episode_lengths.append(episode_length)

    def add_episode_lengths(self, episode_lengths):
        self.episode_lengths = episode_lengths

    def fill_stats(self, combined_stats):
        combined_stats['distances_mean'] = np.mean(self.distances)
        combined_stats['distances_var'] = np.var(self.distances)
        combined_stats['rewards_mean'] = np.mean(self.rewards)
        combined_stats['step_time'] = np.mean(self.step_times)
        combined_stats['episode_length_mean'] = np.mean(self.episode_lengths)
        self.step_times.clear()
        self.episode_lengths.clear()

    def _build_tf_graph(self):
        self.tf_rewards = tf.placeholder(tf.float32, (None,), name="rewards")
        self.tf_rewards_mean = tf.reduce_mean(self.tf_rewards, axis=0)

        self.tf_actor_loss = tf.placeholder(tf.float32, shape=(),
                                            name="actor_loss")
        self.tf_critic_loss = tf.placeholder(tf.float32, shape=(),
                                             name="critic_loss")
        self.tf_distance = tf.placeholder(tf.float32, (None,),
                                          name="distances")
        self.tf_distance_mean, self.tf_distance_var = tf.nn.moments(
            self.tf_distance, axes=[0], name="distance_moments")

        self.summary_dist_mean = tf.summary.scalar(
            "Avreage reached distance", self.tf_distance_mean, family="distance")
        self.summary_dist_var = tf.summary.scalar(
            "Variance reached distance distance", self.tf_distance_var, family="distance")
        self.summary_reward = tf.summary.scalar(
            "Average reward", self.tf_rewards_mean)
        self.summary_al = tf.summary.scalar(
            "Actor loss", self.tf_actor_loss, family="losses")
        self.summary_cl = tf.summary.scalar(
            "Critic loss", self.tf_critic_loss, family="losses")


def train(env, nb_epochs, nb_episodes, nb_epoch_cycles, episode_length, nb_train_steps,
          eval_freq, save_freq, nb_eval_episodes, actor,
          critic, memory, gamma, normalize_returns, normalize_observations,
          critic_l2_reg, action_noise, param_noise, popart, clip_norm,
          batch_size, reward_scale, action_repeat, full, exclude_centering_frame,
          visualize, fail_reward, num_processes, num_processes_to_wait, num_testing_processes,
          learning_session, min_buffer_length, integrator_accuracy=5e-5, max_env_traj=100, tau=0.01):
    """
    Parameters
    ----------
    nb_epochs : the number of epochs to train.

    nb_episodes : the number of episodes for each epoch.

    episode_length : the maximum number of steps for each episode.

    gamma : discount factor.

    tau : soft update coefficient.

    clip_norm : clip on the norm of the gradient.
    """

    assert action_repeat > 0
    assert nb_episodes >= num_processes

    # Get params from learning session
    checkpoint_dir = learning_session.checkpoint_dir
    log_dir = learning_session.log_dir
    training_step = learning_session.last_training_step

    # Initialize DDPG agent (target network and replay buffer)
    agent = DDPG(actor, critic, memory, env.observation_space.shape,
                 env.action_space.shape, gamma=gamma, tau=tau,
                 normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise,
                 param_noise=None, critic_l2_reg=critic_l2_reg,
                 enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale, training_step=training_step)

    # We need max_action because the NN output layer is a tanh.
    # So we must scale it back.
    max_action = env.action_space.high

    # Build Workers
    events = [Event() for _ in range(num_processes)]
    inputQs = [Queue() for _ in range(num_processes)]
    outputQ = Queue()
    # Split work among workers
    nb_episodes_per_worker = nb_episodes // num_processes

    workers = [SamplingWorker(i,
                              actor,
                              critic,
                              episode_length,
                              nb_episodes_per_worker,
                              action_repeat,
                              max_action,
                              gamma,
                              tau,
                              normalize_returns,
                              batch_size,
                              normalize_observations,
                              param_noise,
                              critic_l2_reg,
                              popart,
                              clip_norm,
                              reward_scale,
                              events[i],
                              inputQs[i],
                              outputQ,
                              full,
                              exclude_centering_frame,
                              integrator_accuracy,
                              max_env_traj,
                              visualize,
                              fail_reward) for i in range(num_processes)]

    # Run the Workers
    for w in workers:
        w.start()

    # Create Round Robin tester
    tester = RoundRobinTester(num_testing_processes,
                              actor,
                              critic,
                              episode_length,
                              nb_eval_episodes,
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
                              full,
                              exclude_centering_frame,
                              integrator_accuracy,
                              max_env_traj,
                              visualize,
                              fail_reward)

    # Start training loop
    with U.single_threaded_session() as sess:
        agent.initialize(sess)

        writer = tf.summary.FileWriter(log_dir)
        writer.add_graph(sess.graph)

        # Initialize writer and statistics
        stats = EvaluationStatistics(tf_session=sess, tf_writer=writer)

        # setup saver
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

        get_parameters = U.GetFlat(actor.trainable_vars)

        global_step = 0
        obs = env.reset()
        agent.reset()

        # Processes waiting for a new sampling task
        waiting_indices = [i for i in range(num_processes)]
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # If we have sampling workers waiting, dispatch a sampling job
                if waiting_indices:
                    actor_ws = get_parameters()
                    # Run parallel sampling
                    for i in waiting_indices:
                        inputQs[i].put(('sample', actor_ws))
                        events[i].set()  # Notify worker: sample baby, sample!
                    waiting_indices.clear()

                # Collect results when ready
                for i in range(num_processes_to_wait):
                    process_index, transitions = outputQ.get()
                    waiting_indices.append(process_index)
                    print(
                        'Collecting transition samples from Worker {}/{}'.format(i+1, num_processes_to_wait))
                    for t in transitions:
                        agent.store_transition(*t)

                # try to collect other samples if available
                for i in range(num_processes):
                    try:
                        process_index, transitions = outputQ.get_nowait()
                        if process_index not in waiting_indices:
                            waiting_indices.append(process_index)
                        print('Collecting transition samples from Worker {}'.format(
                            process_index))
                        for t in transitions:
                            agent.store_transition(*t)
                    except queue.Empty:
                        # No sampling ready, keep on training.
                        pass

                # Training phase
                if agent.memory.nb_entries > min_buffer_length:
                    for _ in range(nb_train_steps):
                        critic_loss, actor_loss = agent.train()
                        agent.update_target_net()

                        # Plot statistics
                        stats.add_critic_loss(critic_loss, global_step)
                        stats.add_actor_loss(actor_loss, global_step)
                        global_step += 1

                    # Evaluation phase
                    if cycle % eval_freq == 0:
                        print("Cycle number: ", cycle+epoch*nb_epoch_cycles)
                        print("Sending testing job...")
                        actor_ws = get_parameters()

                        # Send a testing job
                        tester.test(actor_ws, global_step)

                        # Print stats (if any)
                        tester.log_stats(stats, logger)

                    if cycle % save_freq == 0:
                        # Save weights
                        save_path = saver.save(sess, checkpoint_dir)
                        print("Model saved in path: %s" % save_path)
                        # Dump learning session
                        learning_session.dump(agent.training_step)
                        print("Learning session dumped to: %s" %
                              str(learning_session.session_path))
                else:
                    print("Not enough entry in memory buffer")

        # Stop workers
        for i in range(num_processes):
            inputQs[i].put(('exit', None))
            events[i].set()  # Notify worker: exit!
        tester.close()  # Stop testing workers
        env.close()


class SamplingWorker(Process):
    def __init__(self,
                 process_index,
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
                 param_noise,
                 critic_l2_reg,
                 popart,
                 clip_norm,
                 reward_scale,
                 event,
                 inputQ,
                 outputQ,
                 # environment wrapper parameters
                 full,
                 exclude_centering_frame,
                 integrator_accuracy,
                 max_env_traj,
                 visualize,
                 fail_reward,
                 action_noise_prob=0.7):
        # Invoke parent constructor BEFORE doing anything!!
        Process.__init__(self)
        self.process_index = process_index
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
        self.param_noise = param_noise
        self.critic_l2_reg = critic_l2_reg
        self.popart = popart
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale
        self.event = event
        self.inputQ = inputQ
        self.outputQ = outputQ
        self.full = full
        self.exclude_centering_frame = exclude_centering_frame
        self.visualize = visualize
        self.fail_reward = fail_reward
        self.action_noise_prob = action_noise_prob
        self.integrator_accuracy = integrator_accuracy
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

        # keep tracks of the number of trajectory done
        num_traj = 0

        env.seed(os.getpid())
        set_global_seeds(os.getpid())

        # Create OU Noise
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                    sigma=0.2,
                                                    theta=0.1)

        # Allocate ReplayBuffer
        memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                        observation_shape=env.observation_space.shape)

        # Create DPPG agent
        agent = DDPG(self.actor, self.critic, memory, env.observation_space.shape,
                     env.action_space.shape, gamma=self.gamma, tau=self.tau,
                     normalize_returns=self.normalize_returns,
                     normalize_observations=self.normalize_observations,
                     batch_size=self.batch_size, action_noise=action_noise,
                     param_noise=self.param_noise, critic_l2_reg=self.critic_l2_reg,
                     enable_popart=self.popart, clip_norm=self.clip_norm,
                     reward_scale=self.reward_scale)

        # Build the sampling logic fn
        sampling_fn = make_sampling_fn(
            agent, env, self.episode_length, self.action_repeat, self.max_action, self.nb_episodes, self.action_noise_prob)

        # Start TF session
        with U.single_threaded_session() as sess:
            agent.initialize(sess)
            set_parameters = U.SetFromFlat(self.actor.trainable_vars)
            # Start sampling-worker loop.
            while True:
                # self.event.wait()  # Wait for a new message
                # self.event.clear()  # Upon message receipt, mark as read
                message, actor_ws = self.inputQ.get()  # Pop message
                if message == 'sample':
                    # Set weights
                    set_parameters(actor_ws)
                    # Do sampling
                    transitions = sampling_fn()
                    self.outputQ.put((self.process_index, transitions))

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


def make_sampling_fn(agent, env, episode_length, action_repeat, max_action, nb_episodes, action_noise_prob):
    # Define the closure
    def sampling_fn():
        # Sampling logic
        agent.reset()
        obs = env.reset()
        transitions = []
        for n in range(nb_episodes):
            # draw a coin for selecting between param noise and action noise
            apply_action_noise = np.random.uniform() < action_noise_prob
            apply_param_noise = not apply_action_noise
            if apply_param_noise:
                agent.update_param_noise_stddev()
            for t in range(episode_length):
                # Select action a_t according to current policy and
                # exploration noise
                a_t, _ = agent.pi(obs, apply_action_noise=apply_action_noise,
                                  apply_param_noise=apply_param_noise, compute_Q=False)
                assert a_t.shape == env.action_space.shape

                # the action has been chosen, need to repeat it for action_repeat
                # times
                # Skip frames implemented in the environment wrapper
                # Execute action a_t and observe reward r_t and next state s_{t+1}
                new_obs, r_t, done, _ = env.step(max_action * a_t)

                # Store transition in the replay buffer
                transitions.append((obs, a_t, r_t, new_obs, done))
                obs = new_obs

                if done:
                    agent.reset()
                    obs = env.reset()
                    break  # End episode

        return transitions
    return sampling_fn


def _setup_tf_summary():
    import datetime

    now = datetime.datetime.now()
    log_dir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    writer = tf.summary.FileWriter(log_dir)
    return writer
