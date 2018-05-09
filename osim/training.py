import tensorflow as tf
from baselines.ddpg.ddpg import DDPG
from baselines import logger
import baselines.common.tf_util as U
import numpy as np
import time
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from multiprocessing.pool import Pool
from osim.env.osim import L2RunEnv
from baselines.ddpg.memory import Memory


class EvaluationStatistics:

    def __init__(self, tf_session, tf_writer):
        self._build_tf_graph()
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.distances = []
        self.step_times = []
        self.tf_session = tf_session
        self.tf_writer = tf_writer

    def add_reward(self, reward):
        self.rewards.append(reward)

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

    def add_distance(self, env):
        """
        take the whole env and store interesting quantities
        """
        state_desc = env.get_state_desc()
        # x of the pelvis
        distance = state_desc["body_pos"]["pelvis"][0]
        self.distances.append(distance)

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

    def fill_stats(self, combined_stats):
        combined_stats['distances_mean'] = np.mean(self.distances)
        combined_stats['distances_var'] = np.var(self.distances)
        combined_stats['rewards_mean'] = np.mean(self.rewards)
        combined_stats['step_time'] = np.mean(self.step_times)
        self.step_times.clear()

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


def train(env, nb_epochs, nb_episodes, episode_length, nb_train_steps, eval_freq, nb_eval_episodes, actor,
          critic, memory, gamma, normalize_returns, normalize_observations,
          critic_l2_reg, actor_lr, critic_lr, action_noise, popart, clip_norm,
          batch_size, reward_scale, action_repeat, num_processes, tau=0.01):
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

    # Initialize DDPG agent (target network and replay buffer)
    agent = DDPG(actor, critic, memory, env.observation_space.shape,
                 env.action_space.shape, gamma=gamma, tau=tau,
                 normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise,
                 param_noise=None, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart,
                 clip_norm=clip_norm, reward_scale=reward_scale)

    # Start training loop
    with U.single_threaded_session() as sess:
        agent.initialize(sess)

        # We need max_action because the NN output layer is a tanh.
        # So we must scale it back.
        max_action = env.action_space.high

        # Build sampling workers args
        mp_sampling_args = [MPSamplingArgs(actor,
                                           critic,
                                           episode_length,
                                           action_repeat,
                                           max_action,
                                           None,
                                           gamma,
                                           tau,
                                           normalize_returns,
                                           batch_size,
                                           normalize_observations,
                                           critic_l2_reg,
                                           actor_lr,
                                           critic_lr,
                                           popart,
                                           clip_norm,
                                           reward_scale)
                            for i in range(num_processes)]

    # Setup summary writer
        writer = _setup_tf_summary()
        writer.add_graph(sess.graph)

        stats = EvaluationStatistics(tf_session=sess, tf_writer=writer)
        #sess.graph.finalize()

        global_step = 0
        obs = env.reset()
        agent.reset()
        for epoch in range(nb_epochs):
            for episode in range(nb_episodes):
                # Dispatch transitions sampling job to workers
                with Pool(processes=num_processes) as pool:
                    mp_transitions = pool.map(mp_sampling, mp_sampling_args)
                    for transitions in mp_transitions:
                        for t in transitions:
                            # t is a tuple (s_0, a_0, r_0, s_1, is_done)
                            agent.store_transition(*t)

                # Training phase
                for t_train in range(nb_train_steps):
                    critic_loss, actor_loss = agent.train()
                    agent.update_target_net()

                    # Plot statistics
                    stats.add_critic_loss(critic_loss, global_step)
                    stats.add_actor_loss(actor_loss, global_step)
                    global_step += 1

                # Evaluation phase
                if episode % eval_freq == 0:
                    # Generate evaluation trajectories
                    for eval_episode in range(nb_eval_episodes):
                        print("Evaluating episode {}...".format(eval_episode))
                        obs = env.reset()
                        for t in range(episode_length):
                            env.render()

                            # Select action a_t according to current policy and
                            # exploration noise
                            a_t, _ = agent.pi(
                                obs, apply_noise=False, compute_Q=False)
                            assert a_t.shape == env.action_space.shape

                            # Execute action a_t and observe reward r_t and next state s_{t+1}
                            start_step_time = time.time()
                            obs, r_t, eval_done, info = env.step(
                                max_action * a_t)
                            end_step_time = time.time()
                            step_time = end_step_time - start_step_time
                            stats.add_reward(r_t)
                            stats.add_distance(env)
                            stats.add_step_time(step_time)

                            if eval_done:
                                print("  Episode done!")
                                obs = env.reset()
                                break

                    combined_stats = agent.get_stats().copy()
                    stats.fill_stats(combined_stats)
                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    logger.dump_tabular()
                    logger.info('')
                    # Plot average reward
                    stats.plot_reward(global_step)
                    stats.plot_distance(global_step)


class MPSamplingArgs:
    def __init__(self, actor, critic, episode_length, action_repeat, max_action, session,
                 gamma,
                 tau,
                 normalize_returns,
                 batch_size,
                 normalize_observations,
                 critic_l2_reg,
                 actor_lr,
                 critic_lr,
                 popart,
                 clip_norm,
                 reward_scale):
        self.actor = actor
        self.critic = critic
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.max_action = max_action
        self.session = session
        self.gamma = gamma
        self.tau = tau
        self.normalize_returns = normalize_returns
        self.batch_size = batch_size
        self.normalize_observations = normalize_observations
        self.critic_l2_reg = critic_l2_reg
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.popart = popart
        self.clip_norm = clip_norm
        self.reward_scale = reward_scale


def mp_sampling(args):
    # Unpack
    actor = args.actor
    critic = args.critic
    sess = args.session
    episode_length = args.episode_length
    action_repeat = args.action_repeat
    max_action = args.max_action
    gamma = args.gamma
    tau = args.tau
    normalize_returns = args.normalize_returns
    batch_size = args.batch_size
    normalize_observations = args.normalize_observations
    critic_l2_reg = args.critic_l2_reg
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    popart = args.popart
    clip_norm = args.clip_norm
    reward_scale = args.reward_scale

    # Instantiate num_processes OU noise
    env = L2RunEnv(visualize=False)
    nb_actions = env.action_space.shape[-1]
    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                         sigma=np.ones(nb_actions))
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape)
    agent = DDPG(actor, critic, memory, env.observation_space.shape,
                 env.action_space.shape, gamma=gamma, tau=tau,
                 normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=noise,
                 param_noise=None, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart,
                 clip_norm=clip_norm, reward_scale=reward_scale)
    agent.initialize(sess)

    # Sampling logic
    agent.reset()
    obs = env.reset()
    transitions = []
    for t in range(episode_length):
        # Select action a_t according to current policy and
        # exploration noise

        a_t, _ = agent.pi(obs, apply_noise=True, compute_Q=False)
        assert a_t.shape == env.action_space.shape

        # the action has been chosen, need to repeat it for action_repeat
        # times
        for _ in range(action_repeat):
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


def _setup_tf_summary():
    import datetime

    now = datetime.datetime.now()
    logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    writer = tf.summary.FileWriter(logdir)
    return writer
