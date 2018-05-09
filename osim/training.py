import tensorflow as tf
from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

class EvaluationStatistics:
    def __init__(self, tf_session, tf_writer):
        self._build_tf_graph()
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
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

    def _build_tf_graph(self):
        self.tf_rewards = tf.placeholder(tf.float32, (None,), name="rewards")
        self.tf_mean = tf.reduce_mean(self.tf_rewards, axis=0)

        self.tf_actor_loss = tf.placeholder(tf.float32, shape=(), name="actor_loss")
        self.tf_critic_loss = tf.placeholder(tf.float32, shape=(), name="critic_loss")
        
        self.summary_reward = tf.summary.scalar("Average reward", self.tf_mean)
        self.summary_al = tf.summary.scalar("Actor loss", self.tf_actor_loss)
        self.summary_cl = tf.summary.scalar("Critic loss", self.tf_critic_loss)

def train(env, nb_epochs, nb_episodes, episode_length, nb_train_steps, eval_freq, nb_eval_episodes, actor,
          critic, memory, gamma, normalize_returns, normalize_observations,
          critic_l2_reg, actor_lr, critic_lr, action_noise, popart, clip_norm,
          batch_size, reward_scale, tau=0.01):
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
    # Initialize DDPG agent (target network and replay buffer)
    agent = DDPG(actor, critic, memory, env.observation_space.shape,
                 env.action_space.shape, gamma=gamma, tau=tau,
                 normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise,
                 param_noise=None, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart,
                 clip_norm=clip_norm, reward_scale=reward_scale)

    # We need max_action because the NN output layer is a tanh.
    # So we must scale it back.
    max_action = env.action_space.high

    with U.single_threaded_session() as sess:
        agent.initialize(sess)
        
        # Setup summary writer
        writer = _setup_tf_summary()
        writer.add_graph(sess.graph)
        
        stats = EvaluationStatistics(tf_session=sess, tf_writer=writer)
        sess.graph.finalize()
        
        global_step = 0
        obs = env.reset()
        agent.reset()
        for epoch in range(nb_epochs):
            for episode in range(nb_episodes):
                obs = env.reset()
                # Generate a trajectory
                for t in range(episode_length):
                    # Select action a_t according to current policy and
                    # exploration noise
                    a_t, _ = agent.pi(obs, apply_noise=True, compute_Q=False)
                    assert a_t.shape == env.action_space.shape

                    # Execute action a_t and observe reward r_t and next state s_{t+1}
                    new_obs, r_t, done, info = env.step(max_action * a_t)

                    # Store transition in the replay buffer
                    agent.store_transition(obs, a_t, r_t, new_obs, done)
                    obs = new_obs

                    if done:
                        agent.reset()
                        obs = env.reset()
                        break  # End episode

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
                            a_t, _ = agent.pi(obs, apply_noise=False, compute_Q=False)
                            assert a_t.shape == env.action_space.shape

                            # Execute action a_t and observe reward r_t and next state s_{t+1}
                            obs, r_t, eval_done, info = env.step(max_action * a_t)
                            stats.add_reward(r_t)

                            if eval_done:
                                print("  Episode done!")
                                obs = env.reset()
                                break

                    # Plot average reward
                    stats.plot_reward(global_step)
                        
def _setup_tf_summary():
    import datetime

    now = datetime.datetime.now()
    logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    writer = tf.summary.FileWriter(logdir)
    return writer
