from baselines.ddpg.models import Model
import tensorflow as tf

# The net output is the action vector
class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    # obs is a TF placeholder with shape = env.observation shape
    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:

            if reuse:
                scope.reuse_variables()
            x = obs
            x = tf.layers.dense(x, 64)
            x = tf.nn.elu(x)

            x = tf.layers.dense(x, 64)
            x = tf.nn.elu(x)

            x = tf.layers.dense(x, self.nb_actions)

            # The DDPG algorithm multiplies this by the MAX_ACTION
            x = tf.nn.tanh(x)
        return x

# The net output is Q(s, a)
class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            if reuse:
                scope.reuse_variables()
            print("obs shape : {}, action shape: {}".format(obs.shape, action.shape))
            x = tf.concat([obs, action], axis=-1)
            x = tf.layers.dense(x, 64)
            x = tf.nn.tanh(x)

            x = tf.layers.dense(x, 32)
            x = tf.nn.tanh(x)

            x = tf.layers.dense(x, 1)
            # We do not introduce any non-linear transformation we want
            # any number as reward.
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
