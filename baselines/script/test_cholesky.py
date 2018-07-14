#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:20:29 2018

@author: matteo
"""

from baselines.common.distributions import CholeskyGaussianPd as CG
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import baselines.common.tf_util as U
sess = U.single_threaded_session()
sess.__enter__()

params = tf.constant(np.array([0.,0., 0., 1., 0.], dtype=np.float32))
pd = CG(params)
_x = [pd.sample() for _ in range(100)]
std = pd.std
cov = tf.matmul(std, tf.matrix_transpose(std))
print('STD', std.eval(session=sess))
print('COV', cov.eval(session=sess))
x_in = tf.placeholder(name='x_in', dtype=tf.float32, shape=[None, 2])
_logp = -pd.neglogp(x_in)

params2 = tf.constant(np.array([1.,1., 0., 2., 0.], dtype=np.float32))
pd2 = CG(params2)

print('Size', pd.size)

print('KL', sess.run(pd.kl(pd2)))

print('log|SIGMA|', sess.run(pd.log_det_cov))

print('ENT', sess.run(pd.entropy()))

def sample():
    return sess.run(_x)
        
def logp(x):
    return sess.run(_logp, feed_dict={x_in: x})


print('LOGPs', logp([[0,0], 
            [1,1],
            [1,-1]]))

print('RENYI', sess.run(pd.renyi(pd)))
    
data = np.array(sample())
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data[:, 0], data[:, 1])
ax.set_aspect('equal', 'box')
plt.show()