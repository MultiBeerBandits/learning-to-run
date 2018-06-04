import sys
sys.path.append("..")
from env_wrapper import create_environment
from replay_buffer import ReplayBufferFlip
import numpy as np
from numpy.testing import assert_almost_equal

"""
elements of the observation vector in order
names = ["pelvis_x", "pelvis_y"] 2
            names += [joint + "_" + var for (joint, var) in product(["hip_left","hip_right","knee_left","knee_right","ankle_left","ankle_right"],
                ["rz", "vrz"])] 12
            names += ["ground_pelvis_rot", "ground_pelvis_vel_rot"] 2
            names += [body_part + "_" + var for (body_part, var) in product(
                ["head", "torso", "toes_left", "toes_right", "talus_left", "talus_right"],
                ["x", "y"])] 12
            names += ["com_x", "com_y", "com_vel_x", "com_vel_y"] 4
            names += ["pelvis_vel_x", "pelvis_vel_y"] 2
"""

# test the base observation vector consistency
def obs_vector_consistency():
    env = create_environment(False, False, 1, 0, False)
    for _ in range(100):
        # take some random action
        env.step(env.action_space.sample())
        # check consistency between state desc and obs vector
        # plus the order of obs (used in action flip)
        desc = env.get_state_desc()
        obs = env.get_observation_basic()
        
        # check pelvis coordinates
        centering_x = desc['body_pos']['pelvis'][0]
        centering_y = desc['body_pos']['pelvis'][1]
        assert_almost_equal(centering_x, obs[0])
        assert_almost_equal(centering_y, obs[1])
        
        # check joint and speed
        joint_pos = desc['joint_pos']
        joint_vel = desc['joint_vel']
        # hips
        assert_almost_equal(joint_pos['hip_l'][0], obs[2])
        assert_almost_equal(joint_vel['hip_l'][0], obs[3])
        assert_almost_equal(joint_pos['hip_r'][0], obs[4])
        assert_almost_equal(joint_vel['hip_r'][0], obs[5])
        # knees
        assert_almost_equal(joint_pos['knee_l'][0], obs[6])
        assert_almost_equal(joint_vel['knee_l'][0], obs[7])
        assert_almost_equal(joint_pos['knee_r'][0], obs[8])
        assert_almost_equal(joint_vel['knee_r'][0], obs[9])
        # ankles
        assert_almost_equal(joint_pos['ankle_l'][0], obs[10])
        assert_almost_equal(joint_vel['ankle_l'][0], obs[11])
        assert_almost_equal(joint_pos['ankle_r'][0], obs[12])
        assert_almost_equal(joint_vel['ankle_r'][0], obs[13])
        # ground pelvis
        assert_almost_equal(joint_pos['ground_pelvis'][0], obs[14])
        assert_almost_equal(joint_vel['ground_pelvis'][0], obs[15])

        # check body part coordinates
        body_pos = desc['body_pos']
        # head
        assert_almost_equal(body_pos['head'][0], obs[16] + centering_x)
        assert_almost_equal(body_pos['head'][1], obs[17] + centering_y)
        # torso
        assert_almost_equal(body_pos['torso'][0], obs[18] + centering_x)
        assert_almost_equal(body_pos['torso'][1], obs[19] + centering_y)
        # toes 
        assert_almost_equal(body_pos['toes_l'][0], obs[20] + centering_x)
        assert_almost_equal(body_pos['toes_l'][1], obs[21] + centering_y)
        assert_almost_equal(body_pos['toes_r'][0], obs[22] + centering_x)
        assert_almost_equal(body_pos['toes_r'][1], obs[23] + centering_y)
        # talus
        assert_almost_equal(body_pos['talus_l'][0], obs[24] + centering_x)
        assert_almost_equal(body_pos['talus_l'][1], obs[25] + centering_y)
        assert_almost_equal(body_pos['talus_r'][0], obs[26] + centering_x)
        assert_almost_equal(body_pos['talus_r'][1], obs[27] + centering_y)

        # check center of mass
        com_pos = desc['misc']['mass_center_pos']
        com_vel = desc['misc']['mass_center_vel']
        assert_almost_equal(com_pos[0], obs[28] + centering_x)
        assert_almost_equal(com_pos[1], obs[29] + centering_y)
        assert_almost_equal(com_vel[0], obs[30])
        assert_almost_equal(com_vel[1], obs[31])

        # check pelvis speed
        assert_almost_equal(desc['body_vel']['pelvis'][0], obs[32])
        assert_almost_equal(desc['body_vel']['pelvis'][1], obs[33])


def test_state_flip():
    env = create_environment(False, False, 1, 0, False)
    b = ReplayBufferFlip(2, True, env.get_observation_names(),
                         env.action_space.shape,
                         env.observation_space.shape)
    env.reset()
    for _ in range(100):
        obs = env.step(env.action_space.sample())[0]
        fobs = b.swap_states(np.matrix(obs)).tolist()[0]
        assert(len(obs) == 34)
        assert(len(obs) == len(fobs))
        # pelvis does not change
        assert_almost_equal(obs[0:2], fobs[0:2])
        # hip
        assert_almost_equal(obs[2:4], fobs[4:6])
        assert_almost_equal(obs[4:6], fobs[2:4])
        # knee
        assert_almost_equal(obs[6:8], fobs[8:10])
        assert_almost_equal(obs[8:10], fobs[6:8])
        # ankle
        assert_almost_equal(obs[10:12], fobs[12:14])
        assert_almost_equal(obs[12:14], fobs[10:12])
        # up to torso nothing changes
        assert_almost_equal(obs[14:20], fobs[14:20])
        # toes
        assert_almost_equal(obs[20:22], fobs[22:24])
        assert_almost_equal(obs[22:24], fobs[20:22])
        # talus
        assert_almost_equal(obs[24:26], fobs[26:28])
        assert_almost_equal(obs[26:28], fobs[24:26])
        # center of mass does not change
        assert_almost_equal(obs[28:32], fobs[28:32])
        # pelvis speed does not change
        assert_almost_equal(obs[32:34], fobs[32:34])

# we discovered that ['body_pos']['pelvis'][0:2] == ['joint_pos']['ground_pelvis'][1:3]
# and ['body_vel']['pelvis'][0:2] == ['joint_vel']['ground_pelvis'][1:3]
def pay_attention_always_equal():
    env = create_environment(False, False, 1, 0, False)
    env.reset()
    for _ in range(100):
        obs = env.step(env.action_space.sample())[0]
        pelvis_xy = env.get_state_desc()['body_pos']['pelvis'][0:2]
        ground_pelvis_xy = env.get_state_desc()['joint_pos']['ground_pelvis'][1:3]
        assert_almost_equal(pelvis_xy, ground_pelvis_xy)
        pelvis_vel = env.get_state_desc()['body_vel']['pelvis'][0:2]
        ground_pelvis_vel = env.get_state_desc()['joint_vel']['ground_pelvis'][1:3]
        assert_almost_equal(pelvis_vel, ground_pelvis_vel)

if __name__ == '__main__':
    obs_vector_consistency()
    test_state_flip()   
    pay_attention_always_equal()