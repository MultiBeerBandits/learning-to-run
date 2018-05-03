from osim.env.osim import L2RunEnv

env = L2RunEnv(visualize=True)
observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
