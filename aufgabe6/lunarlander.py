import gym
import numpy as np
env = gym.make('LunarLander-v2')
rewards = []
for i_episode in range(2000):
    observation = env.reset()
    for t in range(100):
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            print(f"Max Reward {np.max(np.array(rewards))} timesteps")
            break
env.close()