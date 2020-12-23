import random
import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("FrozenLake-v0")
# random.seed(0)
# np.random.seed(0)
# env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
# Sarsa


class Sarsa:
    def __init__(self, no_episodes, mean_episodes, epsilons,lr, gamma ) -> None:
        self.no_episodes = no_episodes
        self.mean_episodes = mean_episodes
        self.epsilons = epsilons
        self.gamma = gamma
        self.lr = lr
    def play_episode(self, q_values, epsilon):
        state = env.reset()
        action = self.choose_action(q_values, state, epsilon)
        done = False
        r_s = []
        while not done:
            state2, reward, done, info = env.step(action)

            action2 = self.choose_action(q_values, state, epsilon)
            # update Q(S,A)+learningRate(R+gamma Q(S´, A´) - Q(S,A)))
            self.updateQ(state, state2, reward, action, action2, q_values)

            state = state2
            action = action2
            r_s.append(reward)

        return r_s


    def updateQ(self, state, state2, reward, action, action2, q_values):
        predict = q_values[state, action]
        target = reward + self.gamma * q_values[state2, action2]
        q_values[state, action] = q_values[state, action] + self.lr * (target - predict)


    def init_q(self):
        return np.zeros((env.observation_space.n, env.action_space.n))


    def choose_action(self, q_values, state, epsilon):
        if random.random() > epsilon:
            return np.argmax(q_values[state, :])
        else:
            return env.action_space.sample()

    def play(self):
        epsilons_array = {}
        
        for e in self.epsilons:
            epsilons_array[e] = None

        for epsi in self.epsilons:
            total_reward = []
            for m in range(0, self.mean_episodes):
                rewards = []
                q_values = self.init_q()
                for i in range(0, self.no_episodes):
                    r = self.play_episode(q_values, epsilon=epsi)
                    rewards.append(sum(r))
                total_reward.append(np.cumsum(rewards))
            total_reward = np.array(total_reward)

            total_reward = total_reward.T
            epsilons_array[epsi] = total_reward
        return epsilons_array
