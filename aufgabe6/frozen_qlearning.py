import random
import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("FrozenLake-v0")
#Q-Learning
#“If one had to identify one idea as central and novel to reinforcement learning,
#it would undoubtedly be temporal differencelearning.“ - Sutton Book 
#• Temporal differencing (TD) learning is a combination of dynamic programming (DP) and Monte Carlo (MC) methods.
#• Like MC methods, TD methods can learn directly from experience without knowing the MDP structure.
#• Like DP, TD methods update estimates based partially on other learned estimates,
#  without waiting for a final outcome (“bootstrapping”).
# Use reward and estimated state value of next state to update current state value
#TD: New evidence based on current reward and state value of next state
#Q-learning is off-policy and uses:
#• TD-prediction for policy evaluation.
#• An -greedy policy as behaviour policy.
#• A greedy policy as target policy.
#• One can show that the learned Q-values directly approximate the optimal Q-values.

class QLearning:
    def __init__(self, no_episodes, mean_episodes, epsilons,lr, gamma ) -> None:
        self.no_episodes = no_episodes
        self.mean_episodes = mean_episodes
        self.epsilons = epsilons
        self.gamma = gamma
        self.lr = lr

    def play_episode(self,q_values, epsilon):
        state = env.reset()
        done = False
        r_s = []
        while not done:
            #choose action action
            action = self.choose_action(q_values, state, epsilon)
            #take action
            state2, reward, done, _ = env.step(action)
            #update Q(S,A)+learningRate(R+gamma maxQ(S´, a) - Q(S,A)))
            self.updateQ(state, state2, reward, action, q_values)
            state = state2
            r_s.append(reward)
        return r_s

    def updateQ(self,state, state2, reward, action, q_values):
        predict = q_values[state, action]
        target = reward + self.gamma * np.max(q_values[state2, :])
        q_values[state, action] = q_values[state, action] + self.lr * (target - predict)

    def init_q(self,):
        return np.zeros((env.observation_space.n, env.action_space.n))


    def choose_action(self,q_values, state, epsilon):
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