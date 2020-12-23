import random
import gym
import matplotlib.pyplot as plt
import numpy as np
# https://towardsdatascience.com/mc-control-methods-50c018271553
env = gym.make("FrozenLake-v0")
print("## Frozen Lake ##")
print("Start state:")


class MCControl:
    def __init__(self, no_episodes, mean_episodes, epsilons) -> None:
        self.no_episodes = no_episodes
        self.mean_episodes = mean_episodes
        self.epsilons = epsilons

    def init_q(self):
        q_values = {}
        for s in range(0, env.observation_space.n):
            for a in range(0, env.action_space.n):
                q_values[(s, a)] = 0
        return q_values

    def play_episode(self, q_values, epsilon):
        state = env.reset()
        done = False
        r_s = []
        s_a = []
        while not done:
            action = self.choose_action(q_values, state, epsilon)

            s_a.append((state, action))
            state, reward, done, _ = env.step(action)
            r_s.append(reward)
        return s_a, r_s

    def choose_action(self, q_values, state, epsilon):
        if random.random() > epsilon:
            relevant_qs = [q_values[(state, a)]
                           for a in range(0, env.action_space.n)]
            # there can be more than one best action
            best_actions_indexes = [i for i, v in enumerate(
                relevant_qs) if v == max(relevant_qs)]
            # in this case randomly choose on of them
            return random.choice(best_actions_indexes)
        else:
            return random.randint(0, 3)

    def play(self):
        # spielt Frozen lake mit verschiedenen Epsilons (Explorieren)
        # Monte Carlo Control
        # Nach jeder Episode updaten der Q-Values
        # Summe des Rewards
        # Monte Carlo control methods do not suffer from this bias,
        # as each update is made using a true sample of what Q(s,a) should be.
        # However, Monte Carlo methods can suffer from high variance,
        # which means more samples are required to achieve the same degree of learning compared to TD.
        # Monte Carlo Prediction (MC)
        # Any method which solves a problem by generating suitable random numbers,
        # and observing that fraction of numbers obeying some property or properties,
        # can be classified as a Monte Carlo method.”1
        # “Given a policy, create sample episodes following that policy
        # and estimate state and Q-values from the collected experiences.”
        # No knowledge about MDP needed (model-free approach).
        # The more episodes we sample, the closer the estimates of the state
        # values approach the real state values (law of large numbers).
        # Drawback: Can only be applied to episodic problems.
        # MC: New evidence based on observed return (at end of episode)

        epsilons_array = {}

        for e in self.epsilons:
            epsilons_array[e] = None

        for epsi in self.epsilons:
            total_reward = []
            for m in range(0, self.mean_episodes):
                rewards = []
                q_values = self.init_q()
                for i in range(0, self.no_episodes):
                    s, r = self.play_episode(q_values, epsilon=e)
                    # Calculate empirical return Gt in every state that was visited.
                    rewards.append(sum(r))
                    # Calculate V(s) for every state by aggregating the empirical returns received in s.
                    # update q-values
                    for i2, q in enumerate(s):
                        return_i = sum(r[i2:])
                        q_values[q] += 0.3 * (return_i - q_values[q])
                total_reward.append(np.cumsum(rewards))
            total_reward = np.array(total_reward)

            total_reward = total_reward.T
            epsilons_array[epsi] = total_reward
        return epsilons_array
