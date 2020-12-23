import random
import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("FrozenLake-v0")
random.seed(0)
np.random.seed(0)
env.seed(0)
lr = 0.5
gamma = 0.9

print("## Frozen Lake ##")
print("Start state:")
env.render()
# Sarsa


def play_episode(q_values, epsilon):
    state = env.reset()
    action = choose_action(q_values, state, epsilon)
    done = False
    r_s = []
    while not done:
        state2, reward, done, info = env.step(action)

        action2 = choose_action(q_values, state, epsilon)
        #update Q(S,A)+learningRate(R+gamma Q(S´, A´) - Q(S,A)))
        updateQ(state, state2, reward, action, action2, q_values)

        state = state2
        action = action2
        r_s.append(reward)

    return r_s

def updateQ(state, state2, reward, action, action2, q_values):
	predict = q_values[state, action]
	target = reward + gamma * q_values[state2, action2]
	q_values[state, action] = q_values[state, action] + lr * (target - predict)

def init_q():
    return np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(q_values, state, epsilon):
    if random.random() > epsilon:
        return np.argmax(q_values[state, :])
    else:
        return env.action_space.sample()

def main():
    no_episodes = 1000
    epsilons = [0.01, 0.1, 0.5, 1.0]

    plot_data = []
    for e in epsilons:
        rewards = []
        q_values = init_q()
        for i in range(0, no_episodes):
            r = play_episode(q_values, epsilon=e)
            rewards.append(sum(r))

        plot_data.append(np.cumsum(rewards))

    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Sum of rewards")
    for i, eps in enumerate(epsilons):
        plt.plot(range(0, no_episodes), plot_data[i], label="e=" + str(eps))
    plt.legend()
    plt.show()


main()
