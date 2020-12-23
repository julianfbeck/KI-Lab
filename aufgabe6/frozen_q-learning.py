import random
import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("FrozenLake-v0")
random.seed(0)
np.random.seed(0)
env.seed(0)
lr = 0.1
gamma = 0.9

print("## Frozen Lake ##")
print("Start state:")
env.render()
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


def play_episode(q_values, epsilon):
    state = env.reset()
    done = False
    r_s = []
    while not done:
        #choose action action
        action = choose_action(q_values, state, epsilon)
        #take action
        state2, reward, done, _ = env.step(action)
        #update Q(S,A)+learningRate(R+gamma maxQ(S´, a) - Q(S,A)))
        updateQ(state, state2, reward, action, q_values)
        state = state2
        r_s.append(reward)
    return r_s

def updateQ(state, state2, reward, action, q_values):
    predict = q_values[state, action]
    target = reward + gamma * np.max(q_values[state2, :])
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