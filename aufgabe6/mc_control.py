import random
import gym
import matplotlib.pyplot as plt
import numpy as np
#https://towardsdatascience.com/mc-control-methods-50c018271553
env = gym.make("FrozenLake-v0")
random.seed(0)
np.random.seed(0)
env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()


def init_q():
    q_values = {}
    for s in range(0, env.observation_space.n):
        for a in range(0, env.action_space.n):
            q_values[(s, a)] = 0
    return q_values


def play_episode(q_values, epsilon):
    state = env.reset()
    done = False
    r_s = []
    s_a = []
    while not done:
        action = choose_action(q_values, state, epsilon)

        s_a.append((state, action))
        state, reward, done, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s


def choose_action(q_values, state, epsilon):
    if random.random() > epsilon:
        relevant_qs = [q_values[(state, a)] for a in range(0, env.action_space.n)]
        # there can be more than one best action
        best_actions_indexes = [i for i, v in enumerate(relevant_qs) if v == max(relevant_qs)]
        # in this case randomly choose on of them
        return random.choice(best_actions_indexes)
    else:
        return random.randint(0, 3)


def main():
    no_episodes = 1000
    epsilons = [0.01, 0.1, 0.5, 1.0]

    plot_data = []
    for e in epsilons:
        rewards = []
        q_values = init_q()
        for i in range(0, no_episodes):
            s, r = play_episode(q_values, epsilon=e)
            rewards.append(sum(r))

            # update q-values
            for i2, q in enumerate(s):
                return_i = sum(r[i2:])
                q_values[q] += 0.3 * (return_i - q_values[q])

        plot_data.append(np.cumsum(rewards))

    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Sum of rewards")
    for i, eps in enumerate(epsilons):
        plt.plot(range(0, no_episodes), plot_data[i], label="e=" + str(eps))
    plt.legend()
    plt.show()


main()
