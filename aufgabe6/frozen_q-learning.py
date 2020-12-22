import random
import gym
import matplotlib.pyplot as plt
import numpy as np
env = gym.make("FrozenLake-v0")
random.seed(0)
np.random.seed(0)
env.seed(0)

print("## Frozen Lake ##")
print("Start state:")
env.render()

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