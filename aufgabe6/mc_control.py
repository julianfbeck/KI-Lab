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
    #spielt Frozen lake mit verschiedenen Epsilons (Explorieren)
    #Monte Carlo Control
    #Nach jeder Episode updaten der Q-Values
    #Summe des Rewards
    #Monte Carlo control methods do not suffer from this bias, 
    # as each update is made using a true sample of what Q(s,a) should be. 
    # However, Monte Carlo methods can suffer from high variance, 
    # which means more samples are required to achieve the same degree of learning compared to TD.
    #Monte Carlo Prediction (MC)
    #Any method which solves a problem by generating suitable random numbers,
    #and observing that fraction of numbers obeying some property or properties,
    #can be classified as a Monte Carlo method.”1
    #“Given a policy, create sample episodes following that policy
    # and estimate state and Q-values from the collected experiences.”
    #No knowledge about MDP needed (model-free approach).
    # The more episodes we sample, the closer the estimates of the state
    # values approach the real state values (law of large numbers).
    # Drawback: Can only be applied to episodic problems.
    #MC: New evidence based on observed return (at end of episode)
    no_episodes = 1000
    epsilons = [0.01, 0.1, 0.5, 1.0]

    plot_data = []
    for e in epsilons:
        rewards = []
        q_values = init_q()
        for i in range(0, no_episodes):
            #Sample episodes: Start from start state and follow policy π until terminal state n is reached. Repeat this times.
            s, r = play_episode(q_values, epsilon=e)
            #Calculate empirical return Gt in every state that was visited.
            rewards.append(sum(r))
            #Calculate V(s) for every state by aggregating the empirical returns received in s.
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
    #Plot der Summe von Rewards


main()
