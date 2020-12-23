from frozen_sarsa import Sarsa
from frozen_qlearning import QLearning
from mc_control import MCControl

import matplotlib.pyplot as plt
import numpy as np


epsilons = [0.01, 0.1, 0.5, 1.0]
no_episodes = 1000
mean_episodes = 20
lr = 0.5
gamma = 0.9
stats = {1:[0.1, 0.5,], 2:[0.01, 20, 0.5, 1.0], 3:[0.01, 0.1, 0.5, 1.0], 4:[0.01, 0.1, 0.5, 1.0]}
print(max(stats, key=lambda k: sum(stats[k])))

qlearning = QLearning(no_episodes, mean_episodes, epsilons, lr, gamma)
qresults = qlearning.play()
sarsa = Sarsa(no_episodes, mean_episodes, epsilons, lr, gamma)
sarsaresults = sarsa.play()
mccontrol = MCControl(no_episodes, mean_episodes, epsilons)
mcresults = mccontrol.play()


best_epsilon  = {}
best_epsilon["qlearning"] = max(qresults, key=lambda k: np.amax(qresults[k]))
best_epsilon["sarsa"] = max(sarsaresults, key=lambda k: np.amax(sarsaresults[k]))
best_epsilon["mcresults"] = max(mcresults, key=lambda k: np.amax(mcresults[k]))


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(7,10))
for eps in epsilons:
    ax1.plot(range(0, no_episodes), np.mean(
        sarsaresults[eps], axis=1), label="e=" + str(eps))
    ax2.plot(range(0, no_episodes), np.mean(
        qresults[eps], axis=1), label="e=" + str(eps))
    ax3.plot(range(0, no_episodes), np.mean(
        mcresults[eps], axis=1), label="e=" + str(eps))
plt.legend()

ax4.plot(range(0, no_episodes), np.mean(
    qresults[best_epsilon["qlearning"]], axis=1), label="q=" + str(best_epsilon["qlearning"]))
ax4.plot(range(0, no_episodes), np.mean(
    sarsaresults[best_epsilon["sarsa"]], axis=1), label="s=" + str(best_epsilon["sarsa"]))
ax4.plot(range(0, no_episodes), np.mean(
    mcresults[best_epsilon["mcresults"]], axis=1), label="mc=" + str(best_epsilon["mcresults"]))
plt.legend()

ax1.set_title('Sarsa')
ax2.set_title('Qlearning')
ax3.set_title('MC Control')
ax4.set_title('Combined')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()



plt.legend()
plt.show()