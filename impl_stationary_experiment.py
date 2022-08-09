import random

import numpy as np
import matplotlib.pyplot as plt
from impl_create_game import KArmBanditGame, KArmBanditGameNonStationary


def plot_results(results):
    for epsilon, mat in results.items():
        plt.plot(np.mean(mat, 0), label=f'{epsilon}')
    plt.legend()
    plt.show()


def stationary_experiment():
    epsilons = [0, 0.01, 0.1]
    # epsilons = [0.1]
    K = 10
    per_run = 1000
    runs = 400
    results = {epsilon: np.zeros((runs, per_run)) for epsilon in epsilons}
    game = KArmBanditGame(K)
    # game = KArmBanditGameNonStationary(K)

    for epsilon in epsilons:
        for run in range(runs):
            Q_values = {action: 0 for action in range(K)}
            Rate_actions = {action: 0 for action in range(K)}
            for step in range(per_run):
                if random.random() < epsilon:
                    action = random.randint(0, K-1)
                else:
                    action = max(Q_values, key=Q_values.get)
                reward = game.step(action)
                results[epsilon][run][step] = reward
                print(f'\repsilon: {epsilon}, run: {run}, step: {step}, reward={reward}', end='')

                # update Q and Rate
                Rate_actions[action] += 1
                Q_values[action] += (1/Rate_actions[action])*(reward - Q_values[action])

    plot_results(results)


if __name__ == '__main__':
    stationary_experiment()
