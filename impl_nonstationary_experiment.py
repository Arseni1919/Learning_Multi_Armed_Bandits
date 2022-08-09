import random

import numpy as np
import matplotlib.pyplot as plt
from impl_create_game import KArmBanditGame, KArmBanditGameNonStationary


def plot_results(results):
    for method, mat in results.items():
        plt.plot(np.mean(mat, 0), label=f'{method}')
    plt.legend()
    plt.show()


def plot_results_with_epsilon(results_with_epsilon, method_colors):

    from_n = 500

    for eps, results in results_with_epsilon.items():
        for method, mat in results.items():
            y = np.mean(np.mean(mat, 0)[from_n:])
            plt.scatter([eps], [y], label=f'method: {method}, eps:{eps}', c=method_colors[method])
    plt.legend()
    plt.show()


def nonstationary_experiment():
    methods = {
        'avr': None,
        # 'const 0.01': 0.01,
        # 'const 0.05': 0.05,
        'const 0.1': 0.1,
        # 'const 0.2': 0.2,
        # 'const 0.4': 0.4,
        # 'const 0.8': 0.8,
        # 'const 1': 1.0
    }
    method_colors = {
        'avr': 'r',
        # 'const 0.01': 0.01,
        # 'const 0.05': 0.05,
        'const 0.1': 'b',
        # 'const 0.2': 0.2,
        # 'const 0.4': 0.4,
        # 'const 0.8': 0.8,
        # 'const 1': 1.0
    }
    # methods = {'const 0.1': 0.1}
    epsilons = [0, 0.01, 0.05, 0.1, 0.2]
    # epsilon = 0.1
    K = 10
    per_run = 1000
    runs = 200
    # results = {method: np.zeros((runs, per_run)) for method in methods}
    results_with_epsilon = {eps: {method: np.zeros((runs, per_run)) for method in methods} for eps in epsilons}
    # game = KArmBanditGame(K)
    game = KArmBanditGameNonStationary(K)
    for eps in epsilons:
        for method, alpha in methods.items():
            for run in range(runs):
                Q_values = {action: 5 for action in range(K)}
                Rate_actions = {action: 0 for action in range(K)}
                for step in range(per_run):
                    if random.random() < eps:
                        action = random.randint(0, K-1)
                    else:
                        action = max(Q_values, key=Q_values.get)
                    reward = game.step(action)
                    # results[method][run][step] = reward
                    results_with_epsilon[eps][method][run][step] = reward
                    print(f'\rmethod: {method}, run: {run}, step: {step}, reward={reward}', end='')

                    # update Q and Rate
                    if method == 'avr':
                        Rate_actions[action] += 1
                        Q_values[action] += (1/Rate_actions[action])*(reward - Q_values[action])

                    elif 'const' in method:
                        Q_values[action] += alpha * (reward - Q_values[action])
        # plot_results(results)
    plot_results_with_epsilon(results_with_epsilon, method_colors=method_colors)


if __name__ == '__main__':
    nonstationary_experiment()
