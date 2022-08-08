import random

import numpy as np
import matplotlib.pyplot as plt
from impl_create_game import KArmBanditGame


def main():
    epsilons = [0, 0.01, 0.1]
    K = 10
    per_run = 10000
    runs = 2000
    game = KArmBanditGame(K)

    for epsilon in epsilons:
        rewards = []
        Q_values = {action: 0 for action in range(K)}
        Rate_actions = {action: 0 for action in range(K)}
        Reward_actions = {action: 0 for action in range(K)}
        for step in range(per_run):
            if random.random() < epsilon:
                action = random.randint(0, K-1)
            else:
                action = max(Q_values, key=Q_values.get)
            reward = game.step(action)
            rewards.append(reward)

            # update Q and Rate
            Rate_actions[action] += 1
            Reward_actions[action] += reward
            Q_values[action] = Reward_actions[action] / Rate_actions[action]

        plt.plot(rewards, label=f'{epsilon}')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
