# Learning K-Armed Bandits Game

```python
import numpy as np
import matplotlib.pyplot as plt


class KArmBanditGame:
    def __init__(self, k=10):
        self.k = k
        self.g_star_values = np.random.normal(0, 1, size=self.k)

    def step(self, action):
        reward = np.random.normal(self.g_star_values[action], 1, 1)[0]
        return reward


def main():
    K = 10
    game = KArmBanditGame(K)
    reward = game.step(2)
    print(reward)

    for action in range(K):
        plot_x, plot_y = [], []
        for _ in range(200):
            plot_x.append(action)
            plot_y.append(game.step(action))

        plt.scatter(plot_x, plot_y, s=1, label=f'action {action}')

    plt.legend()
    plt.show()
```

## Credits

- Sutton and Barto - RL: An Introduction