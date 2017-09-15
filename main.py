import numpy as np
from bandit import Bandit


def _solve(actors, iterations, rew):
    for j in range(3):
        for i in range(iterations):
            rew[j][i] = actors[j].act()


def run(actors, bandits, arms, iterations):
    rew = np.zeros((3, iterations))
    avg = np.zeros((3, iterations))
    for i in range(bandits - 1):
        _solve(actors, iterations, rew)
        avg += rew
        b = Bandit(arms)
        for a in actors:
            a.set_bandit(b)
    _solve(actors, iterations, rew)
    avg += rew
    avg /= bandits
    return avg
