import numpy as np


def _solve(actors, niterations, rew):
    for j in range(len(actors)):
        for i in range(niterations):
            rew[j][i] = actors[j].act()


def run(actors, bandit_factory, nbandits, niterations):
    l = len(actors)
    rew = np.zeros((l, niterations))
    avg = np.zeros((l, niterations))
    for i in range(nbandits - 1):
        _solve(actors, niterations, rew)
        avg += rew
        b = bandit_factory.new_bandit()
        for a in actors:
            a.set_bandit(b)
    _solve(actors, niterations, rew)
    avg += rew
    avg /= nbandits
    return avg
