import random
import numpy as np


class Bandit:
    def __init__(self, arms):
        self.arms = arms
        self.means = np.array([random.gauss(0, 1) for _ in range(arms)])

    def pull(self, arm):
        return random.gauss(self.means[arm], 1)
