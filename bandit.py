import random


class Bandit:
    def __init__(self, arms):
        self.arms = arms
        self.means = [random.gauss(0, 1) for _ in range(arms)]

    def pull(self, arm):
        return random.gauss(self.means[arm], 1)
