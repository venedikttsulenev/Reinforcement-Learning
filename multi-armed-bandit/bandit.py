import numpy as np


class Bandit:
    def __init__(self, arms):
        self.arms = arms


class StationaryBandit(Bandit):
    def __init__(self, arms):
        Bandit.__init__(self, arms)
        self.means = np.random.normal(0, 1, arms)

    def pull(self, arm):
        return np.random.normal(self.means[arm], 1)


class NonStationaryBandit(Bandit):
    def __init__(self, arms):
        Bandit.__init__(self, arms)
        self.means = np.zeros(arms)

    def pull(self, arm):
        self.means += np.random.normal(0, 0.1, self.arms)      # random walk
        return np.random.normal(self.means[arm], 1)


class BanditFactory:
    def __init__(self, arms):
        self.arms = arms

    def new_bandit(self):
        return Bandit(self.arms)


class StationaryBanditFactory(BanditFactory):
    def __init__(self, arms):
        BanditFactory.__init__(self, arms)

    def new_bandit(self):
        return StationaryBandit(self.arms)


class NonStationaryBanditFactory(BanditFactory):
    def __init__(self, arms):
        BanditFactory.__init__(self, arms)

    def new_bandit(self):
        return NonStationaryBandit(self.arms)
