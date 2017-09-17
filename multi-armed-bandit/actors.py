import random
import math
import numpy as np


class Actor:
    def __init__(self, bandit, initial_estimate):
        self.initial_estimate = initial_estimate
        self.bandit = bandit
        self.estimates = np.empty(bandit.arms)
        self.estimates.fill(initial_estimate)
        self.total_rewards = np.zeros(bandit.arms)
        self.pulls = np.zeros(bandit.arms, dtype=np.int64)

    def _choose_arm(self):
        return 0

    def act(self):
        arm = self._choose_arm()
        rew = self.bandit.pull(arm)
        self.total_rewards[arm] += rew
        self.pulls[arm] += 1
        self.estimates[arm] = self.total_rewards[arm] / self.pulls[arm]
        return rew

    def set_bandit(self, bandit):
        self.bandit = bandit
        self.estimates.fill(self.initial_estimate)
        self.total_rewards.fill(0)
        self.pulls.fill(0)


class Greedy(Actor):
    def __init__(self, bandit, initial_estimate):
        Actor.__init__(self, bandit, initial_estimate)

    def _choose_arm(self):
        return self.estimates.argmax()


class EpsilonGreedy(Actor):
    def __init__(self, epsilon, bandit, initial_estimate):
        self.epsilon = epsilon
        Actor.__init__(self, bandit, initial_estimate)

    def _choose_arm(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.bandit.arms - 1)       # Explore
        else:
            return self.estimates.argmax()                       # Exploit current knowledge


class Softmax(Actor):
    def __init__(self, temperature, bandit, initial_estimate):
        self.temperature = temperature
        self.p = np.empty(bandit.arms)
        Actor.__init__(self, bandit, initial_estimate)

    def _choose_arm(self):
        s = 0
        for i in range(self.bandit.arms):
            s += math.exp(self.estimates[i] / self.temperature)
            self.p[i] = s
        t = random.uniform(0, s)
        arm = 0
        while t > self.p[arm]:
            arm += 1
        return arm
