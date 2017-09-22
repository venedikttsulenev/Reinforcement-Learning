import math
import numpy as np


class StepSize:
    def get_value(self):
        pass


class ConstantStepSize(StepSize):
    def __init__(self, step_size):
        self.step_size = step_size

    def get_value(self):
        return self.step_size


class SampleAverageStepSize(StepSize):
    def __init__(self):
        self.k = 0

    def get_value(self):
        self.k += 1
        return math.pow(self.k, -1)


class Actor:
    def __init__(self, bandit, initial_estimate, step_size):
        self.initial_estimate = initial_estimate
        self.bandit = bandit
        self.step_size = step_size
        self.estimates = np.empty(bandit.arms)
        self.estimates.fill(initial_estimate)
        self.pulls = np.zeros(bandit.arms, dtype=np.int64)

    def _choose_arm(self):
        return 0

    def act(self):
        arm = self._choose_arm()
        rew = self.bandit.pull(arm)
        self.pulls[arm] += 1
        self.estimates[arm] += (rew - self.estimates[arm]) * self.step_size.get_value()
        return rew

    def set_bandit(self, bandit):
        self.bandit = bandit
        self.estimates.fill(self.initial_estimate)
        self.pulls.fill(0)


class Greedy(Actor):
    def __init__(self, bandit, initial_estimate, step_size):
        Actor.__init__(self, bandit, initial_estimate, step_size)

    def _choose_arm(self):
        return self.estimates.argmax()


class EpsilonGreedy(Actor):
    def __init__(self, epsilon, bandit, initial_estimate, step_size):
        self.epsilon = epsilon
        Actor.__init__(self, bandit, initial_estimate, step_size)

    def _choose_arm(self):
        if np.random.uniform(0, 1, 1) < self.epsilon:
            return np.random.randint(0, self.bandit.arms - 1)       # Explore
        else:
            return self.estimates.argmax()                          # Exploit current knowledge


class Softmax(Actor):
    def __init__(self, temperature, bandit, initial_estimate, step_size):
        self.temperature = temperature
        self.p = np.empty(bandit.arms)
        Actor.__init__(self, bandit, initial_estimate, step_size)

    def _choose_arm(self):
        s = 0
        for i in range(self.bandit.arms):
            s += math.exp(self.estimates[i] / self.temperature)
            self.p[i] = s
        t = np.random.uniform(0, s, 1)
        arm = 0
        while t > self.p[arm]:
            arm += 1
        return arm
