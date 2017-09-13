import random
import math


class Actor:
    def __init__(self, bandit):
        self.bandit = bandit
#        self.estimates = [max(bandit.means) for _ in range(bandit.arms)]
        self.estimates = [0 for _ in range(bandit.arms)]
        self.total_rewards = [0 for _ in range(bandit.arms)]
        self.pulls = [0 for _ in range(bandit.arms)]
        self.overall_reward = 0

    def _choose_arm(self):
        return 0

    def act(self):
        arm = self._choose_arm()
        rew = self.bandit.pull(arm)
        self.total_rewards[arm] += rew
        self.pulls[arm] += 1
        self.estimates[arm] = self.total_rewards[arm] / self.pulls[arm]
        self.overall_reward += rew
        return [arm, rew]


class Greedy(Actor):
    def __init__(self, bandit):
        Actor.__init__(self, bandit)

    def _choose_arm(self):
        return self.estimates.index(max(self.estimates))


class EpsilonGreedy(Actor):
    def __init__(self, epsilon, bandit):
        self.epsilon = epsilon
        Actor.__init__(self, bandit)

    def _choose_arm(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.bandit.arms - 1)       # Explore
        else:
            return self.estimates.index(max(self.estimates))     # Exploit current knowledge


class Softmax(Actor):
    def __init__(self, temperature, bandit):
        self.temperature = temperature
        self.p = [1 / bandit.arms for _ in range(bandit.arms)]
        Actor.__init__(self, bandit)

    def _choose_arm(self):
        # TODO: Do not recalculate whole 'p' list every time. Just update what needs to be updated
        s = 0
        for i in range(self.bandit.arms):
            s += math.exp(self.estimates[i] / self.temperature)
            self.p[i] = s
        t = random.uniform(0, s)
        arm = 0
        while t > self.p[arm]:
            arm += 1
        return arm
