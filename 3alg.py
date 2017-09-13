#######################################################################
# Usage: python 3alg.py <bandits> <arms> <iterations> <epsilon> <tau> #
#######################################################################
# Generates "graph.pdf" graph containing learning curves for          #
# greedy, epsilon-greedy and softmax methods used to solve            #
# n-armed bandit problem                                              #
#######################################################################
from bandit import Bandit
from actors import Greedy
from actors import EpsilonGreedy
from actors import Softmax
import sys
import operator
import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pp

pdf = PdfPages('graph.pdf')

bandits = 100
arms = 10
iterations = 1000
epsilon = 0.1
tau = 0.1
if len(sys.argv) > 1:
    bandits = int(sys.argv[1])
    if len(sys.argv) > 2:
        arms = int(sys.argv[2])
        if len(sys.argv) > 3:
            iterations = int(sys.argv[3])
            if len(sys.argv) > 4:
                epsilon = float(sys.argv[4])
                if len(sys.argv) > 5:
                    tau = float(sys.argv[5])
print "bandits:", bandits
print "arms:", arms
print "iterations:", iterations

total_greedy_avg = [0 for _ in range(iterations)]
total_e_greedy_avg = [0 for _ in range(iterations)]
total_softmax_avg = [0 for _ in range(iterations)]

greedy_avg = [0 for _ in range(iterations)]
e_greedy_avg = [0 for _ in range(iterations)]
softmax_avg = [0 for _ in range(iterations)]

bdt = [Bandit(arms) for _ in range(bandits)]
for b in bdt:
    a = Greedy(b)   # Create greedy actor playing 10-armed bandit
    for i in range(iterations):
        a.act()     # Play one time
        greedy_avg[i] = a.overall_reward / sum(a.pulls)
    total_greedy_avg = map(operator.add, total_greedy_avg, greedy_avg)

    a = EpsilonGreedy(epsilon, b)
    for i in range(iterations):
        a.act()
        e_greedy_avg[i] = a.overall_reward / sum(a.pulls)
    total_e_greedy_avg = map(operator.add, total_e_greedy_avg, e_greedy_avg)

    a = Softmax(tau, b)
    for i in range(iterations):
        a.act()
        softmax_avg[i] = a.overall_reward / sum(a.pulls)
    total_softmax_avg = map(operator.add, total_softmax_avg, softmax_avg)
total_greedy_avg = [e / bandits for e in total_greedy_avg]
total_e_greedy_avg = [e / bandits for e in total_e_greedy_avg]
total_softmax_avg = [e / bandits for e in total_softmax_avg]
x = [i for i in range(iterations)]
pp.plot(x, total_greedy_avg, aa=False, label="greedy")
pp.plot(x, total_e_greedy_avg, aa=False, label="$\epsilon$-greedy ($\epsilon$ = " + str(epsilon) + ")")
pp.plot(x, total_softmax_avg, aa=False, label="softmax ($\\tau$ = " + str(tau) + ")")
pp.legend(loc="lower right")
pp.xlabel("Games")
pp.ylabel("Average reward per game")

pdf.savefig()
pdf.close()
