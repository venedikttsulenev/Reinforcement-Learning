##########################################################################################
# Usage: python 3alg.py <bandits> <arms> <iterations> <epsilon> <tau> <initial_estimate> #
##########################################################################################
# Generates "graph.pdf" graph containing learning curves for                             #
# greedy, epsilon-greedy and softmax methods used to solve n-armed bandit problem        #
##########################################################################################
from bandit import Bandit
from actors import Greedy
from actors import EpsilonGreedy
from actors import Softmax
import sys
import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pp
import numpy as np
import time
import main

##########################################################################################
bandits = 100
arms = 10
iterations = 1000
epsilon = 0.1
tau = 0.1
init_estimate = 0
filename = 'graph.pdf'
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
                    if len(sys.argv) > 6:
                        init_estimate = float(sys.argv[6])
                        if len(sys.argv) > 7:
                            filename = sys.argv[7]

print "bandits:", bandits
print "arms:", arms
print "iterations:", iterations
print "init_estimate:", init_estimate
print "filename: \'" + filename + '\''

##########################################################################################
pdf = PdfPages(filename)

cl = time.clock()
b = Bandit(arms)
actors = [Greedy(b, init_estimate), EpsilonGreedy(epsilon, b, init_estimate), Softmax(tau, b, init_estimate)]
avg = main.run(actors, bandits, arms, iterations)
cl = time.clock() - cl
print "time: " + str(cl) + " s"

x = np.arange(iterations)
pp.plot(x, avg[0], aa=False, linewidth=1, label="greedy")
pp.plot(x, avg[1], aa=False, linewidth=1, label="$\epsilon$-greedy ($\epsilon$ = " + str(epsilon) + ")")
pp.plot(x, avg[2], aa=False, linewidth=1, label="softmax ($\\tau$ = " + str(tau) + ")")
pp.gcf().set_size_inches(12, 6)
pp.legend(loc="lower right")
pp.xlabel("Games")
pp.ylabel("Average reward per game")

pdf.savefig()
pdf.close()
