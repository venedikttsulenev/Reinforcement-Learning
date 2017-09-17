####################################################################################################
# Usage: python greedy_eps.py <bandits> <arms> <iterations> <initial_estimate> <eps1> <eps2> <...> #
####################################################################################################
# Learning curves for epsilon-greedy method                                                        #
####################################################################################################

from bandit import Bandit
from actors import EpsilonGreedy
import sys
import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pp
import numpy as np
import time
import main

####################################################################################################
bandits = 2000
arms = 10
iterations = 1000
initial_estimate = 0
eps = [1, 0.1, 0.01]
filename = 'graph.pdf'
if len(sys.argv) > 1:
    bandits = int(sys.argv[1])
    if len(sys.argv) > 2:
        arms = int(sys.argv[2])
        if len(sys.argv) > 3:
            iterations = int(sys.argv[3])
            if len(sys.argv) > 4:
                initial_estimate = float(sys.argv[4])
                if len(sys.argv) > 5:
                    filename = sys.argv[5]
                    if len(sys.argv) > 6:
                        eps = [float(arg) for arg in sys.argv[6:]]
print "bandits:", bandits
print "arms:", arms
print "iterations:", iterations
print "initial_estimate:", initial_estimate
print "output file: \'" + filename + '\''

####################################################################################################
pdf = PdfPages(filename)

cl = time.clock()
b = Bandit(arms)
actors = [EpsilonGreedy(e, b, initial_estimate) for e in eps]
avg = main.run(actors, bandits, arms, iterations)
x = np.arange(iterations)
for i in range(len(avg)):
    pp.plot(x, avg[i], aa=False, linewidth=1, label="$\epsilon$ = " + str(eps[i]))
cl = time.clock() - cl
print "time: " + str(cl) + " s"
pp.gcf().set_size_inches(12, 6)
pp.legend(loc="lower right")
pp.xlabel("Games")
pp.ylabel("Average reward per game")
pdf.savefig()
pdf.close()
