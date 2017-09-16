#####################################################################################################
# Usage: python softmax_tau.py <bandits> <arms> <iterations> <initial_estimate> <tau1> <tau2> <...> #
#####################################################################################################
# Learning curves for softmax method                                                                #
#####################################################################################################
from bandit import Bandit
from actors import Softmax
import sys
import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pp
import numpy as np
import time
import main

#####################################################################################################
bandits = 2000
arms = 10
iterations = 1000
initial_estimate = 0
tau = [1, 0.1, 0.01]
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
                        tau = [float(arg) for arg in sys.argv[6:]]
print "bandits:", bandits
print "arms:", arms
print "iterations:", iterations
print "initial_estimate:", initial_estimate
print "output file: \'" + filename + '\''

#####################################################################################################
pdf = PdfPages(filename)

cl = time.clock()
b = Bandit(arms)
actors = [Softmax(t, b, initial_estimate) for t in tau]
avg = main.run(actors, bandits, arms, iterations)
x = np.arange(iterations)
for i in range(len(avg)):
    pp.plot(x, avg[i], aa=False, linewidth=1, label="$\\tau$ = " + str(tau[i]))
cl = time.clock() - cl
print "time: " + str(cl) + " s"
pp.gcf().set_size_inches(12, 6)
pp.legend(loc="lower right")
pp.xlabel("Games")
pp.ylabel("Average reward per game")
pdf.savefig()
pdf.close()
