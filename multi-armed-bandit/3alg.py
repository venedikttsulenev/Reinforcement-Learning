#####################################################################################################
# Usage: python 3alg.py <bandits> <arms> <iterations> <epsilon> <tau> <initial_estimate> <filename> #
#####################################################################################################
# Generates "graph.pdf" graph containing learning curves for                                        #
# greedy, epsilon-greedy and softmax methods used to solve n-armed bandit problem                   #
#####################################################################################################
from actors import Greedy, EpsilonGreedy, Softmax, SampleAverageStepSize
from bandit import StationaryBanditFactory
import sys
import time
import main
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pp

#####################################################################################################
nbandits = 100
narms = 10
niterations = 1000
epsilon = 0.1
tau = 0.1
init_estimate = 0
filename = 'graph.pdf'
if len(sys.argv) > 1:
    nbandits = int(sys.argv[1])
    if len(sys.argv) > 2:
        narms = int(sys.argv[2])
        if len(sys.argv) > 3:
            niterations = int(sys.argv[3])
            if len(sys.argv) > 4:
                epsilon = float(sys.argv[4])
                if len(sys.argv) > 5:
                    tau = float(sys.argv[5])
                    if len(sys.argv) > 6:
                        init_estimate = float(sys.argv[6])
                        if len(sys.argv) > 7:
                            filename = sys.argv[7]

print "bandits:", nbandits
print "arms:", narms
print "iterations:", niterations
print "init_estimate:", init_estimate
print "filename: \'" + filename + '\''

#####################################################################################################
pdf = PdfPages(filename)

cl = time.clock()
bf = StationaryBanditFactory(narms)
b = bf.new_bandit()
step_size = SampleAverageStepSize()
actors = [Greedy(b, init_estimate, step_size),
          EpsilonGreedy(epsilon, b, init_estimate, step_size),
          Softmax(tau, b, init_estimate, step_size)]
avg = main.run(actors, bf, narms, niterations)
cl = time.clock() - cl
print "time: " + str(cl) + " s"

x = np.arange(niterations)
pp.plot(x, avg[0], aa=False, linewidth=1, label="greedy")
pp.plot(x, avg[1], aa=False, linewidth=1, label="$\epsilon$-greedy ($\epsilon$ = " + str(epsilon) + ")")
pp.plot(x, avg[2], aa=False, linewidth=1, label="softmax ($\\tau$ = " + str(tau) + ")")
pp.gcf().set_size_inches(12, 6)
pp.legend(loc="lower right")
pp.xlabel("Games")
pp.ylabel("Average reward per game")

pdf.savefig()
pdf.close()
