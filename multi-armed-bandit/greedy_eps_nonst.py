##########################################################################################################################
# Usage: python greedy_eps_nonst.py <nbandits> <narms> <niterations> <initial_estimate> <filename> <epsilon> <step_size> #
##########################################################################################################################
#  Generates two learning curves for epsilon-greedy action selection method: one for sample-average step size and        #
# one for constant step size. n-armed bandit problem is not stationary. Means for every arm take random walks.           #
##########################################################################################################################
from bandit import NonStationaryBanditFactory
from actors import EpsilonGreedy, SampleAverageStepSize, ConstantStepSize
import sys
import time
import main
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pp

##########################################################################################################################
nbandits = 2000
narms = 10
niterations = 1000
initial_estimate = 0
eps = 0.01
filename = 'graph.pdf'
css = 0.1
if len(sys.argv) > 1:
    nbandits = int(sys.argv[1])
    if len(sys.argv) > 2:
        narms = int(sys.argv[2])
        if len(sys.argv) > 3:
            niterations = int(sys.argv[3])
            if len(sys.argv) > 4:
                initial_estimate = float(sys.argv[4])
                if len(sys.argv) > 5:
                    filename = sys.argv[5]
                    if len(sys.argv) > 6:
                        eps = float(sys.argv[6])
                        if len(sys.argv) > 7:
                            css = float(sys.argv[7])

print "bandits:", nbandits
print "arms:", narms
print "iterations:", niterations
print "initial_estimate:", initial_estimate
print "output file: \'" + filename + '\''

###########################################################################################################################
pdf = PdfPages(filename)

cl = time.clock()
bf = NonStationaryBanditFactory(narms)
b = bf.new_bandit()
actors = [EpsilonGreedy(eps, b, initial_estimate, SampleAverageStepSize()),
          EpsilonGreedy(eps, b, initial_estimate, ConstantStepSize(css))]
avg = main.run(actors, bf, nbandits, niterations)
x = np.arange(niterations)
pp.plot(x, avg[0], aa=False, linewidth=1, label="Sample-average step size")
pp.plot(x, avg[1], aa=False, linewidth=1, label="Step size = " + str(css))
cl = time.clock() - cl
print "time: " + str(cl) + " s"
pp.gcf().set_size_inches(12, 6)
pp.legend(loc="lower right")
pp.xlabel("Games")
pp.ylabel("Average reward per game")
pdf.savefig()
pdf.close()
