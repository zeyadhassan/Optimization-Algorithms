# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:25:42 2021

@author: Egi Kalaj
"""
###############################################################################

from TrussPSO import Truss, PSO
from matplotlib import pyplot as plt

###############################################################################

truss = Truss(inpNode=3, inpAxis=2)
optimization = PSO(truss=truss, c1=2, c2=2)

###############################################################################

print(">> Optimization:")
print()
optimization.evaluate()

###############################################################################

print(">> Print Results:")
print()
optimization.printOptResults()

###############################################################################

print(">> Plot Cost Function:")
print()
optimization.plotCostFunction(0)

###############################################################################

print(">> Plot Deformation:")
print()
# initial configuration
optimization.plotDeformation(optimization.nodes, 
                             optimization.S.bars, 'black', '--',2 , 'undeformed', 1)
# deformed configuration
optimization.plotDeformation(optimization.nodes + 
                             optimization.displacement[optimization.index], 
                             optimization.S.bars, 'red', '-', 2.5, 'deformed', 1)
    
###############################################################################

print(">> Plot Cross Section Thickness: ")
print()
optimization.plotAreaDesign(optimization.nodes, optimization.S.bars, 2)

###############################################################################

plt.show()