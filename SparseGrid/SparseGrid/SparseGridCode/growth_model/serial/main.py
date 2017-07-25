#======================================================================
#
#     This routine solves an infinite horizon growth model 
#     with dynamic programming and sparse grids
#
#     The model is described in Scheidegger & Bilionis (2017)
#     https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
#
#     external libraries needed:
#     - IPOPT (https://projects.coin-or.org/Ipopt)
#     - PYIPOPT (https://github.com/xuy/pyipopt)
#     - TASMANIAN (http://tasmanian.ornl.gov/)
#
#     Simon Scheidegger, 11/16 ; 07/17
#======================================================================

import nonlinear_solver_initial as solver     #solves opt. problems for terminal VF
import nonlinear_solver_iterate as solviter   #solves opt. problems during VFI
from parameters import *                      #parameters of model
import interpolation as interpol              #interface to sparse grid library/terminal VF
import interpolation_iter as interpol_iter    #interface to sparse grid library/iteration
import postprocessing as post                 #computes the L2 and Linfinity error of the model

import TasmanianSG                            #sparse grid library
import numpy as np

#======================================================================
def main(n_agents, iDepth, thetavec, pi):
    # Start with Value Function Iteration

    # terminal value function
    valnew=TasmanianSG.TasmanianSparseGrid()
    if (numstart==0):
        valnew=interpol.sparse_grid(n_agents, iDepth, theta = thetavec[2])
        valnew.write("valnew_1." + str(numstart) + ".txt") #write file to disk for restart

        # value function during iteration
    else:
        valnew.read("valnew_1." + str(numstart) + ".txt")  #write file to disk for restart

    valold=TasmanianSG.TasmanianSparseGrid()
    valold=valnew

    for i in range(numstart, numits):
        valnew = TasmanianSG.TasmanianSparseGrid()
        valnew0 = TasmanianSG.TasmanianSparseGrid()
        valnew1 = TasmanianSG.TasmanianSparseGrid()
        valnew2 = TasmanianSG.TasmanianSparseGrid()
        valnew3 = TasmanianSG.TasmanianSparseGrid()
        valnew4 = TasmanianSG.TasmanianSparseGrid()
        
        valnew0 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[0])
        valnew1 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[1])
        valnew2 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[2])
        valnew3 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[3])
        valnew4 = interpol_iter.sparse_grid_iter(n_agents, iDepth, valold, thetavec[4])
                
        evalpoints = valnew2.getPoints()
    
        val0 = valnew0.evaluateBatch(evalpoints)[:,0]
        print "val0", val0
        val1 = valnew1.evaluateBatch(evalpoints)[:,0]
        print "val1", val1
        val2 = valnew2.evaluateBatch(evalpoints)[:,0]
        print "val2", val2
        val3 = valnew3.evaluateBatch(evalpoints)[:,0]
        print "val3", val3
        val4 = valnew4.evaluateBatch(evalpoints)[:,0]
        print "val4", val4
        

        #take average of 5 grids
        valmean = pi[0]*val0 + pi[1]*val1 + pi[2]*val2 + pi[3]*val3 + pi[4]*val4
        valmean = np.reshape(valmean, (evalpoints.shape[0],1))
        valold = TasmanianSG.TasmanianSparseGrid()
        valold.copyGrid(valnew2)
        valold.loadNeededPoints(valmean)
        
        valold.write("valnew_1." + str(i+1) + ".txt")

    # compute errors
    avg_err=post.ls_error(n_agents, numstart, numits, No_samples)

    return avg_err

print(main(2,2,thetavec,pi))

'''
errlist = []
for i in range(1,3):
    for j in range(2,4):
        errlist.append(main(i,j,thetavec,pi))

print "Max and Avg Error: ", '\n'
print "Agents: 1, Depth: 2", errlist[0], '\n'
print "Agents: 1, Depth: 3", errlist[1], '\n'    
print "Agents: 2, Depth: 2", errlist[2], '\n'
print "Agents: 2, Depth: 3", errlist[3], '\n'
'''
