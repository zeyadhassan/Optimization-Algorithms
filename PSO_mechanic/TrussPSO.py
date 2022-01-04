# -*- coding: utf-8 -*-
"""
Implements that handle the creation of the truss structure and the PSO optimization

Unit system: kg, m, s, N, Pa
"""


############################################################################### \\ToDo: imperial -> metric system        

import numpy as np
import matplotlib.pyplot as plt

###############################################################################  

class Truss:
    
    def __init__(self, inpNode=3, inpAxis=1):
        
        '''
        TRUSS STRUCTURE
        
        ** Truss with nodes numbers and locations:
            1 - 3 - 5
            0 - 2 - 4
            
        ** Two pin supports are placed in nodes #0 and #4
        
        ** The numbering of the bars is the following:
            bar 0 connects node  0 and 1
            bar 1 connects node  0 and 2
            bar 2 connects node  1 and 2
            bar 3 connects node  1 and 3
            bar 4 connects node  0 and 3
            bar 5 coonects node  2 and 3
            bar 6 connects node  2 and 5
            bar 7 connects node  3 and 4
            bar 8 connects node  3 and 5
            bar 9 connects node  2 and 4
            bar 10 connects node 4 and 5
        
        ** Orientation of x,y axis; x and y axis start at node number 0:
            x: to the right (->)
            y: upwards (^)

            ** Input here how the force can be applied: in which node and which 
            axis direction.

        ** Parameters:
            inpNode : int, optional; varies from 0 to 5 (default value is 3)
            inpAxis : int, optional; x-axis -> 1 / y-axis -> 2 (default value is 2)
        
        Returns
        -------
        None.
        
        '''
        
        # -- Material properties:
        # - Young´s modulus
        self.E = 70e9 #Pa = 70 GPa
        # - material density                                    
        self.p = 2700 #kg/m^3                                       

        # -- Initialize empty bars, force and nodes arrays:
        self.nodes = []
        self.bars = []
        self.force = []

        # -- Support displacement is defined as 0 (4 x 0 for 2 directions/node)
        self.displacement_s = [0, 0, 0, 0]
        
        # -- Declare an empty DOFs array, 2 DOFs per node for x & y dir., resp.
        self.dof = []
                
        # -- Node and the axis of the applied force input:
        # - input node
        self.inpNode = inpNode 
        # - inpit axis
        self.inpAxis = inpAxis 
        
        # -- Force should be defined in the trussdesign() called in init
        self.trussdesign() 

    def trussdesign(self):
        # -- Coordinates of the each node [m] is inserted into nodes
        self.nodes.append([0, 0])
        self.nodes.append([0, 3])
        self.nodes.append([3, 0])
        self.nodes.append([3, 3])
        self.nodes.append([6, 0])
        self.nodes.append([6, 3])

        # -- Bars are defined by the starting and ending node 
        self.bars.append([0, 1])
        self.bars.append([0, 2])
        self.bars.append([1, 2])
        self.bars.append([1, 3])
        self.bars.append([0, 3])
        self.bars.append([2, 3])
        self.bars.append([2, 5])
        self.bars.append([3, 4])
        self.bars.append([3, 5])
        self.bars.append([2, 4])
        self.bars.append([4, 5])

        # -- PyList -> NumPy array
        self.nodes = np.array(self.nodes).astype(float)
        self.bars = np.array(self.bars)

        # -- Load [kN] information
        self.force = np.zeros_like(self.nodes)
        inpN = self.inpNode 
        inpA = self.inpAxis
        self.force[inpN, inpA-1] = -450000 #N = -450 kN

        # -- DOFs: 0 -> fixed DOF, 1 -> free DOF
        self.dof = np.ones_like(self.nodes).astype(int)
        self.dof[0, :] = 0
        self.dof[4, :] = 0
                
    # -- Structural analysis of truss -> displacement, stress and mass calculation
    def analysis(self, A):

        # -- self. -> local
        nodes = self.nodes
        bars = self.bars
        Force = self.force
        E = self.E
        Dof = self.dof
        p = self.p
        Displacement_s = self.displacement_s

        number_nodes = len(nodes)
        number_ele = len(bars)
        
        # -- DOFs per node (2D)
        number_Dof_node = 2                
        
        # -- total DOFs                                   
        number_Dof = number_Dof_node * number_nodes

        # -- structural analysis -> direct stiffness method (DSM)
        # -- increment of the x & y / member
        # -- for members relative to the CS -> transformation matrix

        # -- subtract starting coord. from the end coord. of a bar
        d = (nodes[bars[:, 1], :] - nodes[bars[:, 0], :])   
        
        # -- transpose the d matrix
        L = np.sqrt((d ** 2).sum(axis=1))
        angle = d.T / L                                     
        
        # -- {- cosa, -sina, cosa, sina} of each bar
        transformation_vector = np.concatenate((-angle.T, angle.T), axis=1)
                                                             
        # -- Stiffness matrix:
        # - size(K) = #DOFs
        K = np.zeros([number_Dof, number_Dof])               
        
        # -- if k= 0 -> bar 0
        # -- DOF at each node is defined
        for k in range(number_ele):
            aux = 2 * bars[k, :]                            
            index = np.r_[aux[0] : aux[0] + 2, aux[1] : aux[1] + 2] 
                                                            
            # -- element stiffness matrix
            Ke = (np.dot(transformation_vector[k][np.newaxis].T * E * A[k], 
                         transformation_vector[k][np.newaxis], ) / L[k])
            
            # -- summation of all element stiffness matrices
            # -- define fixed and free DOFs
            K[np.ix_(index, index)] = (K[np.ix_(index, index)] + Ke)
                                                            
        # -- non-zero DOFs
        freeDof = Dof.flatten().nonzero()[0]
        # -- zero DOFs                                                                     
        supportDof = (Dof.flatten() == 0).nonzero()[0]       
        
        # -- kff -> free DOF stiffness matrix
        # -- necessary if axial force calculation is needed
        kff = K[np.ix_(freeDof, freeDof)]
        
        # -- flatten the array of the forces
        # -- value from free DOFs is selected -> possible external force
        # -- displacement for all free DOFs
        Force_f = Force.flatten()[freeDof]                   
        Displacement_f = np.linalg.solve(kff, Force_f)       
        Displacement_total = Dof.astype(float).flatten()
        Displacement_total[freeDof] = Displacement_f
        Displacement_total[supportDof] = Displacement_s
        # -- reshape -> matrix [2x6] needed
        Displacement_total = Displacement_total.reshape(number_nodes, number_Dof_node)
          
        # -- DOFs for each bar information from the bar array                                                 
        displacement = np.concatenate((Displacement_total[bars[:, 0]], 
                                       Displacement_total[bars[:, 1]]), axis=1) 
                                                             
        # -- array of the axial forces
        Axial_force = (E * A[:] / L[:] * (transformation_vector[:] * 
                                          displacement[:]).sum(axis=1))
                                                             
        # -- stress value calculation
        Stress = Axial_force / A
        # -- total weight of truss structure calculation
        Mass = (p * A * L).sum()

        return Stress, Mass, Displacement_total
 
###############################################################################    
    
class PSO:
    
    def __init__(self, truss, c1=2, c2=2):
        
        # -- Create the structure
        self.S = truss     
                               
        # -- Problem definition
        # -- the variable <-> corresponds to the number of elements (11)
        self.d = 11             
                         
        self.xMin, self.xMax = 1e-5, 0.025 #m^2
        self.vMin, self.vMax = (-0.2 * (self.xMax - self.xMin), 
                                0.2 * (self.xMax - self.xMin),)
        
        self.maxIt = 1000
        
        # -- particles in the search space / population size
        self.ps = 30                                     
        ps = self.ps
        
        self.c1 = c1
        self.c2 = c2
        
        # -- initial weight
        self.w = 0.9 - ((0.9 - 0.4) / self.maxIt) *                           \
                 np.linspace(0, self.maxIt, self.maxIt)
      
        # -- stress limit                                        
        self.S_lim = 1.75e8   
        
        # -- displacement limit                               
        self.D_lim = 0.05                                    

        self.position = np.random.uniform(self.xMin, self.xMax, [ps, self.d])
        self.velocity = np.random.uniform(self.vMin, self.vMax, [ps, self.d])
        self.cost = np.zeros(ps)
        self.stress = np.zeros([ps, self.d])
        # -- 2 because we have two axis
        self.displacement = np.zeros([ps, len(self.S.nodes), 2])                                                

        for i in range(ps):
            self.stress[i], self.cost[i], self.displacement[i] =              \
            self.S.analysis(self.position[i])
            
        self.pbest = np.copy(self.position)
        self.pbest_cost = np.copy(self.cost)
        self.index = np.argmin(self.pbest_cost)
        self.gbest = self.pbest[self.index]
        self.gbest_cost = self.pbest_cost[self.index]
        self.BestCost = np.zeros(self.maxIt)
        self.Bestposition = np.zeros([self.maxIt, self.d])
        self.nodes = self.S.nodes
        
    # -- velocity limit
    def limitV(self, V):
        for i in range(len(V)):
            if V[i] > self.vMax:
                V[i] = self.vMax
            if V[i] < self.vMin:
                V[i] = self.vMin
        return V
    
    # -- cross section area limit
    def limitX(self, X):
        for i in range(len(X)):
            if X[i] > self.xMax:
                X[i] = self.xMax
            if X[i] < self.xMin:
                X[i] = self.xMin
        return X

    def evaluate(self):
        # -- self. -> local
        MaxIt = self.maxIt
        ps = self.ps
        w = self.w
        c1 = self.c1
        c2 = self.c2
        d = self.d
        S_lim = self.S_lim
        D_lim = self.D_lim
        nodes = self.nodes
        
        # -- nested loops -> each particle of the population per each iteration 
        for it in range(MaxIt):
            for i in range(ps):
                # -- update the velocity
                self.velocity[i] = (w[it] * self.velocity[i]             
                                    + c1 * np.random.rand(d) * 
                                    (self.pbest[i] - self.position[i])
                                    + c2 * np.random.rand(d) * 
                                    (self.gbest - self.position[i]))
            
                self.velocity[i] = self.limitV(self.velocity[i])
                 # -- update the position (cross-section area) 
                self.position[i] = self.position[i] + self.velocity[i]  
                self.position[i] = self.limitX(self.position[i])
                (self.stress[i], self.cost[i], self.displacement[i]) =        \
                 self.S.analysis(self.position[i])
                
                C_total = 0
                # -- check whether the stress is within predefined limits                
                for cd in range(d):                                      
                    if np.abs(self.stress[i, cd]) > S_lim:
                        C1 = np.abs((self.stress[i, cd] - S_lim) / S_lim)
                    else:
                        C1 = 0
                    C_total = C_total + C1
                    
                # -- check whether the displacement of each node w.r.t. x-axis
                # is within predefined limits
                for cx in range(len(nodes)):                             
                    if np.abs(self.displacement[i, cx, 0]) > D_lim:
                        C2 = np.abs((self.displacement[i, cx, 0] - D_lim) / D_lim)
                    else:
                        C2 = 0
                    C_total = C_total + C2
               
                # -- check whether the displacement of each node w.r.t. y-axis
                # is within predefined limits
                for cy in range(len(nodes)):                            
                    if np.abs(self.displacement[i, cy, 1]) > D_lim:
                        C3 = np.abs((self.displacement[i, cy, 1] - D_lim) / D_lim)
                    else:
                        C3 = 0
                    C_total = C_total + C3
                    
                phi = 1 + C_total
                
                self.cost[i] = self.cost[i] * phi
                
                if self.cost[i] < self.pbest_cost[i]:
                    self.pbest[i] = self.position[i]
                    self.pbest_cost[i] = self.cost[i]
                    if self.pbest_cost[i] < self.gbest_cost:
                        self.gbest = self.pbest[i]
                        self.gbest_cost = self.pbest_cost[i]
                        
            self.BestCost[it] = self.gbest_cost
            self.Bestposition[it] = self.gbest

    ##### PRINT & PLOT RESULTS #####

    # -- print: evaluted cross-section areas, stresses, displacements  
    # -- fitness value
    def printOptResults(self):    
        print("Design Variable A in [m^2]:")
        print(self.Bestposition[-1][np.newaxis].T)
        print()
        stress, cost, displacement = self.S.analysis(self.Bestposition[-1])
        print("Stress [Pa]:")
        print(stress[np.newaxis].T)
        print()
        print("Displacement [m]:")
        print(displacement)
        print()
        print("Best Fitness Value =", self.gbest_cost)
        print()
            
    # -- plot: cost function / fitness value vs. iteration
    def plotCostFunction(self, figNumber):
        plt.figure(figNumber)
        plt.plot(self.BestCost, 'black', linewidth=2.5)
        plt.title("PSO Cost Function")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Value")       
    
    # -- plot: deformation 
    def plotDeformation(self, nodes, bars, c, lt, lw, lg, figNumber):
        plt.figure(figNumber)
        plt.title("Truss Deformation")
        for i in range(len(bars)):
            # Xs coordinate -> start point, Xf coordinate -> end point
            Xs, Xf = nodes[bars[i, 0], 0], nodes[bars[i, 1], 0]   
            Ys, Yf = nodes[bars[i, 0], 1], nodes[bars[i, 1], 1]
            line, = plt.plot([Xs, Xf], [Ys, Yf], color=c, 
                             linestyle=lt, linewidth=lw)   
        line.set_label(lg)
        plt.legend(title = "configuration: ", prop= {'size': 10}, loc ='upper center', \
                bbox_to_anchor=(0.5, -0.1))
        plt.axis('scaled')
        
    
    # -- plot: design variable -> c.s. area
    def plotAreaDesign(self, nodes, bars, figNumber):
        fig = plt.figure(figNumber) 
        ax = fig.gca()
        ax.set_title("Truss Areas")
        ax.set_ylim(-0.5, 2*max(self.nodes[:,1]))

        # - plot supports
        index = 0
        for e in self.S.dof:
            if e[0] == 0 and e[1] == 0:
                ax.plot(nodes[index,0], nodes[index,1], marker='^', 
                         markersize=20, color= 'black', zorder = 1)
            index += 1
        
        # - area scaling (0 -> 1) * factor + n (factor, n affect line width)
        areaScale = np.zeros_like(self.Bestposition[-1])
        for i in range(len(self.Bestposition[-1])):
            areaScale[i] = (self.Bestposition[-1,i] - min(self.Bestposition[-1]))/           \
                        (max(self.Bestposition[-1])-min(self.Bestposition[-1])) * 5 + 3                
        # - plot bars
        count = 0
        for i in range(len(bars)):
            line, = ax.plot([nodes[bars[i, 0], 0], nodes[bars[i, 1], 0]], 
                             [nodes[bars[i, 0], 1], nodes[bars[i, 1], 1]], 
                             color='black', linestyle='-', zorder = 2, 
                             linewidth=areaScale[count])
            count += 1    
        
        # - plot joints
        ax.scatter(*nodes.T, s=150, facecolors='white', edgecolors='black', 
                    marker='o', linewidth = 2.5, zorder = 3)    
        
        # -- plot force position and direction:
            
        # - applied force in the x direction,
        # - hence, the 'insertion point' of the arrow will change in the 
        # - x coordinate to 'land' on the hinge
        if (self.S.inpAxis == 1):
            ax.arrow(self.nodes[self.S.inpNode][0]-self.S.force[self.S.inpNode][0]/225000, 
                      self.nodes[self.S.inpNode][1], 
                      self.S.force[self.S.inpNode][0]/225000, 0,  
                      facecolor='r', width=2.5, length_includes_head=True, zorder = 4)
        # - applied force in the y direction,
        # - hence, the 'insertion point' of the arrow will change in the 
        # - y coordinate to 'land' on the hinge
        elif (self.S.inpAxis == 2):
            ax.arrow(self.nodes[self.S.inpNode][0], 
                      self.nodes[self.S.inpNode][1]-self.S.force[self.S.inpNode][1]/225000, 
                      0, self.S.force[self.S.inpNode][1]/225000, 
                      facecolor='r', width=0.2, head_width=0.4, 
                      length_includes_head=True, zorder = 4)
        #ax.axis('scaled')  

###############################################################################  