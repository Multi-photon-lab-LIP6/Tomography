### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
import scipy
import time

import os
import glob

from pathlib import Path
import fnmatch

import mystic
from mystic.solvers import DifferentialEvolutionSolver, diffev2 ,fmin
from mystic.strategy import Best1Bin
from mystic.monitors import Monitor,VerboseMonitor
from densitymatrix import DensityMatrix, general_unitary, apply_unitary_to_dm, fid,z_rotation
from processmatrix import conversion, P_matrix, ChannelCJ, real_to_complex
from constants import *



class Results:
    def __init__(self, params, qubit_number,data_points, function, *fargs):
        self.params = params
        self.data_points = data_points
        self.function = function
        self.fargs = fargs
        self.qubit_number = qubit_number

    def minimum(self):
        return self.function(self.params,self.qubit_number, self.data_points, *self.fargs)


"""
Functions needed for the quantum state tomography
"""
"""
function_fidelity returns the optimized fidelity of our density matrix to a pure state
This is to find what unitary, U2, would need to be applied to our state to get max fidelity
and what fidelity that would be

class FidelityResults has the respective useful properties we might want to reconver
"""
def function_fidelity_U4(x, qubit_number, dm, bell):
    # This function should work both for the Bell state and the GHZ, IF we want to optimize on all players
    U1 = general_unitary(x[0:3])
    U2 = general_unitary(x[3:6])
    U3 = general_unitary(x[6:9])
    U4 = general_unitary(x[9:12])
    P = np.kron(np.kron(np.kron(U1,U2),U3),U4)

    return -np.abs(dm.apply_unitary(P).fidelity(bell))

def function_fidelity_U2(x, qubit_number, dm, bell):
    U1=general_unitary(x[0:3])
    U2=general_unitary(x[3:6])
    P = np.kron(U1,U2)

    return -np.abs(dm.apply_unitary(P).fidelity(bell))

def function_fidelity_U1_diag(x, qubit_number, dm, bell):
    U1=general_unitary(x[0:3])
    U2=np.diag([1,1])
    P = np.kron(U1,U2)

    return -np.abs(dm.apply_unitary(P).fidelity(bell))

def function_fidelity_U2_diag(x, qubit_number, dm, bell):
    U1=np.diag([1,1])
    U2=general_unitary(x[0:3])
    P = np.kron(U1,U2)

    return -np.abs(dm.apply_unitary(P).fidelity(bell))

def function_fidelity_Rz(x, qubit_number, dm, bell):
    U1=z_rotation(x)
    U2=np.diag([1,1])
    P = np.kron(U1,U2)
    if qubit_number > 2:
        U3=np.diag([1,1])
        P = np.kron(np.kron(U1,U2),U3) 
        if qubit_number > 3:
            U4=np.diag([1,1])
            P = np.kron(np.kron(np.kron(U1,U2),U3),U4)

    return -np.abs(dm.apply_unitary(P).fidelity(bell))

class FidelityResults(Results):
    @property
    def u1(self):
        #return np.diag([1,1])
        return general_unitary(self.params[0:3])
    
    @property
    def u1_Rz(self):
        return z_rotation(self.params)
    
    @property
    def u2(self):
        #return np.diag([1,1])
        #return general_unitary(self.params[0:3])
        return general_unitary(self.params[3:6])
    
    @property
    def u3(self):
        return general_unitary(self.params[6:9])
    
    @property
    def u4(self):
        # return np.diag([1,1])
        return general_unitary(self.params[9:12])

    @property
    def optimized_state(self):
        return self.data_points.apply_unitary(self.u)

    @property
    def u(self):
        u=np.kron(self.u1,self.u2)
        if self.qubit_number > 2:
            u=np.kron(u,self.u3)
            if self.qubit_number > 3:
                u=np.kron(u,self.u4)
        return u
    
    @property
    def u1_id(self):
        self.u2 = np.diag([1,1])
        u=np.kron(self.u1,self.u2)
        if self.qubit_number > 2:
            u=np.kron(u,self.u3)
            if self.qubit_number > 3:
                u=np.kron(u,self.u4)
        return u
    
    @property
    def u2_id(self):
        self.u1 = np.diag([1,1])
        u=np.kron(self.u1,self.u2)
        if self.qubit_number > 2:
            u=np.kron(u,self.u3)
            if self.qubit_number > 3:
                u=np.kron(u,self.u4)
        return u
    
    
    @property
    def u_Rz(self):
        u=np.kron(self.u1_Rz,np.diag([1,1]))
        if self.qubit_number > 2:
            u=np.kron(u,np.diag([1,1]))
            if self.qubit_number > 3:
                u=np.kron(u,np.diag([1,1]))
        return u
    
    @property
    def optimized_state_Rz(self):
        return self.data_points.apply_unitary(self.u_Rz)

    def Density(self,fock_state):
        return(DensityMatrix(fock_state))
"""
Functions needed for the quantum process tomography
"""
"""
function_process returns the minimum of a cost function that quantifies the least sqaures difference between the experimental # of counts we meansured - data_points -
and the # of counts we should have measured theorically - repetitions*soma_f(X, oput, rhoIn) (number of runs times the probability of rhoIn colapsing into oput, 
considering rhoIn goes through channel X - that is the parameter to be optimized - and is measured in oput)

class ProcessResults has the respective useful properties we might want to reconver
"""
def function_process(X, data_points, input_number, mbasis_number, oput, rhoIn, repetitions):
    f_min=0
    counts=np.reshape(data_points, (input_number, mbasis_number)) #This reshapes counts to [j][o], j is input (order: V, H, D, R, A, L) and o is measurement basis (order: D R V A L H)
    X_real, X_imag = conversion(X)
    
    for j in range(input_number):
        for o in range(mbasis_number):
            # Defining n as a probability of measurement outcome
            nab=counts[j][o]
            soma=0+0j
            for m in range(4):
                for n in range(4):
                    t=n%4+4*(m%4)
                    soma += (X_real[t]+1j*X_imag[t])*np.conjugate(oput[o])@Pauli[m]@rhoIn[j]@Pauli[n]@np.transpose(oput[o])
            soma_f = np.abs(soma[0][0])
            f_min += ((nab-soma_f*repetitions)**2)/float(nab)
    return f_min

class ProcessResults(Results):
    @property
    def chi_matrix(self):
        return real_to_complex(self.params)

    @property
    def chi_matrix_normalized(self):
        return real_to_complex(self.params)/float(np.max(np.real(np.linalg.eig(P_matrix(self.params))[0])))

    # @property
    # def trace(self):
    #     return (np.trace(P_matrix(self.chi_matrix_normalized)))

    # @property
    # def eigenvalue(self):
    #     return np.linalg.eig(P_matrix(self.chi_matrix_normalized))[0]




"""
Functions needed for the optimization to find the unitary closest to the process matrix
"""
"""
function_unitary returns the (negative) maximum fidelity we can find between rho_jE and a unitary U (to be optimized) applied to the Bell state
it is used to find the closest unitary to a given channel, according to the Choi–Jamiołkowski isomorphism.

class UnitaryResults has the respective useful properties we might want to reconver
"""
def function_unitary(x, channel, bellmatrix):
    rho_jE=DensityMatrix(ChannelCJ(channel, bellmatrix))
    U=general_unitary(x)

    return -np.abs(rho_jE.fidelity(apply_unitary_to_dm(bellmatrix, np.kron(Pauli[0],U))))


class UnitaryResults(Results):
    @property
    def unitary(self):
        return self.params



"""
class Optimizer to find the minimum of a certain function, according to the initial_guess, the bounds, penalty and data_points

we can use the functions defined above for an optimization process and recover the paramaters associated with the minimum with the recpective classes
"""
class Optimizer:
### Once we have a state in density matrix form we want to calculate different things with it
### With this class we want to be able to easily calculate all these quantities regarding our state

    def __init__(self, initial_guess, function, results=Results):
        self.initial_guess = initial_guess
        self.function = function
        self.results = results


    def optimize(self, qubit_number, data_points, *fargs,  bounds=None, penalty=None):
        params=diffev2(self.function, self.initial_guess, args=(qubit_number,data_points, *fargs),penalty=penalty, strategy=Best1Bin, bounds=bounds, npop=50, gtol=100, disp=False, ftol=1e-8, handler=False)#, itermon=VerboseMonitor(50))
        return self.results(params, qubit_number,data_points, self.function, *fargs)
        
    def optimize_fmin(self, qubit_number, data_points, *fargs,  bounds=None, penalty=None):
        params=fmin(self.function, self.initial_guess, args=(qubit_number,data_points, *fargs),penalty=penalty, bounds=bounds, disp=False, ftol=1e-18,xtol=1e-18, handler=False)#, itermon=VerboseMonitor(50))
        return self.results(params, qubit_number,data_points, self.function, *fargs)